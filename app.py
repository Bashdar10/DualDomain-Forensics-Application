import os
import io
import time
import base64
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================================
# 1. MODEL ARCHITECTURE DEFINITION
# ==========================================================
class SpatialFrequencyFusionNet(nn.Module):
    def __init__(self):
        super(SpatialFrequencyFusionNet, self).__init__()
        # Spatial Branch: EfficientNet-B0
        self.spatial_backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        # Frequency Branch: Swin-Tiny Transformer
        self.freq_backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)

        spatial_dim = self.spatial_backbone.num_features   # 1280
        freq_dim = self.freq_backbone.num_features         # 768
        fused_dim = spatial_dim + freq_dim                # 2048

        # Attention Gating Mechanism
        self.attention_gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 4),
            nn.ReLU(),
            nn.Linear(fused_dim // 4, fused_dim),
            nn.Sigmoid()
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Linear(512, 2)
        )

    def forward(self, x_spatial, x_freq):
        f_spatial = self.spatial_backbone(x_spatial)
        f_freq = self.freq_backbone(x_freq)
        concat_feat = torch.cat([f_spatial, f_freq], dim=1)
        gated_feat = concat_feat * self.attention_gate(concat_feat)
        return self.classifier(gated_feat)

# Load Model Weights
MODEL_PATH = 'best_Spatial_Frequency_Fusion.pth'
model = SpatialFrequencyFusionNet().to(device)

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("✅ Model weights loaded successfully into memory.")
else:
    print(f"⚠️ Warning: '{MODEL_PATH}' not found! Running in evaluation stub mode.")
    model.eval()

# ==========================================================
# 2. IMAGE PREPROCESSING & 2D FFT SPECTRUM
# ==========================================================
spatial_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def compute_fft_representation(pil_img):
    gray = pil_img.convert('L').resize((224, 224), Image.Resampling.LANCZOS)
    img_np = np.array(gray, dtype=np.float32)
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    norm_mag = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-5)
    
    # 3-channel tensor for model forward pass
    img_3ch = np.stack([norm_mag, norm_mag, norm_mag], axis=0)
    tensor_fft = torch.tensor(img_3ch, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Generate base64 visualization for UI rendering
    fig_img = Image.fromarray(np.uint8(norm_mag * 255)).resize((300, 300))
    buffer = io.BytesIO()
    fig_img.save(buffer, format="PNG")
    b64_fft = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return tensor_fft, b64_fft

# ==========================================================
# 3. ROUTES
# ==========================================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model-details')
def model_details():
    return render_template('model_details.html')

@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'})

    try:
        start_time = time.time()
        raw_pil = Image.open(file.stream).convert('RGB')
        
        # 1. Extract Spatial Tensor
        spatial_tensor = spatial_transform(raw_pil).unsqueeze(0).to(device)
        
        # 2. Extract Frequency Tensor & Spectrum Image
        fft_tensor, b64_fft = compute_fft_representation(raw_pil)
        
        # 3. Model Inference
        with torch.no_grad():
            logits = model(spatial_tensor, fft_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        inference_time_ms = round((time.time() - start_time) * 1000, 2)
        
        prob_authentic = float(probs[0])
        prob_synthetic = float(probs[1])
        
        max_prob = max(prob_authentic, prob_synthetic)
        confidence_percent = round(max_prob * 100, 2)
        
        # 4. Out-of-Distribution (OOD) / Low-Confidence Validation (Threshold: 80%)
        CONFIDENCE_THRESHOLD = 80.0
        is_supported_domain = confidence_percent >= CONFIDENCE_THRESHOLD
        
        if is_supported_domain:
            predicted_class = "Authentic (Real Photo)" if prob_authentic > prob_synthetic else "Synthetic (AI-Generated)"
            decision_status = "AUTHENTIC" if prob_authentic > prob_synthetic else "SYNTHETIC"
            message = "High forensic confidence. Sample validated within recognized feature manifolds."
        else:
            predicted_class = "Unrecognized / Out-of-Domain Image"
            decision_status = "UNKNOWN_OOD"
            message = f"Warning: The model confidence ({confidence_percent}%) is below the reliable 80% boundary. This image does not fit the target domain categories (Human Faces, Fauna, Structured Objects)."
        
        # Original Image Base64 for Instant Display
        buf_orig = io.BytesIO()
        raw_pil.resize((300, 300)).save(buf_orig, format="JPEG")
        b64_orig = base64.b64encode(buf_orig.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'is_supported': is_supported_domain,
            'prediction': predicted_class,
            'status': decision_status,
            'confidence': confidence_percent,
            'prob_authentic': round(prob_authentic * 100, 2),
            'prob_synthetic': round(prob_synthetic * 100, 2),
            'latency_ms': inference_time_ms,
            'message': message,
            'orig_img': f"data:image/jpeg;base64,{b64_orig}",
            'fft_img': f"data:image/png;base64,{b64_fft}"
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)