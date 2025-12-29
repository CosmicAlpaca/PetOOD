import os
import io
import zipfile
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from flask import Flask, request, render_template, jsonify

# ================= CẤU HÌNH =================
app = Flask(__name__)
MODEL_PATH = 'model/best_contrastive_model_caltech.pt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OOD_THRESHOLD = 1.3049


# ================= MODEL DEFINITION (GIỮ NGUYÊN) =================
class ProjectionHead(nn.Module):
    def __init__(self, in_features, projection_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features, bias=False),
            nn.ReLU(),
            nn.Linear(in_features, projection_dim, bias=False)
        )

    def forward(self, x): return self.projection(x)


class OODResNet18(nn.Module):
    def __init__(self, num_classes=37, projection_dim=128):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection_head = ProjectionHead(self.backbone_dim, projection_dim)
        self.classifier = nn.Linear(self.backbone_dim, num_classes)

    def forward(self, x, return_features=False):
        backbone_features = self.backbone(x)
        logits = self.classifier(backbone_features)
        if return_features:
            projected_features = self.projection_head(backbone_features)
            return logits, backbone_features, projected_features
        return logits


# ================= LOAD MODEL & UTILS =================
print(">>> Dang tai model...")
CLASS_NAMES = [
    'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound',
    'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair',
    'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 'English Setter',
    'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin',
    'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland',
    'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard',
    'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx',
    'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier'
]

model = OODResNet18(num_classes=len(CLASS_NAMES))
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    print(">>> Model da tai thanh cong!")
except Exception as e:
    print(f"LOI LOAD MODEL: {e}")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Hàm xử lý logic dự đoán cho 1 ảnh (byte stream)
def process_single_image(filename, img_bytes):
    try:
        # 1. Chuyển bytes thành PIL Image
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # 2. Predict
        with torch.no_grad():
            logits, _, projected_features = model(img_tensor, return_features=True)
            ood_score = torch.norm(projected_features, p=2, dim=1).item()
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            top5_prob, top5_idx = torch.topk(probs, 5)

            top5_classes = [CLASS_NAMES[idx] for idx in top5_idx.cpu().numpy()]
            top5_values = [round(p.item() * 100, 2) for p in top5_prob.cpu().numpy()]

        is_ood = ood_score > OOD_THRESHOLD
        label = "OOD - Vật thể lạ" if is_ood else f"ID - {top5_classes[0]}"

        # 3. Tạo Base64 string để hiển thị ở Frontend
        # (Nén lại chút cho nhẹ JSON nếu cần, ở đây giữ nguyên)
        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{base64_str}"

        return {
            'filename': filename,
            'ood_score': round(ood_score, 4),
            'threshold': OOD_THRESHOLD,
            'is_ood': is_ood,
            'result_label': label,
            'top5_classes': top5_classes,
            'top5_probs': top5_values,
            'image_data': img_data_url  # Trả về ảnh dạng text
        }
    except Exception as e:
        return {'filename': filename, 'error': str(e)}


# ================= ROUTES =================
@app.route('/')
def home(): return render_template('index.html')


@app.route('/app')
def web_app(): return render_template('analysis.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    uploaded_files = request.files.getlist('files')
    results = []

    for file in uploaded_files:
        if file.filename == '': continue

        # 1. Xử lý file ZIP
        if file.filename.endswith('.zip'):
            try:
                # Đọc file zip từ RAM
                with zipfile.ZipFile(file) as z:
                    for zip_filename in z.namelist():
                        # Lọc file ảnh, bỏ qua thư mục/file ẩn
                        if zip_filename.lower().endswith(('.png', '.jpg', '.jpeg')) and not zip_filename.startswith(
                                '__'):
                            with z.open(zip_filename) as image_file:
                                img_bytes = image_file.read()
                                # Gọi hàm xử lý chung
                                res = process_single_image(zip_filename, img_bytes)
                                results.append(res)
            except Exception as e:
                results.append({'filename': file.filename, 'error': f"Lỗi Zip: {str(e)}"})

        # 2. Xử lý file ảnh thường
        else:
            img_bytes = file.read()
            res = process_single_image(file.filename, img_bytes)
            results.append(res)

    return jsonify({'results': results})


if __name__ == '__main__':
    app.run(debug=True, port=5000)