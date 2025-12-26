import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from flask import Flask, request, render_template, jsonify


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/best_contrastive_model_caltech.pt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


OOD_THRESHOLD = 1.3049

os.makedirs(UPLOAD_FOLDER, exist_ok=True)



class ProjectionHead(nn.Module):
    def __init__(self, in_features, projection_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features, bias=False),
            nn.ReLU(),
            nn.Linear(in_features, projection_dim, bias=False)
        )

    def forward(self, x):
        return self.projection(x)


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



print("Dang tai model...")

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
    # Load state dict
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("Model da tai thanh cong!")
except Exception as e:
    print(f"LOI LOAD MODEL: {e}")

# Transform cho ảnh đầu vào (Giống hệt phần Val/Test)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Xử lý ảnh
        try:
            img = Image.open(filepath).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits, _, projected_features = model(img_tensor, return_features=True)

                # 1. Tính OOD Score (Feature Norm)
                ood_score = torch.norm(projected_features, p=2, dim=1).item()

                # 2. Tính xác suất lớp (Classification)
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
                top5_prob, top5_idx = torch.topk(probs, 5)

                top5_classes = [CLASS_NAMES[idx] for idx in top5_idx.cpu().numpy()]
                top5_values = [round(p.item() * 100, 2) for p in top5_prob.cpu().numpy()]

            # Logic xác định ID hay OOD
            is_ood = ood_score > OOD_THRESHOLD
            result_label = "OOD - Vật thể lạ / Chưa biết" if is_ood else f"ID - {top5_classes[0]}"

            return jsonify({
                'image_url': filepath,
                'ood_score': round(ood_score, 4),
                'threshold': OOD_THRESHOLD,
                'is_ood': is_ood,
                'result_label': result_label,
                'top5_classes': top5_classes,
                'top5_probs': top5_values
            })

        except Exception as e:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
