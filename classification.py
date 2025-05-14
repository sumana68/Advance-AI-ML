# classifier_pipeline.py

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define CNN model (matching training architecture)
class TumorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = TumorClassifier().to(device)
model.load_state_dict(torch.load("tumor_classifier.pth", map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Image folder
image_folder = "slices/test"
all_imgs = []

for fname in sorted(os.listdir(image_folder)):
    if fname.endswith(".png"):
        label = "Tumor" if "tumor" in fname else "Normal"
        all_imgs.append((os.path.join(image_folder, fname), label))

# Visualize predictions
plt.figure(figsize=(20, 10))
correct = 0

for idx, (img_path, true_label) in enumerate(all_imgs[:20]):
    img = Image.open(img_path).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = output.item()
        pred_label = "Tumor" if prob > 0.5 else "Normal"

    if pred_label == true_label:
        correct += 1

    plt.subplot(4, 5, idx + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Pred: {pred_label} ({prob:.2f})\nActual: {true_label}")
    plt.axis("off")

plt.tight_layout()
plt.show()

print(f"\n✅ Accuracy on 20 random samples: {correct}/20 ({(correct/20)*100:.1f}%)")