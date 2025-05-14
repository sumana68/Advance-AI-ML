
import argparse
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from ultralytics import YOLO
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to the input CT image")
args = parser.parse_args()
img_path = args.image

# Classification with YOLO 
print("\n🔍 Running YOLOv8 for tumor classification...")
yolo_model = YOLO("yolov8_tumor.pt") 
yolo_results = yolo_model.predict(img_path, conf=0.4)
tumor_detected = len(yolo_results[0].boxes) > 0

yolo_annotated = yolo_results[0].plot()
yolo_img = Image.fromarray(yolo_annotated[..., ::-1])
plt.imshow(yolo_img)
plt.title("YOLOv8 Classification Result")
plt.axis("off")
plt.show()

if not tumor_detected:
    print("Prediction: Normal — No tumor detected.")
    exit()
else:
    print("Prediction: Tumor detected — Running segmentation...")

# Residual U-Net 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class UNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU()
            )

        self.enc1 = ResidualBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

# Segmentation with ResUNet 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

model = UNet(num_classes=3).to(device)
model.load_state_dict(torch.load("resunet_segmentation.pth", map_location=device))
model.eval()

img = Image.open(img_path).convert("L").resize((128, 128))
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

liver_pixels = np.sum(pred_mask == 1)
tumor_pixels = np.sum(pred_mask == 2)

if liver_pixels == 0:
    severity = "No liver detected"
    ratio = 0.0
else:
    ratio = tumor_pixels / liver_pixels
    if ratio > 0.15:
        severity = "Severe"
    elif ratio > 0.05:
        severity = "Moderate"
    else:
        severity = "Mild"

print(f" {os.path.basename(img_path)} → Liver Pixels: {liver_pixels}, Tumor Pixels: {tumor_pixels}, Ratio: {ratio:.3f}, Severity: {severity}")

img_rgb = np.stack([np.array(img)] * 3, axis=-1)
overlay = np.zeros_like(img_rgb)
overlay[pred_mask == 1] = [0, 255, 0]     # liver → green
overlay[pred_mask == 2] = [255, 0, 0]     # tumor → red

blended = (img_rgb * 0.6 + overlay * 0.4).astype(np.uint8)

plt.figure(figsize=(5, 5))
plt.imshow(blended)
plt.title(f"Segmentation Output\nSeverity: {severity}")
plt.axis("off")
plt.show()
