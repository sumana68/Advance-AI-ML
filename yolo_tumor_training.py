#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Step 1: Install gdown
get_ipython().system('pip install gdown')

# Step 2: Download your shared dataset from Google Drive
import gdown

file_id = "1iK7D-WTWx7qscAc5-9-yZMb6Aav_K6lL"
gdown.download(f"https://drive.google.com/uc?id={file_id}", output="DATASET.zip", quiet=False)

# Step 3: Unzip the dataset
import zipfile
with zipfile.ZipFile("DATASET.zip", 'r') as zip_ref:
    zip_ref.extractall("liver_dataset")

print("✅ Dataset is ready at: /content/liver_dataset")


# In[3]:


import os

# Set your base project path (change this if needed)
base_path = "/content/drive/MyDrive/liver_dataset"

# Define required subfolders
folders = [
    "images/train",
    "images/val",
    "images/test",
    "labels/train",
    "labels/val",
    "labels/test",
    "cropped_tumors"
]

# Create each folder
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    os.makedirs(folder_path, exist_ok=True)
    print(f"✅ Created: {folder_path}")


# In[4]:


import os
import random
import shutil

# Set your source and destination paths
source_images = "/content/drive/MyDrive/liver_dataset/images"   # change this
source_labels = "/content/drive/MyDrive/liver_dataset/labels"   # change this
dest_base = "/content/drive/MyDrive/liver_dataset"

# Define ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Collect all image filenames
all_images = [f for f in os.listdir(source_images) if f.endswith(".png") or f.endswith(".jpg")]
random.shuffle(all_images)

# Split
total = len(all_images)
train_files = all_images[:int(train_ratio * total)]
val_files = all_images[int(train_ratio * total):int((train_ratio + val_ratio) * total)]
test_files = all_images[int((train_ratio + val_ratio) * total):]

# Helper function
def copy_files(file_list, split):
    for filename in file_list:
        # Image file
        img_src = os.path.join(source_images, filename)
        img_dst = os.path.join(dest_base, f"images/{split}", filename)
        shutil.copy2(img_src, img_dst)

        # Label file
        label_file = os.path.splitext(filename)[0] + ".txt"
        lbl_src = os.path.join(source_labels, label_file)
        lbl_dst = os.path.join(dest_base, f"labels/{split}", label_file)
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, lbl_dst)

# Copy to folders
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("✅ Dataset split complete.")


# In[5]:


pip install nibabel opencv-python tqdm


# In[6]:


import os

search_path = "/content/drive/MyDrive/liver_dataset"

for root, dirs, files in os.walk(search_path):
    for file in files:
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            print(os.path.join(root, file))


# In[7]:


import nibabel as nib
import numpy as np
import cv2
import os
from tqdm import tqdm

# Path to .nii files
nii_folder = "/content/liver_dataset/volume_pt1"

# Where to save .png slices
output_folder = "/content/drive/MyDrive/liver_dataset/images/raw_slices"
os.makedirs(output_folder, exist_ok=True)

for file in tqdm(os.listdir(nii_folder)):
    if file.endswith(".nii") or file.endswith(".nii.gz"):
        path = os.path.join(nii_folder, file)
        img = nib.load(path)
        data = img.get_fdata()

        for i in range(data.shape[2]):  # loop over axial slices
            slice_img = data[:, :, i]
            norm_img = ((slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img)) * 255).astype(np.uint8)
            save_path = os.path.join(output_folder, f"{file[:-4]}_slice_{i:03d}.png")
            cv2.imwrite(save_path, norm_img)

print("✅ Slices extracted and saved as .png")


# In[9]:


import os

model_path = "/content/drive/MyDrive/liver_dataset/yolov8_tumor/weights/best.pt"
print("✅ Model exists!" if os.path.exists(model_path) else "❌ Model NOT found.")


# In[11]:


slice_folder = "/content/drive/MyDrive/liver_dataset/images/raw_slices"
slices = os.listdir(slice_folder)

print(f"✅ Found {len(slices)} slice images." if slices else "❌ No slices found.")
print("Sample slices:", slices[:5])


# In[12]:


get_ipython().system('unzip yolo_tumor_dataset.zip -d /content/')


# In[16]:


get_ipython().system('unzip /content/yolo_tumor_dataset.zip -d /content/')


# In[22]:


import os

print("📁 /content/yolo_dataset/images/train/:")
print(os.listdir("/content/yolo_dataset/images/train"))

print("\n📁 /content/yolo_dataset/labels/train/:")
print(os.listdir("/content/yolo_dataset/labels/train"))


# In[23]:


import shutil
import os

nested_img_dir = "/content/yolo_dataset/images/train/train"
target_img_dir = "/content/yolo_dataset/images/train"

for file in os.listdir(nested_img_dir):
    shutil.move(os.path.join(nested_img_dir, file), target_img_dir)

os.rmdir(nested_img_dir)


# In[24]:


yaml_content = """
train: /content/yolo_dataset/images/train
val: /content/yolo_dataset/images/train
nc: 1
names: ['tumor']
"""

with open("/content/yolo_dataset/data.yaml", "w") as f:
    f.write(yaml_content.strip())


# In[ ]:


import os
import shutil

train_img_dir = "/content/drive/MyDrive/liver_dataset/images/train"
train_lbl_dir = "/content/drive/MyDrive/liver_dataset/labels/train"
val_img_dir = "/content/drive/MyDrive/liver_dataset/images/val"
val_lbl_dir = "/content/drive/MyDrive/liver_dataset/labels/val"

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# Move 10 samples
moved = 0
for fname in os.listdir(train_img_dir):
    if fname.endswith(".png") or fname.endswith(".jpg"):
        src_img = os.path.join(train_img_dir, fname)
        dst_img = os.path.join(val_img_dir, fname)
        shutil.copy2(src_img, dst_img)

        # move matching label
        lbl_name = os.path.splitext(fname)[0] + ".txt"
        src_lbl = os.path.join(train_lbl_dir, lbl_name)
        dst_lbl = os.path.join(val_lbl_dir, lbl_name)

        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

        moved += 1
        if moved >= 10:
            break

print(f"✅ Copied {moved} images and labels to validation set.")


# In[26]:


get_ipython().system('pip install ultralytics')


# In[30]:


from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # ✅ Pretrained model
model.train(
    data="/content/yolo_dataset/data.yaml",
    epochs=20,
    imgsz=640,
    batch=4,
    name="tumor_detector_augmented",
    fliplr=0.5,
    mosaic=1.0,
    hsv_h=0.05,
    hsv_s=0.7,
    hsv_v=0.4
)


# In[35]:


import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load your trained model
model = YOLO("/content/runs/detect/tumor_detector/weights/best.pt")

# Path to a test image
img_path = "/content/yolo_dataset/images/train/tumor_733.png"  # or any other labeled one

# Run detection
results = model(img_path)
annotated = results[0].plot()

# Convert BGR to RGB for matplotlib
annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# Show image
plt.imshow(annotated)
plt.axis("off")
plt.title("YOLOv8 Tumor Detection")
plt.show()


# In[36]:


with open("/content/yolo_dataset/labels/train/tumor_733.txt") as f:
    print(f.read())


# In[2]:


from ultralytics import YOLO

# Load the trained model (adjust the path if needed)
model = YOLO("/content/runs/detect/tumor_detector_augmented/weights/best.pt")


# In[6]:


results = model.predict(
    source="/content/yolo_dataset/images/train/tumor_733.png",
    conf=0.4,
    show_labels=False,
    show_conf=False

)

# Display
import cv2
import matplotlib.pyplot as plt

img = results[0].plot()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis("off")
plt.title("YOLOv8 Tumor Detection (Conf ≥ 0.4)")
plt.show()


# In[14]:


import cv2
import matplotlib.pyplot as plt

# Load original image
img_path = "/content/yolo_dataset/images/train/tumor_733.png"
original = cv2.imread(img_path)
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# Make a copy for drawing detections
detected_img = original_rgb.copy()

# Run YOLO prediction
results = model.predict(source=img_path, conf=0.4)
detections = results[0].boxes

# Draw boxes manually with smaller font
for box in detections:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    conf = float(box.conf[0])
    label = f"tumor {conf:.2f}"

    cv2.rectangle(detected_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(detected_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)

# Plot side-by-side
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(detected_img)
plt.title("YOLOv8 Detection (Smaller Font)")
plt.axis("off")

plt.tight_layout()
plt.show()

