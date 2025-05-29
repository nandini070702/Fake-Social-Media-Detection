import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image  

# Load ResNet50 Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights="IMAGENET1K_V1").to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define paths
data_path = "datasets/image_dataset/face_data"
output_path = "datasets/image_dataset/processed_faces"

os.makedirs(output_path, exist_ok=True)

image_features = []
labels = []
total_processed = 0

# Process images
for label in ["real", "fake"]:
    folder_path = os.path.join(data_path, label)

    if not os.path.exists(folder_path):  
        print(f"⚠️ Warning: Folder not found - {folder_path}")
        continue

    num_images = 0  
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model(img).cpu().numpy().flatten()
                image_features.append(feature)
                labels.append(0 if label == "fake" else 1)

            num_images += 1
            total_processed += 1

        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            continue

    print(f"✔ Processed {num_images} images from {folder_path}")  

# Save dataset
np.save(os.path.join(output_path, "face_features.npy"), np.array(image_features))
np.save(os.path.join(output_path, "face_labels.npy"), np.array(labels))

print(f"✅ Face Data Processing Complete! {total_processed} images processed.")
