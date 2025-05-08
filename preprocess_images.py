import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# === Configurations ===
DATASET_DIR = "Plant_disease_dataset"
IMAGE_SIZE = (128, 128)  # Change to (224, 224) if you want to use models like ResNet
OUTPUT_FILE = "processed_plant_disease_dataset.pkl"

# === Lists to hold data ===
images = []
labels = []

# === Traverse folders ===
for crop in os.listdir(DATASET_DIR):
    crop_path = os.path.join(DATASET_DIR, crop)
    if os.path.isdir(crop_path):
        for disease in os.listdir(crop_path):
            disease_path = os.path.join(crop_path, disease)
            if os.path.isdir(disease_path):
                for img_name in os.listdir(disease_path):
                    img_path = os.path.join(disease_path, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(IMAGE_SIZE)
                        images.append(np.array(img))
                        labels.append(f"{crop}___{disease}")
                    except Exception as e:
                        print(f"Skipping {img_path}: {e}")

# === Encode labels ===
le = LabelEncoder()
y = le.fit_transform(labels)
X = np.array(images)

# === Optionally split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Save ===
joblib.dump({
    "X_train": X_train,
    "X_val": X_val,
    "y_train": y_train,
    "y_val": y_val,
    "label_encoder": le
}, OUTPUT_FILE)

print(f"✅ Preprocessing complete. Data saved to {OUTPUT_FILE}")
