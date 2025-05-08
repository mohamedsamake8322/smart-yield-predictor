import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# ===========================
#     Load preprocessed data
# ===========================
DATA_FILE = "processed_plant_disease_dataset.pkl"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("Preprocessed data file not found. Please run preprocess_images.py first.")

X, y, label_encoder = joblib.load(DATA_FILE)

# ===========================
#         Split data
# ===========================
X = X / 255.0  # Normalize images
y_encoded = to_categorical(y, num_classes=len(label_encoder.classes_))

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ===========================
#         Build model
# ===========================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ===========================
#     Train and Save model
# ===========================
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=20, batch_size=32,
          validation_data=(X_val, y_val),
          callbacks=[early_stop])

model.save("plant_disease_cnn_model.h5")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model and label encoder saved.")
