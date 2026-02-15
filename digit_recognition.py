import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------- 1. Dataset Path -----------
data_dir = "capture handwritten"  # <-- Change this!
img_size = 28

X = []
y = []

# --------- 2. Load Images -----------
for idx, label in enumerate(sorted(os.listdir(data_dir))):
    folder_path = os.path.join(data_dir, label)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        if os.path.isdir(img_path) or not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping broken image: {img_path}")
            continue

        img = cv2.resize(img, (img_size, img_size))
        X.append(img)
        y.append(int(label))

# --------- 3. Preprocess -----------
X = np.array(X).reshape(-1, img_size, img_size, 1).astype('float32') / 255.0
y = to_categorical(np.array(y))

print("X shape:", X.shape)
print("y shape:", y.shape)

# --------- 4. Train/Test Split -----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------- 5. Build CNN Model -----------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')  # dynamic classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --------- 6. Train Model -----------
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# --------- 7. Plot Training Accuracy -----------
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training Curve")
plt.show()

# --------- 8. Save Model -----------
model.save("digit_model.h5")
print("✅ Model saved as digit_model.h5")

# --------- 9. Prediction Function -----------
def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found or can't be opened.")
        return
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)
    print(f"✅ Predicted Digit: {predicted_label} with confidence {confidence:.2f}")
