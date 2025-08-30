# train_cnn.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMG_SIZE = 28
BATCH_SIZE = 32
DEFAULT_EPOCHS = 10

def train_cnn(processed_dir="processed_chars", epochs=DEFAULT_EPOCHS, save_model_path="cnn_license_char_recognition.h5", save_labels_path="labels.npy"):
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Folder processed chars not found: {processed_dir}")

    train_datagen = ImageDataGenerator(rescale=1.0/255)
    train_generator = train_datagen.flow_from_directory(
        processed_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=epochs)

    model.save(save_model_path)
    np.save(save_labels_path, train_generator.class_indices)

    return {
        "epochs": epochs,
        "final_accuracy": float(history.history['accuracy'][-1]),
        "final_loss": float(history.history['loss'][-1]),
        "model_path": save_model_path,
        "labels_path": save_labels_path
    }

if __name__ == "__main__":
    print("Bắt đầu huấn luyện (mặc định)...")
    res = train_cnn()
    print("Kết quả:", res)
