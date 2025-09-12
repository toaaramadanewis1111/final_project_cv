import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ======================
# Paths
# ======================
dataset_dir = r"C:\Users\mdry\Desktop\compvispr1\hiero"   # <-- هنا الفولدر الأساسي للصور
model_path = r"C:\Users\mdry\Desktop\compvispr1\hieroglyphs_cnn.h5"
labels_path = r"C:\Users\mdry\Desktop\compvispr1\hieroglyphs_labels.txt"

# ======================
# Data Generators
# ======================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# ======================
# Model
# ======================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64,64,3)),                 # input layer
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(train_gen.class_indices), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ======================
# Training
# ======================
history = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen
)

# ======================
# Save Model
# ======================
model.save(model_path)
print(f"✅ Model saved at {model_path}")

# Save class labels
with open(labels_path, "w") as f:
    for cls, idx in train_gen.class_indices.items():
        f.write(f"{idx}:{cls}\n")
print(f"✅ Labels saved at {labels_path}")
