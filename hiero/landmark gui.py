import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

# Path to the model
model_path = r"C:\Users\mdry\Desktop\compvispr1\egypt_landmarks_cnn.h5"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at {model_path}")

# Load the trained model
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully")

# Define class names (make sure these match your training set folders order)
class_names = [
    'Abdeen_Palace',
    'Dream_Park',
    'Edfu_Temple',
    'Egyptian_Geological_Museum',
    'Egyptian_Museum_(Cairo)',
    'Egyptian_National_Library',
    'Egyptian_National_Military_Museum',
    'El_Alamein_Military_Museum',
    'Grand_Egyptian_Museum',
    'Great_Hypostyle_Hall_of_Karnak',
    'Great_Pyramid_of_Giza',
    'Great_Sphinx_of_Giza',
    'Great_Temple_of_the_Aten',
    'Greco-Roman_Museum,_Alexandria',
    'Greek_Orthodox_Cathedral_of_Evangelismos,_Alexandria',
    'Saqqara',
    'Siwa',
    'WV22'
]

# GUI Window
root = tk.Tk()
root.title("Egypt Landmarks Classifier")
root.geometry("600x500")

# Image display panel
panel = tk.Label(root)
panel.pack(pady=20)

# Result label
result_label = tk.Label(root, text="📷 Select an image to predict", font=("Arial", 14))
result_label.pack(pady=10)

# Preprocess function
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

# Load and predict
def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    # Preprocess
    img_array, img = preprocess_image(file_path)

    # Prediction
    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    confidence = np.max(preds)

    # Show image
    img_tk = ImageTk.PhotoImage(img.resize((300, 300)))
    panel.config(image=img_tk)
    panel.image = img_tk

    # Show result
    landmark_name = class_names[class_index] if class_index < len(class_names) else "Unknown"
    result_label.config(
        text=f"🏛 Landmark: {landmark_name}\n📊 Confidence: {confidence:.2f}"
    )

# Select image button
btn = tk.Button(root, text="📂 Select Image", command=load_image, font=("Arial", 12), bg="lightblue")
btn.pack(pady=20)

root.mainloop()
