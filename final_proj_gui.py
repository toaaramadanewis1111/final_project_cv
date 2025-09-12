import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

# ======================
# Paths
# ======================
landmark_model_path = r"C:\Users\mdry\Desktop\compvispr1\egypt_landmarks_cnn.keras"
hiero_model_path = r"C:\Users\mdry\Desktop\compvispr1\hieroglyphs_cnn.h5"
hiero_labels_path = r"C:\Users\mdry\Desktop\compvispr1\hieroglyphs_labels.txt"

# ======================
# Load Models
# ======================
if not os.path.exists(landmark_model_path) or not os.path.exists(hiero_model_path):
    raise FileNotFoundError("❌ One of the models was not found!")

landmark_model = tf.keras.models.load_model(landmark_model_path)
hiero_model = tf.keras.models.load_model(hiero_model_path)

# ======================
# Load Hieroglyph labels
# ======================
hiero_labels = []
with open(hiero_labels_path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        idx, label = line.strip().split(":")
        hiero_labels.append(label)

# ======================
# Landmarks classes
# ======================
landmark_classes = [
    'Abdeen_Palace','Dream_Park','Edfu_Temple','Egyptian_Geological_Museum',
    'Egyptian_Museum_(Cairo)','Egyptian_National_Library','Egyptian_National_Military_Museum',
    'El_Alamein_Military_Museum','Grand_Egyptian_Museum','Great_Hypostyle_Hall_of_Karnak',
    'Great_Pyramid_of_Giza','Great_Sphinx_of_Giza','Great_Temple_of_the_Aten',
    'Greco-Roman_Museum,_Alexandria','Greek_Orthodox_Cathedral_of_Evangelismos,_Alexandria',
    'Saqqara','Siwa','WV22'
]

# ======================
# GUI
# ======================
root = tk.Tk()
root.title("Egypt Landmarks & Hieroglyphs Classifier")
root.geometry("750x650")
root.configure(bg="#F4EBD0")  # Pharaonic color

panel = tk.Label(root, bg="#F4EBD0")
panel.pack(pady=20)

# Labels for results
landmark_result_label = tk.Label(root, text="🏛 Landmark Prediction", font=("Arial", 14), bg="#F4EBD0")
landmark_result_label.pack(pady=5)

hiero_result_label = tk.Label(root, text="✒️ Hieroglyph Prediction", font=("Arial", 14), bg="#F4EBD0")
hiero_result_label.pack(pady=5)

# ======================
# Preprocessing functions
# ======================
def preprocess_landmark(img):
    img = img.resize((224,224))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

def preprocess_hiero(img):
    img = img.resize((64,64))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

# ======================
# Functions for buttons
# ======================
def predict_landmark():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    img = Image.open(file_path).convert("RGB")
    img_input = preprocess_landmark(img)
    preds = landmark_model.predict(img_input)
    idx = np.argmax(preds)

    img_tk = ImageTk.PhotoImage(img.resize((350,350)))
    panel.config(image=img_tk)
    panel.image = img_tk

    landmark_result_label.config(
        text=f"🏛 Landmark: {landmark_classes[idx]}"
    )
    hiero_result_label.config(text="✒️ Hieroglyph Prediction")

def predict_hiero():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    img = Image.open(file_path).convert("RGB")
    img_input = preprocess_hiero(img)
    preds = hiero_model.predict(img_input)
    idx = np.argmax(preds)

    img_tk = ImageTk.PhotoImage(img.resize((350,350)))
    panel.config(image=img_tk)
    panel.image = img_tk

    hiero_result_label.config(
        text=f"✒️ Hieroglyph: {hiero_labels[idx]}"
    )
    landmark_result_label.config(text="🏛 Landmark Prediction")

# ======================
# Buttons
# ======================
btn_landmark = tk.Button(root, text="🏛 Predict Landmark", command=predict_landmark, font=("Arial", 12), bg="#C59D5F")
btn_landmark.pack(pady=10)

btn_hiero = tk.Button(root, text="✒️ Predict Hieroglyph", command=predict_hiero, font=("Arial", 12), bg="#7BAE7F")
btn_hiero.pack(pady=10)

root.mainloop()
