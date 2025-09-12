# main.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# ------------------------
# Paths to models
# ------------------------
landmark_model_path = r"C:\Users\mdry\Desktop\compvispr1\egypt_landmarks_cnn.keras"
hiero_model_path = r"C:\Users\mdry\Desktop\compvispr1\hieroglyphs_cnn.h5"
hiero_labels_path = r"C:\Users\mdry\Desktop\compvispr1\hieroglyphs_labels.txt"

# ------------------------
# Load models
# ------------------------
if not os.path.exists(landmark_model_path) or not os.path.exists(hiero_model_path):
    raise FileNotFoundError("❌ One of the models was not found!")

landmark_model = tf.keras.models.load_model(landmark_model_path)
hiero_model = tf.keras.models.load_model(hiero_model_path)

# Load hieroglyph labels
with open(hiero_labels_path, "r", encoding="utf-8") as f:
    hiero_labels = [line.strip().split(":")[1] for line in f.readlines()]

# Landmark class names
landmark_classes = [
    'Abdeen_Palace','Dream_Park','Edfu_Temple','Egyptian_Geological_Museum',
    'Egyptian_Museum_(Cairo)','Egyptian_National_Library','Egyptian_National_Military_Museum',
    'El_Alamein_Military_Museum','Grand_Egyptian_Museum','Great_Hypostyle_Hall_of_Karnak',
    'Great_Pyramid_of_Giza','Great_Sphinx_of_Giza','Great_Temple_of_the_Aten',
    'Greco-Roman_Museum,_Alexandria','Greek_Orthodox_Cathedral_of_Evangelismos,_Alexandria',
    'Saqqara','Siwa','WV22'
]

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Egypt Classifier", page_icon="🏛")
st.title("🏛 Egypt Landmarks & Hieroglyphs Classifier")
st.write("Upload an image and select the model to predict Landmark or Hieroglyph.")

# Upload image
uploaded_file = st.file_uploader("📂 Choose an image", type=["jpg", "jpeg", "png"])

# Choose model
model_choice = st.radio("Select Model", ["Landmark", "Hieroglyph"])

# ------------------------
# Functions
# ------------------------
def preprocess_landmark(img):
    img = img.resize((224,224))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

def preprocess_hiero(img):
    img = img.resize((64,64))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

# ------------------------
# Prediction
# ------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if model_choice == "Landmark":
        img_input = preprocess_landmark(img)
        preds = landmark_model.predict(img_input)
        idx = np.argmax(preds)
        st.success(f"🏛 Landmark: {landmark_classes[idx]}")
    
    elif model_choice == "Hieroglyph":
        img_input = preprocess_hiero(img)
        preds = hiero_model.predict(img_input)
        idx = np.argmax(preds)
        st.success(f"✒️ Hieroglyph: {hiero_labels[idx]}")
