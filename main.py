import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import base64

landmark_model_path = r"C:\Users\mdry\Desktop\compvispr1\egypt_landmarks_cnn.keras"
hiero_model_path = r"C:\Users\mdry\Desktop\compvispr1\hieroglyphs_cnn.h5"
hiero_labels_path = r"C:\Users\mdry\Desktop\compvispr1\hieroglyphs_labels.txt"

if not os.path.exists(landmark_model_path) or not os.path.exists(hiero_model_path):
    raise FileNotFoundError("❌ One of the models was not found!")

landmark_model = tf.keras.models.load_model(landmark_model_path)
hiero_model = tf.keras.models.load_model(hiero_model_path)

with open(hiero_labels_path, "r", encoding="utf-8") as f:
    hiero_labels = [line.strip().split(":")[1] for line in f.readlines()]

landmark_classes = [
    'Abdeen_Palace','Dream_Park','Edfu_Temple','Egyptian_Geological_Museum',
    'Egyptian_Museum_(Cairo)','Egyptian_National_Library','Egyptian_National_Military_Museum',
    'El_Alamein_Military_Museum','Grand_Egyptian_Museum','Great_Hypostyle_Hall_of_Karnak',
    'Great_Pyramid_of_Giza','Great_Sphinx_of_Giza','Great_Temple_of_the_Aten',
    'Greco-Roman_Museum,_Alexandria','Greek_Orthodox_Cathedral_of_Evangelismos,_Alexandria',
    'Saqqara','Siwa','WV22'
]

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

background_image_path = r"C:\Users\mdry\Downloads\0801-0336_egypt-xlarge.jpg"
background_base64 = get_base64_of_bin_file(background_image_path)

st.set_page_config(page_title="Pharaonic Vision 🔺", page_icon="🏺", layout="centered")

st.markdown(f"""
<style>
body {{
    background-image: url("data:image/jpg;base64,{background_base64}");
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-position: center;
    color: #4b3b2a;
    font-family: 'Georgia', serif;
}}
h1, h2, h3 {{
    color: #b6893b;
    text-shadow: 1px 1px 2px #3c2e1a;
    text-align: center;
    background: rgba(248,240,216,0.75);
    border-radius: 12px;
    padding: 8px;
}}
.stApp {{
    background: rgba(255, 255, 255, 0.80);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 0 25px rgba(0,0,0,0.3);
}}
.stButton>button {{
    background-color: #b6893b;
    color: white;
    border-radius: 12px;
    border: 2px solid #704c1e;
    font-size: 16px;
}}
.stButton>button:hover {{
    background-color: #d4a24c;
    color: #fff8dc;
}}
</style>
""", unsafe_allow_html=True)

st.title("🏺 **Pharaonic Vision: Egyptian Classifier**")
st.markdown("""
Welcome to **Pharaonic Vision** —  
A deep learning app that can recognize **Egyptian Landmarks** and **Hieroglyphic Symbols** 🏛️✒️  
Upload an image and let the ancient magic of AI reveal its identity 🌞.
""")

uploaded_file = st.file_uploader("📜 Upload your image (JPG, PNG)", type=["jpg", "jpeg", "png"])
model_choice = st.radio("🔮 Choose the Model", ["Landmark 🏛", "Hieroglyph ✒️"])

def preprocess_landmark(img):
    img = img.resize((224,224))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

def preprocess_hiero(img):
    img = img.resize((64,64))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼 Uploaded Image", use_column_width=True)

    if model_choice.startswith("Landmark"):
        img_input = preprocess_landmark(img)
        preds = landmark_model.predict(img_input)
        idx = np.argmax(preds)
        st.success(f"🏛 **Landmark Identified:** {landmark_classes[idx]}")
    
    elif model_choice.startswith("Hieroglyph"):
        img_input = preprocess_hiero(img)
        preds = hiero_model.predict(img_input)
        idx = np.argmax(preds)
        st.success(f"📜 **Hieroglyph Symbol:** {hiero_labels[idx]}")

st.markdown("<hr><center>🌅 Made with ❤️ and Ancient Egyptian Spirit</center>", unsafe_allow_html=True)
