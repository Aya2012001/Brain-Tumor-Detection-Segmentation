import streamlit as st
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

st.set_page_config(page_title="Brain Tumor Detection & Segmentation", page_icon="🧠", layout="centered")
st.title("Brain Tumor Detection & Segmentation")


@st.cache_resource
def load_classifier():
    return load_model("brain-tumor-detection-using-resnet50.keras")

@st.cache_resource
def load_segmenter():
    return load_model("brain-tumor-segmentation-unet.keras", compile=False)

classifier = load_classifier()
segmenter = load_segmenter()


def preprocess_for_classifier(img: Image.Image, target_size=(224, 224)):
    img = img.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def preprocess_for_segmenter(img: Image.Image, target_size=(128, 128)):
    img = img.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_tumor(img):
    arr = preprocess_for_classifier(img)
    prob = classifier.predict(arr)[0][0]
    return prob

def segment_tumor(img):
    arr = preprocess_for_segmenter(img)
    mask = segmenter.predict(arr)[0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (img.width, img.height))
    return mask

def overlay_mask(img: Image.Image, mask):
    img_arr = np.array(img.convert("RGB"))
    mask_colored = np.zeros_like(img_arr)
    mask_colored[:, :, 0] = mask  
    overlay = cv2.addWeighted(img_arr, 0.7, mask_colored, 0.3, 0)
    return overlay


uploaded_file = st.file_uploader("Upload MRI (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(img, caption="Uploaded image", use_container_width=True)

    st.write("")
    prob = predict_tumor(img)

    if prob >= 0.5:
        st.success(f"Tumor Detected (Probability: {prob:.2f})")
        st.write("Doing Segmentation ...")
        mask = segment_tumor(img)
        overlay = overlay_mask(img, mask)
        st.image(overlay, caption="Result with Tumor Segmentation", use_container_width=True)
    else:
        st.info(f"No Tumor Detected (Probability: {prob:.2f})")
