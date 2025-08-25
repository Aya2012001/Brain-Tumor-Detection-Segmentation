import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_detection_model():
    model_path = hf_hub_download(
        repo_id="Aya2012001/Brain_Tumor_Detection_Segmentation",
        filename="brain_tumor_detection_best_model.keras"
    )
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_resource
def load_segmentation_model():
    model_path = hf_hub_download(
        repo_id="Aya2012001/Brain_Tumor_Detection_Segmentation",
        filename="Brain_Tumor_Segmentation_UNet.keras"
    )
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_for_detection(image: Image.Image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_for_segmentation(image: Image.Image):
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("🧠 Brain Tumor Detection & Segmentation")
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    st.subheader("Step 1: Detection")
    detection_model = load_detection_model()
    processed_det = preprocess_for_detection(image)
    prediction = detection_model.predict(processed_det)[0]

    if prediction[0] > 0.5:
        st.error("⚠️ Tumor Detected")

        st.subheader("Step 2: Segmentation")
        seg_model = load_segmentation_model()
        processed_seg = preprocess_for_segmentation(image)
        mask = seg_model.predict(processed_seg)[0]

        mask = (mask.squeeze() > 0.5).astype(np.uint8) * 255

        st.image(mask, caption="Segmentation Mask", use_column_width=True, clamp=True)

    else:
        st.success("✅ No Tumor Detected")
