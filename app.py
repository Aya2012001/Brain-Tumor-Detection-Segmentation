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
    model = tf.keras.models.load_model(model_path, compile=False)  # 👈 important
    return model

@st.cache_resource
def load_segmentation_model():
    model_path = hf_hub_download(
        repo_id="Aya2012001/Brain_Tumor_Detection_Segmentation",
        filename="Brain_Tumor_Segmentation_UNet.keras"
    )
    model = tf.keras.models.load_model(model_path, compile=False)  # 👈 important
    return model

def preprocess_for_detection(image: Image.Image):
    img = image.resize((224, 224))  
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_for_segmentation(image: Image.Image):
    img = image.resize((128, 128)) 
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_ar_
