# app.py
import streamlit as st
from PIL import Image
import numpy as np
from inference import predict

st.title("Deepfake GAN Image Identification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = predict(image)
    if prediction[0][0] > 0.5:
        st.write(f"Prediction: Real")
    else:
        st.write(f"Prediction: Fake")
