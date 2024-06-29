import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model("audio.h5")


def preprocess_audio(audio_file):
    y, sr = librosa.load(audio_file)
    D = librosa.stft(y)
    D_db = librosa.amplitude_to_db(abs(D))

    # Convert spectrogram to image
    img = Image.fromarray(D_db)

    # Resize the image
    resized_img = img.resize((256, 256))

    # Convert the resized image back to array
    resized_spectrogram = np.array(resized_img)

    # Expand dimensions to make it (height, width, channels)
    resized_spectrogram = np.expand_dims(resized_spectrogram, axis=-1)

    return resized_spectrogram


def main():
    st.title("Deepfake Audio Detection")

    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        spectrogram = preprocess_audio(uploaded_file)
        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                # Perform prediction
                prediction = model.predict(np.expand_dims(spectrogram, axis=0))
                class_names = ["Fake", "Real"]
                predicted_class = class_names[np.argmax(prediction)]
                # Display prediction
                st.write("Predicted Class:", predicted_class)


if __name__ == "__main__":
    main()

# # Streamlit app
# st.title("Audio Classification")
# st.sidebar.title("Upload Audio File")

# # Upload audio file
# uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=["wav"])

# if uploaded_file is not None:
#     st.audio(uploaded_file, format="audio/wav")

#     # Preprocess the audio file
#     spectrogram = preprocess_audio(uploaded_file)

#     # Perform prediction
#     prediction = model.predict(np.expand_dims(spectrogram, axis=0))
#     class_names = ["Fake", "Real"]
#     predicted_class = class_names[np.argmax(prediction)]

#     # Display prediction
#     st.write("Predicted Class:", predicted_class)
