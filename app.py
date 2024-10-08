import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import os
import tensorflow as tf

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    # Resize the image while maintaining aspect ratio
    image.thumbnail(target_size, Image.LANCZOS)

    # Calculate padding to make the image square
    delta_width = target_size[0] - image.size[0]
    delta_height = target_size[1] - image.size[1]
    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

    # Pad the image and make it square
    image = ImageOps.expand(image, padding, fill=(0, 0, 0))  # You can change the fill color as needed

    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# loaded_model = joblib.load('./bird_image_classification.joblib')

model_file = 'bird_image_classification.h5'
loaded_model = None
if not os.path.exists(model_file):
    st.error(f"Model file '{model_file}' not found.")
else:
    try:
        loaded_model = tf.keras.models.load_model(model_file)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

def main():
    st.title("Bird Image Classification")

    classes = np.array(['ABBOTTS BABBLER',
    'ABBOTTS BOOBY','ABYSSINIAN GROUND HORNBILL','AFRICAN CROWNED CRANE',
    'AFRICAN EMERALD CUCKOO','AFRICAN FIREFINCH','AFRICAN OYSTER CATCHER',
    'AFRICAN PIED HORNBILL','AFRICAN PYGMY GOOSE','ALBATROSS',
    'ALBERTS TOWHEE','ALEXANDRINE PARAKEET','ALPINE CHOUGH',
    'ALTAMIRA YELLOWTHROAT','AMERICAN AVOCET','AMERICAN BITTERN',
    'AMERICAN COOT','AMERICAN FLAMINGO','AMERICAN GOLDFINCH','AMERICAN KESTREL'])

    reshaped_classes = classes.reshape(4,5)
    
    st.write("Choose from below Bird Species: ")
    for row in reshaped_classes:
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            col.markdown(f"<div style='background-color: black; color: white; padding: 10px; margin: 5px; text-align: center; border-radius: 10px;'>{row[idx]}</div>", unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image...",width=300)
        st.write("")
        st.write("Classifying...")

        converted_image = load_and_preprocess_image(uploaded_image)
        
        result = loaded_model.predict(converted_image)

        st.warning(classes[np.argmax(result)])

if __name__ == "__main__":
    main()
