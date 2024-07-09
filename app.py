import streamlit as st
import numpy as np
import joblib
from PIL import Image
import os
import tensorflow as tf

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(target_size)
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
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    classes = np.array(['ABBOTTS BABBLER',
    'ABBOTTS BOOBY','ABYSSINIAN GROUND HORNBILL','AFRICAN CROWNED CRANE',
    'AFRICAN EMERALD CUCKOO','AFRICAN FIREFINCH','AFRICAN OYSTER CATCHER',
    'AFRICAN PIED HORNBILL','AFRICAN PYGMY GOOSE','ALBATROSS',
    'ALBERTS TOWHEE','ALEXANDRINE PARAKEET','ALPINE CHOUGH',
    'ALTAMIRA YELLOWTHROAT','AMERICAN AVOCET','AMERICAN BITTERN',
    'AMERICAN COOT','AMERICAN FLAMINGO','AMERICAN GOLDFINCH','AMERICAN KESTREL'])
        
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image...",width=200)
        st.write("")
        st.write("Classifying...")

        converted_image = load_and_preprocess_image(uploaded_image)
        
        result = loaded_model.predict(converted_image)

        st.warning(classes[np.argmax(result)])

if __name__ == "__main__":
    main()
