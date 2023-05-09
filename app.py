import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

class_names = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
               "Apple___healthy", "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
               "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
               "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
               "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
               "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
               "Pepper,_bell___Bacterial_spot", "Potato___Early_blight", "Potato___Late_blight",
               "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
               "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
               "Tomato___Early_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
               "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
               "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"]

class SessionState:
    def __init__(self):
        self.cache = {}

# Create a SessionState object
session_state = SessionState()

def load_model():
    model = tf.keras.models.load_model('/content/best_plant_model.h5')
    return model

def get_model():
    if 'model' not in session_state.cache:
        session_state.cache['model'] = load_model()
    return session_state.cache['model']

model = get_model()

st.write("""
         # Leaf disease detection
         """
         )

file = st.file_uploader("Please upload a brain scan file", type=["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    pred = model.predict(img_reshape)
    return pred

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    st.write(score)
    predicted_class_index = np.argmax(score)
    predicted_class_name = class_names[predicted_class_index]
    st.write("Predicted Class:", predicted_class_name)



