import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Custom styles with CSS
st.markdown(
    """
    <style>
    body {
        background-color: #F0F8FF;
        font-family: 'Arial', sans-serif;
    }
    .header {
        font-size: 45px;
        font-weight: bold;
        color: #FF4500;
        text-align: center;
    }
    .subheader {
        font-size: 20px;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-box {
        font-size: 25px;
        color: #2F4F4F;
        background-color: #DCDCDC;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Header
st.markdown('<p class="header">Image Classification Model</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload a Vegetable or Fruit Image to Classify</p>', unsafe_allow_html=True)

# Load model
model = load_model('Image_classify.keras')
data_cat = [
    'Bean', 'Bitter Gourd', 'Bottle Gourd', 'Brinjal', 'Broccoli', 'Cabbage',
    'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato',
    'Pumpkin', 'Radish', 'Tomato'
]

# Image input section
img_height = 180
img_width = 180

# Image upload section
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Load image and display
    image_load = Image.open(uploaded_file)
    st.image(image_load, caption='Uploaded Image', width=200)

    # Preprocess image
    image_resized = image_load.resize((img_height, img_width))
    img_arr = tf.keras.preprocessing.image.img_to_array(image_resized)
    img_bat = tf.expand_dims(img_arr, 0)

    # Prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])

    # Display result in a visually appealing box
    st.markdown(
        f'<div class="result-box">Veg/Fruit in the image is <b>{data_cat[np.argmax(score)]}</b> '
        f'with an accuracy of <b>{np.max(score) * 100:.2f}%</b></div>',
        unsafe_allow_html=True
    )

    # Progress bar for accuracy score
    st.progress(float(np.max(score)))
else:
    st.write("Please upload an image to classify.")
