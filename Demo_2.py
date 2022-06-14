import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
vgg_model = mobilenet.MobileNet(weights='imagenet')

header = st.container()
imgInput,imgOutput = st.columns(2)

img = []

def mriRecon(img):
    outI = img
    return outI

def mriRecon(img):
    outI = img
    return outI

def mriRecon(img):
    outI = img
    return outI

def Imagenet(img):
    numpy_image = img_to_array(img)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = mobilenet.preprocess_input(image_batch.copy())
    predictions = vgg_model.predict(processed_image)
    label_vgg = decode_predictions(predictions)
    return label_vgg

with header:
    st.title('Object identification Demo')


with imgInput:
    mri_k = st.file_uploader('Pick your input data')
    if mri_k is not None:
        img = Image.open(mri_k)
        rezImg = img.resize((224,224))
        st.header('Input')
        st.image(img)

with imgOutput:
    if st.button('Analyze the image'):
        observation = Imagenet(rezImg)
        st.header('Output')
        st.text(observation[0][0])
        st.text(observation[0][1])
        st.text(observation[0][2])

