import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image

model=tf.keras.models.load_model(r"C:\Users\haider\Desktop\New folder (3)\face_mask_detection.keras")

st.title("Face Mask Detection")
input_image=st.file_uploader("Enter the image")

if st.button("Detect"):
    input_image=Image.open(input_image)
    input_image=input_image.convert("RGB")
    input_image=input_image.resize((128,128))
    input_image=np.array(input_image)
    input_image=input_image/255.0
    input_image=input_image.reshape(1,128,128,3)
    pred=model.predict(input_image)
    pre=pred.argmax()

    if pre==0:
        st.write('This person is wearing a mask')
    else:
        st.write("This person is not wearing mask")
