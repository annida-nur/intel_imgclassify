
import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2


st.title("Intel Image Classification")
st.caption("""You can classify your pictures into 6 classes (Buildings, Forest, Glacier, Mountain, Sea, Street) in here.
           This App Created by Annida Nur Islami with Implement Image Classification Concept and Convolutional Neural Network (CNN) Algorithm.
           However This App still needs any improvement on its model to increase its accuracy.""")

upload_file = st.file_uploader("Upload CT Scan image", type = ['png','jpg'], accept_multiple_files=True)
generate_pred = st.button("predict")
model = tf.keras.models.load_model("model/lima_1.h5")

def import_n_pred(image_data, model):
    size = (64,64)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, dsize=(64,64),interpolation=cv2.INTER_CUBIC)
    img_reshape = img_resize[np.newaxis,...]
    prediction = model.predict(img_reshape)

    return prediction

if generate_pred:
    for upload_file in upload_file:
        image = Image.open(upload_file)
        with st.expander(upload_file.name, expanded=True):
            x = upload_file.name
            x = x.split(".",1)
            x = x[0]
            pred = import_n_pred(image, model)
            labels = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'sea', 'Street']
            final = labels[np.argmax(pred)]
            
            st.write(f"This Image is Classified as {final}")
            st.image(image, use_column_width=True)
