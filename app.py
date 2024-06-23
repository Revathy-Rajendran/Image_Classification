import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 
st.set_page_config(page_title='Image Classification App', layout='centered')
st.title('Image Classification App')
st.markdown("""
Upload an image to classify it into different fruits and vegetables.
""")
model = load_model(r'C:\Users\91815\Image_Classification\Image_classify.keras')
data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
img_height = 180
img_width = 180
# image =st.text_input('Enter Image name','Apple.jpg')
# image = st.text_input("Enter Image name", "Apple.jpg")
# if(st.button('Submit')):
#     result = image.title()
#     st.success(result + " Uploaded successfully...")

# Function to classify
def classify_image(image):
    image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
    img_arr = tf.keras.utils.array_to_img(image_load)
    img_bat=tf.expand_dims(img_arr,0)

    predict = model.predict(img_bat)

    score = tf.nn.softmax(predict)
    # st.image(image, width=200)
    st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
    st.write('With accuracy of ' + str(np.max(score)*100))

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Classification button
    if st.button('Classify'):
        classify_image(uploaded_file)