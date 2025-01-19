import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = keras.models.load_model("fruit_detection.keras")
    image = keras.preprocessing.image.load_img(test_image,target_size=(256,256))
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array=image_array/255.0
    image_array=np.expand_dims(image_array,axis=0) #convert single image to batch
    predictions = model.predict(image_array)
    return predictions #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Fruit Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("FRUIT DETECTION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (2432 images)
                2. test (352 images)
                3. validation (288 images)

                """)
    
#Prediction Page
elif(app_mode=="Fruit Recognition"):
    st.header("Fruit Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    
    #Predict button
    if(st.button("Predict")):
        st.balloons()
        st.write("Here you Go")
        predictions=model_prediction(test_image)
        result_index = np.argmax(predictions)
        confidence_score = np.max(predictions)

        #Reading Labels
        class_names =['apple_6','apple_braeburn_1','apple_crimson_snow_1','apple_golden_1','apple_golden_2','apple_golden_3','apple_granny_smith_1','apple_hit_1','apple_pink_lady_1','apple_red_1','apple_red_2','apple_red_3','apple_red_delicios_1','apple_red_yellow_1','apple_rotten_1','cabbage_white_1','carrot_1','cucumber_1','cucumber_3','eggplant_long_1','pear_1','pear_3','zucchini_1','zucchini_dark_1']
        st.success("Model is Predicting it's a {} ".format(class_names[result_index]))
        st.success("confidence : {} %".format(round(confidence_score*100,2)))

    st.markdown(""" 

                 [clik here](https://agrolearner.com/diseases-of-maize-and-treatment/) to know cure for maize diseases

                """)