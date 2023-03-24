# Import dependencies
import os 
import streamlit as st
import cv2
import numpy as np
import pickle
#from PIL import Image
from sklearn.svm import SVC

# Load the saved SVM model
svm = pickle.load(open("svm_model.sav", 'rb'))

# Save Label categories to print from predicted index
Cats = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


st.write("""Fashion Classifier App""")
st.write("This web app can classify clothing types into 10 Fasion Categories")

# Specifying file types to be jpg images/png images
file=st.file_uploader("Please upload clothing image", type = ["png","jpg"])

if file == None: # Prevent further code from being run if there is no file and prompt for file upload
    st.text("Please upload an image file")
else:
	file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8) # Read as array
	opencv_image = cv2.imdecode(file_bytes, 1) # Convery array to image
	gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY) # Convert RGB Image to a grayscale
	st.image(opencv_image) # Show image to user
	dim = (28,28) #Original Fashion Dataset pixels
	new_img = cv2.resize(gray,dim) # Resize image to 28*28
	array = np.asarray(new_img) # Convert into a 2D array
	array2 = array.flatten()[np.newaxis] # Convery into a 1 row array
	prediction=svm.predict(array2) # Predict from model
	pred_label=Cats[int(prediction[0])] # Get label from indexes
	st.text("Uploaded Image is a ")
	st.text(pred_label)
