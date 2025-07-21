import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import gdown
import os

# Step 1: Extract file ID
file_id = '1S93Iqlu0N4_YxEppmIJUPdpzgOc3ZTlB'  # Your actual file ID
url = f'https://drive.google.com/uc?id={file_id}'
output = 'handwriting.csv'



# Load trained model
clf = pickle.load(open("decision_tree.pkl", "rb"))
# Step 2: Download only if not already downloaded
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Step 3: Load the dataset
df = pd.read_csv(output)

st.markdown(
    """
    ## 📝 Handwritten Digit Classifier (Simple Demo)
    
    This is a basic handwritten digit classification demo using a **Decision Tree** model.  
    It predicts the digit label from preloaded image data (from a CSV file), where each image is a flattened 28×28 grayscale matrix.  
    
    This app is purely for demonstration and educational purposes — it doesn’t perform real-time handwriting recognition, but instead classifies based on existing CSV data.
    """
)
st.write("Step 1: Enter any number between 0-42000 and hit Enter (It contian 42K data point only)")
inputNum = st.number_input("Enter Index", min_value=0, max_value=len(df)-1, step=1)

X = df.iloc[:,1:].values
y = df.iloc[:,0].values


if inputNum:
    try:
        idx = int(inputNum)
        if 0 <= idx < len(X):
            image = X[idx].reshape(28, 28)

            # ↓ Decrease figure size here (width, height in inches)
            fig, ax = plt.subplots(figsize=(1, 1))  
            ax.imshow(image, cmap='gray')
            ax.axis('off')
            st.image(image, width=150, clamp=True)
        else:
            st.error("Index out of range.")
    except ValueError:
        st.error("Please enter a valid integer.")


st.write("Step 2: Click on Predict Button to check if he is able to predict the number properly or not.")
if st.button("Detect Number", type="primary"):
    prediction = clf.predict(X[inputNum].reshape(1,-1))
    st.header(f"The number in the image is {prediction}")

