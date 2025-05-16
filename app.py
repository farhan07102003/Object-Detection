import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the Teachable Machine model
model = tf.keras.models.load_model(
    'keras_model.h5',
    compile=False,
    custom_objects={'Functional': tf.keras.models.Model}  # Add this line
)

# Load class labels (optional, but useful for debugging)
with open('labels.txt', 'r') as f:
    class_labels = [line.strip() for line in f]

st.title("Anomaly Detection System 🔍")

# --- Option 1: Upload Image ---
uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "png"])
if uploaded_file:
    # Preprocess the image
    image = Image.open(uploaded_file).resize((224, 224))  # Resize to 224x224 (Teachable Machine's default)
    img_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (shape = [1, 224, 224, 3])

    # Predict
    prediction = model.predict(img_array)
    st.write("Prediction scores:", prediction)  # Optional: Show raw scores

    # Check for anomaly (assuming class 0 = Normal, class 1 = Anomaly)
    if prediction[0][1] > 0.5:  # Threshold at 50% confidence
        st.error("🚨 **Anomaly Detected!** (Class: Anomaly)")
    else:
        st.success("✅ **Product is Normal!** (Class: Normal)")

# --- Bonus: Real-Time Camera Input ---
st.header("OR Use Live Camera")
camera_image = st.camera_input("Take a photo of the product")
if camera_image:
    image = Image.open(camera_image).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0][1] > 0.5:
        st.error("🚨 **Defect Detected!**")
    else:
        st.success("✅ **Product is Normal!**")