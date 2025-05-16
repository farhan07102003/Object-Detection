import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the Teachable Machine model with compatibility fix
def load_keras_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={'Functional': tf.keras.models.Model}
    )

# Load class labels
def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f]

# Preprocess image for model input
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Load assets
model = load_keras_model('keras_model.h5')
class_labels = load_labels('labels.txt')

# Streamlit UI
st.title("Anomaly Detection System 🔍")
st.markdown("""
### How to Use:
1. Upload an image of your product **OR** use the camera
2. Wait for AI analysis
3. Review detection results
""")

# Image upload section
uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        normal_confidence = prediction[0][0] * 100
        anomaly_confidence = prediction[0][1] * 100
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.metric("Normal Confidence", f"{normal_confidence:.1f}%")
            st.metric("Anomaly Confidence", f"{anomaly_confidence:.1f}%")
            
            if anomaly_confidence > 50:
                st.error("🚨 Anomaly Detected!")
            else:
                st.success("✅ Product Normal")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Real-time camera section
st.divider()
st.subheader("Real-time Camera Inspection")
camera_image = st.camera_input("Take a live photo")
if camera_image:
    try:
        image = Image.open(camera_image)
        processed_image = preprocess_image(image)
        
        prediction = model.predict(processed_image)
        anomaly_confidence = prediction[0][1] * 100
        
        if anomaly_confidence > 50:
            st.error(f"🚨 Defect Detected! ({anomaly_confidence:.1f}% confidence)")
        else:
            st.success(f"✅ Product Normal ({anomaly_confidence:.1f}% anomaly confidence)")
            
        st.image(image, caption="Captured Image", use_column_width=True)

    except Exception as e:
        st.error(f"Camera error: {str(e)}")