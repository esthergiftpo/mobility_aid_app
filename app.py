import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO

# Load model and classes
model = load_model('best_model.keras')
classes = ['crutches', 'no mobility aids', 'wheelchair', 'whitecane']

st.title("Accessibility Object Classifier")
st.write("Upload an image or use your webcam to classify objects.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Camera input
camera_input = st.camera_input("Take a picture")

image_data = None
if uploaded_file is not None:
    image_data = uploaded_file
elif camera_input is not None:
    image_data = camera_input

if image_data is not None:
    # Load and preprocess image
    image = load_img(image_data, target_size=(224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_array)
    threshold = 0.3
    predicted_binary = (predictions[0] > threshold).astype(int)

    # Map indices to class labels
    predicted_labels = [classes[i] for i in np.where(predicted_binary == 1)[0]]

    # Show result
    st.image(image, caption="Uploaded Image", use_container_width=True)
    if predicted_labels:
        st.success("Predicted classes: " + ", ".join(predicted_labels))
    else:
        st.warning("No confident prediction found.")
else:
    st.info("Please upload an image or use the camera.")







