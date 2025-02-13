# Import required libraries
import os
import pickle
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import cv2

# Initialize the MTCNN detector and VGGFace model
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load pre-trained features and filenames
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Ensure the 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Function to save uploaded images
def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving the image: {e}")
        return False

# Function to extract features from an image
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if len(results) == 0:
        st.error("No face detected in the image. Please upload a clearer image.")
        return None

    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]

    # Resize and preprocess the face
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

# Function to recommend the closest match
def recommend(feature_list, features):
    similarity = [cosine_similarity(features.reshape(1, -1), feature.reshape(1, -1))[0][0] for feature in feature_list]
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

# Streamlit app UI
st.title('Which Bollywood Celebrity Are You?')

uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        # Load and display the uploaded image
        display_image = Image.open(uploaded_image)

        # Extract features
        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)
        if features is not None:
            # Get recommendation
            index_pos = recommend(feature_list, features)
            predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
            predicted_actor_path = os.path.normpath(filenames[index_pos])

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.header('Your Uploaded Image')
                st.image(display_image)
            with col2:
                st.header(f"Seems like {predicted_actor}")
                st.image(predicted_actor_path, width=300)
