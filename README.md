# -Which-Bollywood-Celebrity-Do-You-Resemble-A-Deep-Learning-Project-
This is a Streamlit-based Deep Learning project that uses VGGFace and MTCNN to recognize Bollywood celebrities based on uploaded images. It detects facial features and matches them with the closest Bollywood celebrity using cosine similarity.

🛠 Features
Face detection using MTCNN (Multi-task Cascaded Convolutional Networks)
Feature extraction with VGGFace (ResNet50)
Image similarity matching using cosine similarity
Streamlit-based web interface for easy interaction
🚀 Installation
Clone the repository:

git clone https://github.com/Sahil489s/Bollywood-Celebrity-Recognition.git
cd Bollywood-Celebrity-Recognition
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
📌 Usage
Run the Streamlit app:

streamlit run app.py
Upload an image, and the model will predict which Bollywood celebrity you resemble!

📂 Dataset
The images of Bollywood celebrities were initially downloaded from Kaggle. You can also download additional images using Bing Image Downloader to expand the dataset.

To download more celebrity images, use the following Python code:


from bing_image_downloader import downloader
downloader.download('Shah Rukh Khan', limit=100, output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
This will help you extend the dataset and include more celebrity images.

🔍 How It Works
Face Detection: The uploaded image is processed, and a face is detected using MTCNN.
Feature Extraction: Facial features are extracted using the VGGFace (ResNet50) model.
Similarity Matching: The extracted features are compared with the stored celebrity embeddings using cosine similarity.
The most similar celebrity image is displayed with the prediction.
📜 Requirements
Python 3.x
OpenCV
NumPy
TensorFlow
Keras
MTCNN
keras-vggface
Streamlit
Install the required dependencies by running:

pip install opencv-python numpy tensorflow keras mtcnn keras-vggface streamlit
💡 Future Improvements
Fine-tune the models for better accuracy
Deploy the model as a public web app for easy access
Expand the dataset to include more Bollywood celebrities
🏆 Author
Sahil Sharma


⭐ If you like this project, give it a star!

