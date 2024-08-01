from flask import Flask, render_template, request
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import os
import io
from PIL import Image

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.getcwd(), 'model.keras')  # Ensure the path is correct
model = load_model(model_path)

# List of disease types
disease_types = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
    'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    try:
        # Convert file to byte data and use BytesIO
        img_data = file.read()
        img = image.load_img(io.BytesIO(img_data), target_size=(64, 64))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        # Debug: Check the shape of the input image
        print(f'Image shape: {img.shape}')

        # Model prediction
        predictions = model.predict(img)
        
        # Debug: Check the raw prediction values
        print(f'Raw predictions: {predictions}')
        
        predicted_class = np.argmax(predictions)
        
        # Debug: Check the predicted class index
        print(f'Predicted class index: {predicted_class}')
        
        predicted_class_name = disease_types[predicted_class]

        return render_template('index.html', prediction=predicted_class_name)

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
