from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your trained Keras model
MODEL_PATH = 'model/euphoscan_model.h5'
# model = load_model(MODEL_PATH)
model = load_model(MODEL_PATH, compile=False)

print("Model input shape:", model.input_shape)  # Should print (None, 64, 64, 3)

# Define your class names here (update as per your model's classes)
class_names = [
    "Melanoma",
    "Nevus",
    "Seborrheic Keratosis",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/howItWorks')
def how_it_works():
    return render_template('howItWorks.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400

        if file:
            upload_folder = 'uploads'
            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)

            try:
                # Open image in RGB mode and resize to 64x64
                img = Image.open(filepath).convert('RGB')
                img = img.resize((64, 64))

                # Convert to numpy array and normalize pixels to [0,1]
                img_array = np.array(img) / 255.0
                img_array = img_array.astype(np.float32)

                # Add batch dimension: (1, 64, 64, 3)
                img_array = np.expand_dims(img_array, axis=0)

                print("Input shape to model:", img_array.shape)  # Should be (1, 64, 64, 3)

                # Predict
                prediction = model.predict(img_array)

                predicted_class = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction) * 100

                # Get confidence for all classes
                confidences_all = prediction[0] * 100  # as percentage

                # Create sorted list of tuples (class_name, confidence)
                confidence_list = list(zip(class_names, confidences_all))
                confidence_list.sort(key=lambda x: x[1], reverse=True)

                label = class_names[predicted_class]

                return render_template('predict.html', label=label, confidence=round(confidence, 2), confidence_list=confidence_list)
            
            except Exception as e:
                return f"Error processing the image: {str(e)}", 500

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
