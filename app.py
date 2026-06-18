# # from flask import Flask, render_template, request
# # import os
# # from tensorflow.keras.models import load_model
# # import numpy as np
# # from PIL import Image

# # app = Flask(__name__)

# # # Load your trained Keras model
# # MODEL_PATH = 'model/euphoscan_model.h5'
# # # model = load_model(MODEL_PATH)
# # from tensorflow.keras.models import load_model

# # model = load_model(
# #     MODEL_PATH,
# #     compile=False,
# #     safe_mode=False
# # )

# # print("Model input shape:", model.input_shape)  # Should print (None, 64, 64, 3)

# # # Define your class names here (update as per your model's classes)
# # class_names = [
# #     "Melanoma",
# #     "Nevus",
# #     "Seborrheic Keratosis",
# #     "Basal Cell Carcinoma",
# #     "Actinic Keratosis",
# #     "Benign Keratosis",
# #     "Dermatofibroma",
# #     "Vascular Lesion"
# # ]

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/howItWorks')
# # def how_it_works():
# #     return render_template('howItWorks.html')

# # @app.route('/about')
# # def about():
# #     return render_template('about.html')

# # @app.route('/contact')
# # def contact():
# #     return render_template('contact.html')

# # @app.route('/predict', methods=['GET', 'POST'])
# # def predict():
# #     if request.method == 'POST':
# #         if 'file' not in request.files:
# #             return 'No file uploaded', 400
        
# #         file = request.files['file']
# #         if file.filename == '':
# #             return 'No selected file', 400

# #         if file:
# #             upload_folder = 'uploads'
# #             os.makedirs(upload_folder, exist_ok=True)
# #             filepath = os.path.join(upload_folder, file.filename)
# #             file.save(filepath)

# #             try:
# #                 # Open image in RGB mode and resize to 64x64
# #                 img = Image.open(filepath).convert('RGB')
# #                 img = img.resize((64, 64))

# #                 # Convert to numpy array and normalize pixels to [0,1]
# #                 img_array = np.array(img) / 255.0
# #                 img_array = img_array.astype(np.float32)

# #                 # Add batch dimension: (1, 64, 64, 3)
# #                 img_array = np.expand_dims(img_array, axis=0)

# #                 print("Input shape to model:", img_array.shape)  # Should be (1, 64, 64, 3)

# #                 # Predict
# #                 prediction = model.predict(img_array)

# #                 predicted_class = np.argmax(prediction, axis=1)[0]
# #                 confidence = np.max(prediction) * 100

# #                 # Get confidence for all classes
# #                 confidences_all = prediction[0] * 100  # as percentage

# #                 # Create sorted list of tuples (class_name, confidence)
# #                 confidence_list = list(zip(class_names, confidences_all))
# #                 confidence_list.sort(key=lambda x: x[1], reverse=True)

# #                 label = class_names[predicted_class]

# #                 return render_template('predict.html', label=label, confidence=round(confidence, 2), confidence_list=confidence_list)
            
# #             except Exception as e:
# #                 return f"Error processing the image: {str(e)}", 500

# #     return render_template('predict.html')

# # # if __name__ == '__main__':
# # #     app.run(debug=True)
# # import os

# # if __name__ == "__main__":
# #     port = int(os.environ.get("PORT", 5000))
# #     app.run(host="0.0.0.0", port=port)

# from flask import Flask, render_template, request
# import os
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load model
# MODEL_PATH = "model/euphoscan_model.keras"

# try:
#     model = load_model(MODEL_PATH, compile=False)
#     print("Model loaded successfully!")
#     print("Model input shape:", model.input_shape)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     raise

# # Class names
# class_names = [
#     "Melanoma",
#     "Nevus",
#     "Seborrheic Keratosis",
#     "Basal Cell Carcinoma",
#     "Actinic Keratosis",
#     "Benign Keratosis",
#     "Dermatofibroma",
#     "Vascular Lesion"
# ]

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/howItWorks")
# def how_it_works():
#     return render_template("howItWorks.html")

# @app.route("/about")
# def about():
#     return render_template("about.html")

# @app.route("/contact")
# def contact():
#     return render_template("contact.html")

# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":

#         if "file" not in request.files:
#             return "No file uploaded", 400

#         file = request.files["file"]

#         if file.filename == "":
#             return "No selected file", 400

#         try:
#             upload_folder = "uploads"
#             os.makedirs(upload_folder, exist_ok=True)

#             filepath = os.path.join(upload_folder, file.filename)
#             file.save(filepath)

#             img = Image.open(filepath).convert("RGB")
#             img = img.resize((64, 64))

#             img_array = np.array(img, dtype=np.float32) / 255.0
#             img_array = np.expand_dims(img_array, axis=0)

#             prediction = model.predict(img_array)

#             predicted_class = np.argmax(prediction, axis=1)[0]
#             confidence = float(np.max(prediction) * 100)

#             confidences_all = prediction[0] * 100

#             confidence_list = list(zip(class_names, confidences_all))
#             confidence_list.sort(key=lambda x: x[1], reverse=True)

#             label = class_names[predicted_class]

#             return render_template(
#                 "predict.html",
#                 label=label,
#                 confidence=round(confidence, 2),
#                 confidence_list=confidence_list
#             )

#         except Exception as e:
#             return f"Error processing image: {str(e)}", 500

#     return render_template("predict.html")

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)


from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

import tensorflow as tf
import keras

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

# ==================================
# Load Model
# ==================================

# MODEL_PATH = "model/euphoscan_model.keras"

# print("Loading model...")

# model = load_model(MODEL_PATH, compile=False)
MODEL_PATH = "model/euphoscan_model.keras"

print("Current Directory:", os.getcwd())
print("Model Exists:", os.path.exists(MODEL_PATH))

print("Loading model...")
model = load_model(MODEL_PATH, compile=False)

print("Model loaded successfully!")

print("Model loaded successfully!")
print("Model input shape:", model.input_shape)

# ==================================
# Class Names
# ==================================

class_names = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Nevus",
    "Vascular Lesions"
]

# ==================================
# Routes
# ==================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/howItWorks")
def how_it_works():
    return render_template("howItWorks.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]

        if file.filename == "":
            return "No selected file", 400

        try:

            # Save upload
            upload_folder = "uploads"
            os.makedirs(upload_folder, exist_ok=True)

            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)

            # =====================
            # Image preprocessing
            # =====================

            img = Image.open(filepath).convert("RGB")

            # MobileNetV2 input size
            img = img.resize((224, 224))

            img_array = np.array(img).astype(np.float32)

            # Normalize
            img_array = img_array / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            print("=" * 50)
            print("MODEL EXPECTS :", model.input_shape)
            print("IMAGE SENT    :", img_array.shape)
            print("=" * 50)

            # Prediction
            prediction = model.predict(img_array)

            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction) * 100)

            confidence_list = []

            for i in range(len(class_names)):
                confidence_list.append(
                    (
                        class_names[i],
                        float(prediction[0][i] * 100)
                    )
                )

            confidence_list.sort(
                key=lambda x: x[1],
                reverse=True
            )

            label = class_names[predicted_class]

            return render_template(
                "predict.html",
                label=label,
                confidence=round(confidence, 2),
                confidence_list=confidence_list
            )

        except Exception as e:
            print("PREDICTION ERROR:")
            print(str(e))
            return f"Error processing image: {str(e)}", 500

    return render_template("predict.html")


# ==================================
# Run Flask
# ==================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    # app.run(
    #     host="0.0.0.0",
    #     port=port,
    #     debug=True
    # )
    app.run(
    host="0.0.0.0",
    port=port
)