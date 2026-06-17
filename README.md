# EuphoScan — AI-Powered Skin Cancer Classifier

## What is this project?
EuphoScan is an AI tool that uses deep learning to classify types of skin lesions from dermatoscopic images. It is trained on the HAM10000 dataset and can help identify potentially dangerous skin cancers early.

## Project Structure
├── data/
│ ├── images/ # Dermatoscopic images (HAM10000)
│ ├── metadata.csv # Labels and metadata


├── notebooks/
│ ├── 01_data_exploration.ipynb # Explore dataset and visualize
│ ├── 02_preprocessing.ipynb # Image preprocessing and augmentation
│ ├── 03_model_training.ipynb # CNN model building and training
│ ├── 04_evaluation.ipynb # Model evaluation and analysis


├── model/
│ ├── euphoscan_model.h5 # Saved Keras model
│ ├── euphoscan_model.tflite # Converted TensorFlow Lite model
├── README.md # Project overview
├── requirements.txt # Python dependencies


## How to set up
1. Download the HAM10000 dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
2. Put images in `data/images/` and `metadata.csv` in `data/`.
3. Install required Python packages using `requirements.txt`.
4. Open and run the notebooks in this order:
   - `01_data_exploration.ipynb`
   - `02_preprocessing.ipynb`
   - `03_model_training.ipynb`
   - `04_evaluation.ipynb`

## How to use the trained model
- The trained Keras model is saved as `model/euphoscan_model.h5`.
- The TensorFlow Lite version is `model/euphoscan_model.tflite`.
- Use the `.tflite` model in your mobile app (e.g., Flutter + TensorFlow Lite).

## Disclaimer
This project is for educational and research purposes only. It is **not a medical tool**. Always consult qualified health professionals for any medical concerns.

## Author
Ishika Prajapati(https://github.com/IshikaPrajapati)
