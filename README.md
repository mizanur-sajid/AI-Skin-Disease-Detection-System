# AI Skin Disease Detection System

An end-to-end Deep Learning project for classifying skin lesions into 7 distinct categories using an EfficientNetV2 model. This project includes a complete pipeline: from data loading and model training, to model interpretation (Grad-CAM), and finally a Flask web application for real-time predictions.

## Features
*   **Model Architecture**: TensorFlow/Keras-based classification (EfficientNetV2).
*   **Explainable AI**: Includes Grad-CAM (`grad_cam.py`) to generate heatmaps showing exactly which parts of the lesion the model used to make its decision.
*   **Web Interface**: A Flask web app (`app.py`) for users to upload skin images and receive instant predictions.
*   **Google Colab Ready**: All python scripts have corresponding `.ipynb` versions for easy execution on cloud GPUs.

## Dataset Support
This project natively supports the **HAM10000** dataset schema (7 classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`). 
*   **Images Directory**: `data/images/`
*   **Metadata File**: `data/HAM10000_metadata.csv` (used for train/validation splitting).

*(Note: The project is currently configured with the lightweight **DermaMNIST** dataset for fast, local training!)*

---

## 🛠️ Installation & Setup

1. **Install Dependencies**
   You must install the required libraries (like TensorFlow, Flask, Pandas, etc.) before running any scripts. Run this command in your terminal:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Data Directory**
   Ensure your `data/` directory contains `images/` and the `HAM10000_metadata.csv` file.

---

## 🚀 How to Use

### 1. Training the Model
Train the model locally from your terminal. The script will automatically load the dataset, apply augmentations, and save the best model to `models/best_model.h5`.
```bash
python src/train.py --epochs 10 --batch_size 32
```
*(Colab Users: You can upload this project to Google Drive and run `src/train.ipynb` instead).*

### 2. Visualizing with Grad-CAM
After training, verify what your model is "looking at" using the Grad-CAM visualization script. This requires a trained model in the `models/` directory.
```bash
python src/grad_cam.py
```

### 3. Launching the Web App
Start the Flask application to use the model via a friendly web interface.
```bash
python app.py
```
Then, open your web browser and navigate to `http://127.0.0.1:5000`.

---

## File Structure

```text
skin_disease_diagnosis/
│
├── app.py                     # Main Flask web application
├── app.ipynb                  # Jupyter notebook version of the web app
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation
│
├── data/
│   ├── images/                # Folder containing all .jpg skin lesion images
│   └── HAM10000_metadata.csv  # Labels and dataset split information
│
├── models/                    
│   └── best_model.h5          # Saved model weights (generated after training)
│
├── src/
│   ├── data_loader.py         # Pipeline for loading and augmenting images
│   ├── model.py               # EfficientNetV2 model definition
│   ├── train.py               # Main training script
│   └── grad_cam.py            # Generates heatmap overlays for model interpretability
│
├── static/                    # CSS, JS, and uploaded images for the web app
└── templates/                 # HTML files for the web app
```
