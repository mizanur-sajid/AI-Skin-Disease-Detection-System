# AI Skin Disease Detection System

An end-to-end Deep Learning project for classifying skin lesions into 7 distinct categories using an EfficientNetV2 model. This project includes a complete pipeline: from data loading and model training, to model interpretation (Grad-CAM), and finally a Flask web application for real-time predictions. 

**Note: The entire codebase is structured using Jupyter Notebooks (`.ipynb`) for an interactive, cell-by-cell execution experience.**

## Features
*   **Model Architecture**: TensorFlow/Keras-based classification (EfficientNetV2).
*   **Explainable AI**: Includes Grad-CAM (`src/grad_cam.ipynb`) to generate heatmaps showing exactly which parts of the lesion the model used to make its decision.
*   **Web Interface**: A Flask web app (`app.ipynb`) for users to upload skin images and receive instant predictions.
*   **Interactive & Explained Code**: All core components are written in heavily documented, human-readable `.ipynb` notebooks, making it perfect for learning and experimenting.

## Dataset Support
This project natively supports the **HAM10000** dataset schema (7 classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`). 
*   **Images Directory**: `data/images/`
*   **Metadata File**: `data/HAM10000_metadata.csv` (used for train/validation splitting).

*(Note: The project is currently configured with the lightweight **DermaMNIST** dataset for fast, local training!)*

---

## 🛠️ Installation & Setup

1. **Install Dependencies**
   You must install the required libraries (like TensorFlow, Flask, Pandas, Jupyter, etc.) before running any notebooks. Run this command in your terminal:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Data Directory**
   Ensure your `data/` directory contains your images and metadata CSV file.

---

## 🚀 How to Use

Since all code is in `.ipynb` format, you should use Jupyter Notebook, JupyterLab, or an IDE like VS Code to open and run the files.

### 1. Training the Model
Open `src/train.ipynb`. This notebook will guide you through loading the dataset, building the model architecture, and executing the training loop. Run the cells sequentially to save your best model to the `models/` folder.

### 2. Visualizing with Grad-CAM
After training, open `src/grad_cam.ipynb` to visualize what your model is "looking at" when making predictions.

### 3. Launching the Web App
Open `app.ipynb` and run all cells. The final cell will start the Flask application locally.
Once it's running, open your web browser and navigate to `http://127.0.0.1:5000` to interact with the model via a friendly web interface.

---

## File Structure

```text
skin_disease_diagnosis/
│
├── app.ipynb                  # Main Flask web application
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
│   ├── data_loader.ipynb      # Pipeline for loading and augmenting images
│   ├── model.ipynb            # EfficientNetV2 model definition
│   ├── train.ipynb            # Main training script
│   ├── grad_cam.ipynb         # Generates heatmap overlays for model interpretability
│   └── generate_dummy_model.ipynb # Helper notebook to create a quick test model
│
├── static/                    # CSS, JS, and uploaded images for the web app
└── templates/                 # HTML files for the web app
```

---

<div align="center">
  <b>Made with ❤️ by Mizanur Sajid</b>
</div>
