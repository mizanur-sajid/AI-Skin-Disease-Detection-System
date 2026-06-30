try:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Google Drive mounted successfully!')
except ImportError:
    print('Not running in Google Colab, skipping drive mount.')


import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from data_loader import SkinLesionDataLoader
from model import build_efficientnetv2_model, compile_model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def train(data_dir, csv_path, epochs, batch_size, model_save_path):
    # 1. Load Data
    print("Loading data...")
    loader = SkinLesionDataLoader(data_dir=data_dir, csv_path=csv_path, batch_size=batch_size)
    train_ds, val_ds = loader.load_data()

    # 2. Build Model
    print("Building model...")
    model = build_efficientnetv2_model(num_classes=loader.num_classes)
    model = compile_model(model)

    # 3. Setup Callbacks
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        TensorBoard(log_dir='./logs')
    ]

    # Compute Class Weights to handle imbalance
    print('Computing class weights...')
    # We need to gather all labels from the training dataset
    all_labels = []
    for _, labels in train_ds:
        all_labels.extend(labels.numpy())
    
    classes = np.unique(all_labels)
    weights = compute_class_weight('balanced', classes=classes, y=all_labels)
    class_weight_dict = {c: w for c, w in zip(classes, weights)}
    print('Class Weights:', class_weight_dict)

    # 4. Train Model
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    print("Training complete. Best model saved to:", model_save_path)
    return history, model, val_ds, loader


if __name__ == '__main__':
    # Configuration (Adjust paths if running from a different root in Colab)
    DATA_DIR = '../data/images' if os.path.exists('../data/images') else 'data/images'
    CSV_PATH = '../data/HAM10000_metadata.csv' if os.path.exists('../data/HAM10000_metadata.csv') else 'data/HAM10000_metadata.csv'
    RESULTS_DIR = '../results' if os.path.exists('../data') else 'results'
    MODEL_SAVE = '../models/best_model.h5' if os.path.exists('../models') else 'models/best_model.h5'
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run Training
    history, model, val_ds, loader = train(
        data_dir=DATA_DIR,
        csv_path=CSV_PATH,
        epochs=5, # Reduced for quick testing, change to 20+
        batch_size=32,
        model_save_path=MODEL_SAVE
    )


if __name__ == '__main__':
    # 1. Plot & Save Training History
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    history_path = os.path.join(RESULTS_DIR, 'training_history.png')
    plt.savefig(history_path)
    print(f"Saved training history plot to {history_path}")
    plt.show()

    # 2. Evaluate and save Confusion Matrix
    print("Evaluating on validation set...")
    # Extract true labels and predictions
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=loader.classes)
    report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")
    print(report)
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=loader.classes, yticklabels=loader.classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Saved confusion matrix plot to {cm_path}")
    plt.show()


