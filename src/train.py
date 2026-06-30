import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from data_loader import SkinLesionDataLoader
from model import build_efficientnetv2_model, compile_model

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

    # 4. Train Model
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    print("Training complete. Best model saved to:", model_save_path)
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Skin Lesion Classification Model')
    parser.add_argument('--data_dir', type=str, default='data/images', help='Path to image directory')
    parser.add_argument('--csv_path', type=str, default='data/HAM10000_metadata.csv', help='Path to metadata CSV')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--save_path', type=str, default='models/best_model.h5', help='Path to save the model')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.save_path
    )
