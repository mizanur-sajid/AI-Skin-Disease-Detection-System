import os
import tensorflow as tf
from model import build_efficientnetv2_model, compile_model


def generate_dummy():
    print("Building model...")
    model = build_efficientnetv2_model(num_classes=7)
    model = compile_model(model)
    
    save_path = '../models/dummy_model.h5'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Saving dummy model to {save_path}...")
    model.save(save_path)
    print("Done!")


if __name__ == "__main__":
    generate_dummy()


