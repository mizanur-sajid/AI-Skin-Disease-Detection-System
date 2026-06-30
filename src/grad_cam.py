try:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Google Drive mounted successfully!')
except ImportError:
    print('Not running in Google Colab, skipping drive mount.')


import numpy as np
import tensorflow as tf
import cv2


class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        
        if self.layerName is None:
            self.layerName = self.find_target_layer()
            
    def find_target_layer(self):
        # find the final convolutional layer in the network
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4: # conv layers usually have 4 dims
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        gradModel = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)

        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        
        # normalize heatmap
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        
        # apply colormap
        heatmap = (heatmap * 255).astype("uint8")
        
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        # apply colormap to heatmap
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            # If the image was loaded by tf and is [0, 255] float
            image = np.clip(image, 0, 255).astype("uint8")
            
        # overlay
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return output


import glob
import random
import matplotlib.pyplot as plt
import os
from PIL import Image

if __name__ == '__main__':
    # We need a trained model for this!
    model_path = '../models/best_model.h5'
    if not os.path.exists(model_path):
        print(f"Could not find model at {model_path}. Train the model first!")
    else:
        print("Loading model...")
        model = tf.keras.models.load_model(model_path)
        
        # Find some random images
        image_paths = glob.glob('../data/images/*.jpg')
        if len(image_paths) == 0:
            image_paths = glob.glob('data/images/*.jpg')
            
        if len(image_paths) > 0:
            sample_paths = random.sample(image_paths, min(5, len(image_paths)))
            
            plt.figure(figsize=(12, 18))
            for i, path in enumerate(sample_paths):
                # Load image
                img = Image.open(path).convert('RGB').resize((224, 224))
                img_orig = np.array(img)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                preds = model.predict(img_array, verbose=0)[0]
                class_idx = np.argmax(preds)
                
                # GradCAM
                cam = GradCAM(model, class_idx)
                heatmap = cam.compute_heatmap(img_array)
                overlay = cam.overlay_heatmap(heatmap, img_orig)
                
                # Plot Original
                plt.subplot(5, 2, 2*i + 1)
                plt.imshow(img_orig)
                plt.title(f"Original (Pred: {class_idx})")
                plt.axis('off')
                
                # Plot Overlay
                plt.subplot(5, 2, 2*i + 2)
                plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                plt.title("Grad-CAM")
                plt.axis('off')
                
            plt.tight_layout()
            os.makedirs('../results', exist_ok=True)
            save_path = '../results/gradcam_showcase.png'
            plt.savefig(save_path)
            print(f"\nSaved collage to {save_path}")
            plt.show()
        else:
            print("No images found to generate collage.")


