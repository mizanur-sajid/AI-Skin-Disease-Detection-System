import os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

class SkinLesionDataLoader:
    def __init__(self, data_dir, csv_path, batch_size=32, img_size=(224, 224)):
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.num_classes = len(self.classes)

    def load_data(self):
        """Loads data from metadata CSV (assuming HAM10000 structure)"""
        if not os.path.exists(self.csv_path):
            print(f"Warning: Metadata file {self.csv_path} not found. Returning dummy datasets.")
            return self._create_dummy_dataset(), self._create_dummy_dataset()

        df = pd.read_csv(self.csv_path)
        
        # Assume CSV has 'image_id' and 'dx' (diagnosis class) columns
        # Map dx to integers based on self.classes
        class_map = {c: i for i, c in enumerate(self.classes)}
        df['label'] = df['dx'].map(class_map)
        
        # Image paths
        df['image_path'] = df['image_id'].apply(lambda x: os.path.join(self.data_dir, f"{x}.jpg"))
        
        # Filter out rows where image doesn't exist
        df = df[df['image_path'].apply(os.path.exists)]

        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
        
        train_ds = self._create_tf_dataset(train_df['image_path'].values, train_df['label'].values, is_training=True)
        val_ds = self._create_tf_dataset(val_df['image_path'].values, val_df['label'].values, is_training=False)
        
        return train_ds, val_ds

    def _process_path(self, file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        # EfficientNetV2 includes its own preprocessing (scaling), so we just resize here.
        return img, label

    def _augment(self, img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        return img, label

    def _create_tf_dataset(self, file_paths, labels, is_training=False):
        ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        ds = ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            ds = ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.shuffle(buffer_size=1000)
            
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def _create_dummy_dataset(self):
        """Creates a dummy dataset for testing when real data is unavailable."""
        dummy_images = tf.random.uniform((100, *self.img_size, 3), minval=0, maxval=255, dtype=tf.float32)
        dummy_labels = tf.random.uniform((100,), minval=0, maxval=self.num_classes, dtype=tf.int32)
        
        ds = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels))
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

if __name__ == "__main__":
    # Test data loader
    loader = SkinLesionDataLoader("data/images", "data/HAM10000_metadata.csv")
    train_ds, val_ds = loader.load_data()
    for img, label in train_ds.take(1):
        print(f"Batch shape: {img.shape}, Labels shape: {label.shape}")
