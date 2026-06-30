import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

def build_efficientnetv2_model(num_classes, img_size=(224, 224)):
    """
    Builds an EfficientNetV2B0 model for skin lesion classification.
    """
    input_shape = (*img_size, 3)
    
    # Load base model, excluding top classification layer
    base_model = EfficientNetV2B0(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape
    )
    
    # Freeze base model layers initially (optional, but good practice for transfer learning)
    base_model.trainable = False

    # Create new classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pooling')(x)
    x = Dropout(0.2, name='top_dropout')(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def compile_model(model, learning_rate=1e-3):
    """
    Compiles the model with optimizer, loss function, and metrics.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Using categorical crossentropy since we have exclusive classes (7 classes in HAM10000)
    # Note: If labels are integers, use sparse_categorical_crossentropy
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    ]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

if __name__ == "__main__":
    # Test model building
    model = build_efficientnetv2_model(num_classes=7)
    model = compile_model(model)
    model.summary()
