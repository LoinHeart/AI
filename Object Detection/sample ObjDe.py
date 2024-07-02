import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.activations import gelu
import numpy as np

def create_model(input_size=(224, 224, 3), num_classes=20):
    input_layer = Input(shape=input_size)
    
    # Convolutional layers
    x = Conv2D(32, (3, 3), activation=gelu, padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation=gelu, padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation=gelu, padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), activation=gelu, padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(512, activation=gelu)(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation=gelu)(x)
    x = Dropout(0.5)(x)
    
    # Output layers
    # Bounding box coordinates (4 coordinates)
    bbox_output = Dense(4, activation='linear', name='bbox_output')(x)
    
    # Class prediction
    class_output = Dense(num_classes, activation='softmax', name='class_output')(x)
    
    model = Model(inputs=input_layer, outputs=[bbox_output, class_output])
    
    return model

def custom_loss(y_true, y_pred):
    bbox_true, class_true = y_true
    bbox_pred, class_pred = y_pred
    
    # Loss for bounding box regression (mean squared error)
    bbox_loss = tf.reduce_mean(tf.square(bbox_true - bbox_pred))
    
    # Loss for classification (categorical crossentropy)
    class_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(class_true, class_pred))
    
    # Combine the losses
    total_loss = bbox_loss + class_loss
    
    return total_loss

def main():
    # Create the model
    model = create_model()
    model.compile(optimizer='adam', loss=custom_loss)
    model.summary()
    
    # Generate dummy data for demonstration
    num_samples = 1000
    input_size = (224, 224, 3)
    num_classes = 20
    
    X_train = np.random.rand(num_samples, *input_size)
    y_bbox_train = np.random.rand(num_samples, 4)
    y_class_train = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, num_samples), num_classes)
    
    # Combine the bounding box and class labels
    y_train = [y_bbox_train, y_class_train]
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
if __name__ == "__main__":
    main()
