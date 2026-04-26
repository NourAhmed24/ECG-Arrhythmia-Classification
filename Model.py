#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_ecg_cnn(input_length=360, num_classes=5):

    # input layer
    inputs = keras.Input(shape=(input_length, 1), name='ecg_input')

    # First Convolutional Block
    x = layers.Conv1D(filters=32, kernel_size=7, padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPool1D(pool_size=3, strides=2, name='pool1')(x)

    # Second Convolutional Block
    x = layers.Conv1D(filters=64, kernel_size=5, padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.MaxPool1D(pool_size=3, strides=2, name='pool2')(x)

    # Third Convolutional Block
    x = layers.Conv1D(filters=128, kernel_size=3, padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    x = layers.MaxPool1D(pool_size=3, strides=2, name='pool3')(x)

    # Global Pooling and Regularization
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)  
    x = layers.Dropout(0.3, name='dropout1')(x)

    # Fully Connected Layers
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.2, name='dropout2')(x)

    # Output Layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='ECG_1D_CNN')
    return model


def get_model_summary(model):
    
    model.summary()
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    # Approximate size assuming float32 (4 bytes per parameter)
    print(f"Approximate model size: {total_params * 4 / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
 
    model = build_ecg_cnn()
    get_model_summary(model)
    
    # Test with dummy data
    dummy_input = np.random.randn(8, 360, 1)   # batch of 8 beats
    dummy_output = model(dummy_input, training=False)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
    print("Model is working correctly!")

