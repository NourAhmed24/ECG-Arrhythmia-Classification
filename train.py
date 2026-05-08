#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from data_preprocessing import prepare_data
from model import build_ecg_cnn


CONFIG = {
    'batch_size':     128,
    'epochs':          50,
    'learning_rate':   0.01,
    'input_length':    360,
    'num_classes':     5,
    'data_path':       'data/mitdb',
    'model_save_path': 'results/best_model.keras',
    'results_dir':     'results',
}

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']

def train_model(config=CONFIG, use_class_weights=True):
    """
    Train the 1D CNN.
    Returns: model, history, (X_test, y_test, label_encoder)
    """
    os.makedirs(config['results_dir'], exist_ok=True)

    # --- Step 1: load data ---
    print("Step 1 — Loading data")
    X_train, X_test, y_train, y_test, class_weights, le = prepare_data(
        data_path=config['data_path']
    )

    # --- Step 2: build model ---
    print("\nStep 2 — Building model")
    model = build_ecg_cnn(
        input_length=config['input_length'],
        num_classes=config['num_classes']
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # --- Step 3: callbacks ---
    callbacks = [
        # Save the best model (lowest val_loss)
        keras.callbacks.ModelCheckpoint(
            filepath=config['model_save_path'],
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Stop training if val_loss does not improve for 10 epochs
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Halve the learning rate if val_loss stalls for 5 epochs
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    # --- Step 4: train ---
    print("\nStep 3 — Training")
    history = model.fit(
        X_train, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=0.15,
        class_weight=class_weights if use_class_weights else None,
        callbacks=callbacks,
        verbose=1
    )

    # --- Step 5: evaluate on test set ---
    print("\nStep 4 — Test evaluation")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")

    # Save config + results to JSON
    _save_json(
        {**config, 'test_accuracy': test_acc, 'test_loss': test_loss},
        f"{config['results_dir']}/run_config.json"
    )
    print(f"  Config saved  → {config['results_dir']}/run_config.json")

    return model, history, (X_test, y_test, le)

def plot_training_history(history, save_path='results/training_curves.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history.history['loss']) + 1)

    # Loss curve
    axes[0].plot(epochs, history.history['loss'],     'b-',  label='Train',      lw=2)
    axes[0].plot(epochs, history.history['val_loss'], 'r--', label='Validation', lw=2)
    axes[0].set_title('Loss');      axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss');     axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy curve
    axes[1].plot(epochs, history.history['accuracy'],     'b-',  label='Train',      lw=2)
    axes[1].plot(epochs, history.history['val_accuracy'], 'r--', label='Validation', lw=2)
    axes[1].set_title('Accuracy');  axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved → {save_path}")
    
def _save_json(data, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

   
if __name__ == '__main__':
    print("  ECG Arrhythmia — Training Pipeline")

    # Train the model
    model, history, test_data = train_model()
    X_test, y_test, le = test_data

    # Plot training curves
    plot_training_history(history)
    
    print("  Done! Check the results/ folder.")


# In[ ]:




