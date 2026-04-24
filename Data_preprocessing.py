#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install wfdb')


# In[3]:


import wfdb  # For reading ECG files
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt   # For signal filtering
from sklearn.preprocessing import LabelEncoder
import os

# Record names in MIT-BIH (48 recordings total)
MIT_BIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107',
    '108', '109', '111', '112', '113', '114', '115', '116',
    '117', '118', '119', '121', '122', '123', '124', '200',
    '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

# Records designated for Testing (DS2 set - AAMI standard) "This split is standard in scientific research"
TEST_RECORDS = [
    '101', '105', '114', '118', '124', '201', '210', '217'
]

TRAIN_RECORDS = [r for r in MIT_BIH_RECORDS if r not in TEST_RECORDS]

# Mapping beat symbols to the five categories (AAMI standard)
BEAT_MAPPING = {
    # Normal beats → N
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',

    # Supraventricular → S
    # (Abnormal signals originating from above the ventricles)
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',

    # Ventricular → V (dangerous)
    'V': 'V', 'E': 'V',

    # Fusion beats → F
    'F': 'F',

    # Unknown/Pacemaker → Q
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

# Size of each beat in data points: 180 points before R-peak and 180 after = 360 points
BEAT_BEFORE = 180
BEAT_AFTER = 180
BEAT_SIZE = BEAT_BEFORE + BEAT_AFTER 

# Signal Processing Functions


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=4):
  
    #Apply a Butterworth bandpass filter to the signal.
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design Butterworth filter
    b, a = butter(order, [low, high], btype='band')

    # Apply filter in both directions (forward + backward) to prevent phase shift
    filtered = filtfilt(b, a, signal)
    return filtered


def normalize_beat(beat):
    min_val = np.min(beat)
    max_val = np.max(beat)

    # Avoid division by zero if the signal is flat
    if max_val - min_val == 0:
        return beat
    return (beat - min_val) / (max_val - min_val)



# In[4]:


# Loading and Processing a Single Record

def load_and_process_record(record_name, data_path='data/mitdb'):
    record_path = os.path.join(data_path, record_name)

    # wfdb.rdrecord reads .hea and .dat files
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]   # Take only the first channel (MLII)
    fs = record.fs                    # Sampling rate (360 Hz)


    # Each annotation contains: R-peak position and classification
    annotation = wfdb.rdann(record_path, 'atr')
    beat_samples = annotation.sample    # R-peak positions in points
    beat_symbols = annotation.symbol    # Classification symbols

    signal_filtered = bandpass_filter(signal, fs=fs)
    beats = []
    labels = []

    for i in range(len(beat_samples)):
        symbol = beat_symbols[i]

        # Ignore beats not in the mapping dictionary
        if symbol not in BEAT_MAPPING:
            continue

        # Determine start and end points of the beat
        start = beat_samples[i] - BEAT_BEFORE
        end = beat_samples[i] + BEAT_AFTER

        # Ignore beats at the signal edges
        if start < 0 or end > len(signal_filtered):
            continue

        # Extract and normalize the beat
        beat = signal_filtered[start:end]
        beat = normalize_beat(beat)

        beats.append(beat)
        labels.append(BEAT_MAPPING[symbol])

    return np.array(beats), np.array(labels)


# In[5]:


def build_dataset(records, data_path='data/mitdb'):
    """
    Builds a complete dataset from a list of record names.
    Prints a summary report of the beats in each category.
    """
    all_beats = []
    all_labels = []

    print(f"Processing {len(records)} records...")

    for record in records:
        try:
            beats, labels = load_and_process_record(record, data_path)
            all_beats.append(beats)
            all_labels.append(labels)
            print(f"  Record {record}: {len(beats)} beats")
        except Exception as e:
            print(f"  Error processing record {record}: {e}")

    # Concatenate all beats into one matrix
    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")

    return X, y


def download_mitdb(save_path='data/mitdb'):
   
    os.makedirs(save_path, exist_ok=True)
    print("Downloading MIT-BIH Database...")

    wfdb.dl_database('mitdb', dl_dir=save_path)
    print(f"Downloaded to: {save_path}")



def compute_class_weights(y_encoded, num_classes=5):
 
    #Computes class weights to handle class imbalance. (Give higher weights to rare classes so the model doesn't ignore them)
  
  
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(num_classes)
    weights = compute_class_weight('balanced', classes=classes, y=y_encoded)
    weight_dict = {i: w for i, w in enumerate(weights)}

    print("\nClass weights:")
    class_names = ['N', 'S', 'V', 'F', 'Q']
    for i, name in enumerate(class_names):
        print(f"  {name}: {weights[i]:.3f}")

    return weight_dict


# In[7]:


def prepare_data(data_path='data/mitdb', download=False):
    # Download data if needed
    if download:
        download_mitdb(data_path)

    # Build Training and Testing sets
    print("Training Set:")
   
    X_train, y_train = build_dataset(TRAIN_RECORDS, data_path)

    print("Testing Set:")
    X_test, y_test = build_dataset(TEST_RECORDS, data_path)

    # Encode labels into integers (N=0, S=1, V=2, F=3, Q=4)
    le = LabelEncoder()
    le.fit(['N', 'S', 'V', 'F', 'Q'])
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    # Shape becomes: (N_samples, 360, 1)
    X_train = X_train.reshape(-1, BEAT_SIZE, 1)
    X_test = X_test.reshape(-1, BEAT_SIZE, 1)

    # Compute class weights
    class_weights = compute_class_weights(y_train_enc)

    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    return X_train, X_test, y_train_enc, y_test_enc, class_weights, le


if __name__ == '__main__':
    # Quick test run
    X_train, X_test, y_train, y_test, weights, le = prepare_data(download=True)
    print("\nData is ready for training!")


# In[ ]:




