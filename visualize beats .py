#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import load_and_process_record, bandpass_filter, normalize_beat
import wfdb

def visualize_preprocessing(record_name='100'):
   
    record = wfdb.rdrecord(f'data/mitdb/{record_name}')
    raw_signal = record.p_signal[:1000, 0] 
    filtered_signal = bandpass_filter(raw_signal)
    normalized_signal = normalize_beat(filtered_signal)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(raw_signal, color='red', alpha=0.5, label='Raw Signal (with Noise)')
    plt.title(f'Record {record_name}: Before Preprocessing')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(normalized_signal, color='green', label='Processed Signal (Filtered & Normalized)')
    plt.title('After Preprocessing')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

visualize_preprocessing('100')


# In[ ]:




