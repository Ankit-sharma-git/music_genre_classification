import os
import numpy as np
import librosa
from src.config import DATA_PATH, SAMPLE_RATE, SAMPLES_PER_TRACK, N_MFCC


def extract_features():
    
    features=[]
    labels = []
    
    for genre in os.listdir(DATA_PATH):
        genre_path = os.path.join(DATA_PATH, genre)
        
        if not os.path.isdir(genre_path):
            continue
        
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)
            
            try:
                
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                mfcc = librosa.feature.mfcc(
                    y = signal,
                    sr = sr,
                    n_mfcc= N_MFCC
                )
                
                mfcc_mean = np.mean(mfcc.T, axis=0)
                features.append(mfcc_mean)
                
                labels.append(genre)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
    return np.array(features), np.array(labels)
                