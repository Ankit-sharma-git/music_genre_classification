import librosa
import numpy as np
import joblib
from src.config import SAMPLE_RATE, N_MFCC

def predict_genre(file_path):
    
    model = joblib.load("model/saved_models/genre.model.pkl")
    scaler = joblib.load("models/saved_models/scaler.pkl")
    encoder = joblib.load("models/saved_models/label_encoder.pkl")
    
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr = sr,
        n_mfcc= N_MFCC
    )
    
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    mfcc_scaled = scaler.transform([mfcc_mean])
    
    prediction = model.predict(mfcc_scaled)
    
    genre = encoder.inverse_transform(prediction)
    
    return genre[0]