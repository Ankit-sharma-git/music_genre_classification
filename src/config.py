import os 
DATA_PATH = "data/raw/Data/genres_original"

SAMPLE_RATE = 22050
DURATION = 30 # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

N_MFCC = 13

MODEL_PATH = "models/saved_models/genre_model.pkl"