import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from src.config import DATA_PATH, SAMPLE_RATE

def generate_spectograms(output_path="data/spectograms"):
    
    for genre in os.listdir(DATA_PATH):
        genre_path = os.path.join(output_path, genre)
        output_genre_path = os.path.join(output_path, genre)
        
        
        os.makedirs(output_genre_path, exist_ok=True)
        
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)
            
            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                spectogram = librosa.features.melspectrogram(
                    y=signal,
                    sr=sr
                )
                spectogram_db = librosa.power_to_db(spectogram, ref = max)
                
                plt.figure(figsize=(3,3))
                librosa.display.specshow(spectogram_db, sr=sr)
                plt.axis("off")
                
                save_path = os.path.join(output_genre_path, file.replace(".wav",".png"))
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
                
                plt.close()
                
                
            except Exception as e:
                print("Error:", e)