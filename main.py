from src.feature_extraction import extract_features
from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.spectrogram_generator import generate_spectograms
from src.train_cnn import train_cnn

def main():
    
    
    
    print("Extracting features...")
    X, y = extract_features()
    
    print("Preprocessing...")
    
    X_train, X_test, y_train, y_test, encoder, scaler = preprocess_data(X,y)
    
    print("Training Model...")
    model = train_model(X_train, y_train, scaler, encoder)
    
    print("Evaluating...")
    evaluate_model(model, X_test, y_test, encoder)
    
    generate_spectograms()
    
    train_cnn()
    
if __name__ == "__main__":
    main()