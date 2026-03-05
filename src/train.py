from sklearn.ensemble import RandomForestClassifier
import joblib
from src.config import MODEL_PATH

def train_model(X_train, y_train, scaler, encoder):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train,y_train)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, "models/saved_models/scaler.pkl")
    joblib.dump(encoder, "models/saved_models/label_encoder.pkl")
    
    return model