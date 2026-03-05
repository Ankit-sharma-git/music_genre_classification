from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

def preprocess_data(X,y):
    
    encoder = LabelEncoder()
    
    y_encoded = encoder.fit_transform(y)
    
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, encoder, scaler