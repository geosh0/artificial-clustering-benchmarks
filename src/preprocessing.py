from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_artificial_data(X, y):
    """
    Standardizes features and encodes string labels into integers.
    No outlier removal is applied to preserve artificial geometric structures.
    """
    # 1. Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Encode Labels (e.g., changing string classes into 0, 1, 2...)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X_scaled, y_encoded