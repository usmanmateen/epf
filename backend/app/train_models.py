import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import os

# Load Data
DATA_PATH = "app/data/processed_elexon_data.csv"

def load_data():
    print(f"Loading data from {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Please run the preprocessing script first.")
    
    df = pd.read_csv(DATA_PATH)

    # Identify the target column (we assume the first price column is our target)
    price_columns = [col for col in df.columns if 'price' in col.lower()]
    if not price_columns:
        raise ValueError("No price column found in the dataset")
    
    target_column = price_columns[0]
    print(f"Using '{target_column}' as the target column for energy price prediction.")

    # Drop unnecessary columns for training (we don't need ID columns, duplicate time information, etc.)
    # Keep all numeric features that aren't redundant
    date_columns = [col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()]
    
    # We'll keep the datetime for reference but not for training
    # Exclude the target column from features
    X = df.drop(columns=[target_column] + date_columns)
    y = df[target_column]

    print(f"Features: {X.columns.tolist()}")
    print(f"Target: {target_column}")
    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Normalize data using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns

# Train XGBoost Model
def train_xgboost(X, y):
    print("Training XGBoost model...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"XGBoost R² score - Training: {train_score:.4f}, Testing: {test_score:.4f}")
    
    return model

# Prepare Data for LSTM
def prepare_lstm_data(X, y, time_steps=10):
    X_lstm, y_lstm = [], []
    for i in range(len(X) - time_steps):
        X_lstm.append(X[i:i + time_steps])
        y_lstm.append(y.iloc[i + time_steps])  # Ensure correct index handling
    
    # Convert to numpy arrays
    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)
    
    print(f"LSTM data shape: X={X_lstm.shape}, y={y_lstm.shape}")
    
    return X_lstm, y_lstm

# Train LSTM Model
def train_lstm(X, y):
    # Prepare data for LSTM (add time dimension)
    X_lstm, y_lstm = prepare_lstm_data(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

    print("Training LSTM model...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    
    # Train with early stopping
    history = model.fit(
        X_train, y_train, 
        epochs=20, 
        batch_size=32, 
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"LSTM MSE loss - Training: {train_loss:.4f}, Testing: {test_loss:.4f}")
    
    return model

# Save models and metadata
def save_models(xgb_model, lstm_model, scaler, feature_names):
    # Ensure models directory exists
    os.makedirs("app/models", exist_ok=True)
    
    # Save models
    joblib.dump(xgb_model, "app/models/xgboost_model.pkl")
    joblib.dump(scaler, "app/models/scaler.pkl")
    joblib.dump(feature_names, "app/models/feature_names.pkl")
    
    # Save LSTM Model correctly
    lstm_model.save("app/models/lstm_model.keras")  # Use .keras extension
    
    print("Models saved successfully to app/models/")

# Train and Save Both Models
if __name__ == "__main__":
    try:
        X, y, scaler, feature_names = load_data()
        xgb_model = train_xgboost(X, y)
        lstm_model = train_lstm(X, y)
        save_models(xgb_model, lstm_model, scaler, feature_names)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
