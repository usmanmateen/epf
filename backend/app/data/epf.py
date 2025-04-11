import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
from datetime import datetime
import joblib
import xgboost as xgb

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data(file_path='./processed_elexon_data.csv'):
  
    print(f"Loading data from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}. Please run preprocessing first.")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Dataset loaded with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def prepare_data(df):
  
    # Identify the price columns (target)
    price_cols = [col for col in df.columns if 'price' in col.lower()]
    
    if not price_cols:
        raise ValueError("No price column found in the dataset")
    
    # Use imbalance_price as the target (first price column)
    target_column = price_cols[0]
    print(f"Using '{target_column}' as target variable")
    
    # Identify timestamp column
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    time_column = time_cols[0] if time_cols else None
    
    # Create a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    # Handle categorical columns:
    # 1. One-hot encode the 'action' column if it exists
    if 'action' in df_processed.columns:
        print(f"One-hot encoding 'action' column with unique values: {df_processed['action'].unique()}")
        action_dummies = pd.get_dummies(df_processed['action'], prefix='action', drop_first=False)
        df_processed = pd.concat([df_processed.drop('action', axis=1), action_dummies], axis=1)
        
    # Check for other categorical columns and handle them
    categorical_cols = []
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object' and col not in time_cols:
            print(f"Detected categorical column: {col}")
            categorical_cols.append(col)
    
    # Apply one-hot encoding to any other categorical columns
    for col in categorical_cols:
        if col != 'action':  # Already handled above
            print(f"One-hot encoding '{col}' column")
            cat_dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=False)
            df_processed = pd.concat([df_processed.drop(col, axis=1), cat_dummies], axis=1)
    
    # Extract features (X) and target (y)
    # Exclude timestamp columns from features
    X = df_processed.drop(columns=[target_column] + (time_cols if time_cols else []))
    y = df_processed[target_column]
    
    # Keep the timestamp for later
    timestamps = df[time_column] if time_column else None
    
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Feature columns: {X.columns.tolist()}")
    
    return X, y, timestamps, target_column

def split_data(X, y, timestamps, test_size=0.2):
 
    # For time series, we use the last portion of data as test set
    train_size = int(len(X) * (1 - test_size))
    
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    if timestamps is not None:
        train_timestamps = timestamps.iloc[:train_size]
        test_timestamps = timestamps.iloc[train_size:]
    else:
        train_timestamps = None
        test_timestamps = None
    
    print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, train_timestamps, test_timestamps

def scale_features(X_train, X_test):

    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def prepare_lstm_data(X_train_scaled, y_train, X_test_scaled, y_test, sequence_length=24):
    """
    Prepare data for LSTM model by creating sequences.
    
    Parameters:
    -----------
    X_train_scaled : np.array
        Scaled training features
    y_train : pd.Series
        Training target values
    X_test_scaled : np.array
        Scaled testing features
    y_test : pd.Series
        Testing target values
    sequence_length : int
        Number of time steps to use for sequences
        
    Returns:
    --------
    tuple
        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm
    """
    # Convert to numpy arrays if they're not already
    y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    # Create sequences for training data
    X_train_lstm, y_train_lstm = [], []
    for i in range(len(X_train_scaled) - sequence_length):
        X_train_lstm.append(X_train_scaled[i:i+sequence_length])
        y_train_lstm.append(y_train_np[i+sequence_length])
    
    # Create sequences for testing data
    X_test_lstm, y_test_lstm = [], []
    for i in range(len(X_test_scaled) - sequence_length):
        X_test_lstm.append(X_test_scaled[i:i+sequence_length])
        y_test_lstm.append(y_test_np[i+sequence_length])
    
    # Convert to numpy arrays
    X_train_lstm = np.array(X_train_lstm)
    y_train_lstm = np.array(y_train_lstm)
    X_test_lstm = np.array(X_test_lstm)
    y_test_lstm = np.array(y_test_lstm)
    
    print(f"LSTM training data shape: {X_train_lstm.shape}")
    print(f"LSTM testing data shape: {X_test_lstm.shape}")
    
    return X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm

def train_xgboost_model(X_train_scaled, y_train, X_test_scaled, y_test):
    
    print("\n==== Training XGBoost Model ====")
    
    # Create validation set for early stopping
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    # Initialize XGBoost model with a reasonable fixed number of estimators
    # We'll use 100 trees since we can't use early stopping
    model = XGBRegressor(
        n_estimators=100,  # Fixed number since early stopping isn't available
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",  
        random_state=42
    )
    
    # Train model without early stopping
    print("Training XGBoost with 100 iterations (early stopping not available in this XGBoost version)")
    model.fit(
        X_train_part, y_train_part,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    # Print metrics
    print(f"XGBoost Training RMSE: {metrics['train_rmse']:.4f}")
    print(f"XGBoost Testing RMSE: {metrics['test_rmse']:.4f}")
    print(f"XGBoost Training MAE: {metrics['train_mae']:.4f}")
    print(f"XGBoost Testing MAE: {metrics['test_mae']:.4f}")
    print(f"XGBoost Training R²: {metrics['train_r2']:.4f}")
    print(f"XGBoost Testing R²: {metrics['test_r2']:.4f}")
    
    # Feature importance
    feature_importance = model.feature_importances_
    
    return model, y_test_pred, metrics, feature_importance

def train_lstm_model(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm):
    """
    Train LSTM model with hyperparameter tuning and improved architecture.
    
    Parameters:
    -----------
    X_train_lstm : np.array
        Training sequences
    y_train_lstm : np.array
        Training target values
    X_test_lstm : np.array
        Testing sequences
    y_test_lstm : np.array
        Testing target values
        
    Returns:
    --------
    tuple
        Trained model, predictions, evaluation metrics
    """
    print("\n==== Training LSTM Model with Improved Architecture ====")
    
    # Define early stopping callback with longer patience
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,  # Increased patience
        restore_best_weights=True,
        verbose=1
    )
    
    # Add reduce learning rate callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Build improved LSTM model with better regularization
    model = Sequential([
        # First LSTM layer with more units
        LSTM(128, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]),
             recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
        
        # Batch normalization to help with internal covariate shift
        tf.keras.layers.BatchNormalization(),
        
        # Dropout for regularization
        Dropout(0.3),
        
        # Second LSTM layer
        LSTM(64, return_sequences=False, 
             recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
        
        # Batch normalization
        tf.keras.layers.BatchNormalization(),
        
        # Dropout
        Dropout(0.3),
        
        # Dense layers with appropriate activation
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        # Output layer with linear activation for regression
        Dense(1)
    ])
    
    # Compile model with appropriate learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    # Model summary
    model.summary()
    
    # Train model with larger batch size and more epochs
    history = model.fit(
        X_train_lstm, y_train_lstm,
        epochs=200,  # More epochs with early stopping
        batch_size=64,  # Increased batch size
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=2
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save training history plot
    output_dir = 'app/data/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/lstm_training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()
    
    # Make predictions
    y_train_lstm_pred = model.predict(X_train_lstm).flatten()
    y_test_lstm_pred = model.predict(X_test_lstm).flatten()
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train_lstm, y_train_lstm_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test_lstm, y_test_lstm_pred)),
        'train_mae': mean_absolute_error(y_train_lstm, y_train_lstm_pred),
        'test_mae': mean_absolute_error(y_test_lstm, y_test_lstm_pred),
        'train_r2': r2_score(y_train_lstm, y_train_lstm_pred),
        'test_r2': r2_score(y_test_lstm, y_test_lstm_pred)
    }
    
    # Print metrics
    print(f"LSTM Training RMSE: {metrics['train_rmse']:.4f}")
    print(f"LSTM Testing RMSE: {metrics['test_rmse']:.4f}")
    print(f"LSTM Training MAE: {metrics['train_mae']:.4f}")
    print(f"LSTM Testing MAE: {metrics['test_mae']:.4f}")
    print(f"LSTM Training R²: {metrics['train_r2']:.4f}")
    print(f"LSTM Testing R²: {metrics['test_r2']:.4f}")
    
    return model, y_test_lstm_pred, metrics, history

def save_predictions(y_test, xgb_predictions, lstm_predictions, test_timestamps, target_column):
    
    # Create dataframe
    predictions_df = pd.DataFrame()
    
    # Add timestamps if available
    if test_timestamps is not None:
        predictions_df['timestamp'] = test_timestamps.values[:len(lstm_predictions)]
    
    # Add actual and predicted values
    predictions_df[f'actual_{target_column}'] = y_test.values[:len(lstm_predictions)]
    predictions_df[f'xgboost_predicted_{target_column}'] = xgb_predictions[:len(lstm_predictions)]
    predictions_df[f'lstm_predicted_{target_column}'] = lstm_predictions
    
    # Calculate errors
    predictions_df['xgboost_error'] = predictions_df[f'actual_{target_column}'] - predictions_df[f'xgboost_predicted_{target_column}']
    predictions_df['lstm_error'] = predictions_df[f'actual_{target_column}'] - predictions_df[f'lstm_predicted_{target_column}']
    
    # Save to CSV
    output_dir = 'app/data/predictions'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/energy_price_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    predictions_df.to_csv(output_file, index=False)
    
    print(f"\nPredictions saved to {output_file}")
    
    return predictions_df

def compare_models(xgb_metrics, lstm_metrics):

    print("\n==== Model Comparison ====")
    
    comparison_df = pd.DataFrame({
        'Metric': ['Training RMSE', 'Testing RMSE', 'Training MAE', 'Testing MAE', 'Training R²', 'Testing R²'],
        'XGBoost': [
            f"{xgb_metrics['train_rmse']:.4f}",
            f"{xgb_metrics['test_rmse']:.4f}",
            f"{xgb_metrics['train_mae']:.4f}",
            f"{xgb_metrics['test_mae']:.4f}",
            f"{xgb_metrics['train_r2']:.4f}",
            f"{xgb_metrics['test_r2']:.4f}"
        ],
        'LSTM': [
            f"{lstm_metrics['train_rmse']:.4f}",
            f"{lstm_metrics['test_rmse']:.4f}",
            f"{lstm_metrics['train_mae']:.4f}",
            f"{lstm_metrics['test_mae']:.4f}",
            f"{lstm_metrics['train_r2']:.4f}",
            f"{lstm_metrics['test_r2']:.4f}"
        ]
    })
    
    print(comparison_df.to_string(index=False))
    
    # Determine best model
    if xgb_metrics['test_rmse'] < lstm_metrics['test_rmse']:
        print("\nXGBoost model performs better in terms of RMSE.")
    elif lstm_metrics['test_rmse'] < xgb_metrics['test_rmse']:
        print("\nLSTM model performs better in terms of RMSE.")
    else:
        print("\nBoth models perform similarly in terms of RMSE.")
        
    # Save comparison to CSV
    output_dir = 'app/data/metrics'
    os.makedirs(output_dir, exist_ok=True)
    comparison_df.to_csv(f'{output_dir}/model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)

def plot_predictions(predictions_df, target_column):

    plt.figure(figsize=(14, 7))
    
    # Plot actual values
    plt.plot(predictions_df['timestamp'], predictions_df[f'actual_{target_column}'], 
             label='Actual', color='black', linewidth=2)
    
    # Plot XGBoost predictions
    plt.plot(predictions_df['timestamp'], predictions_df[f'xgboost_predicted_{target_column}'], 
             label='XGBoost', color='blue', alpha=0.7)
    
    # Plot LSTM predictions
    plt.plot(predictions_df['timestamp'], predictions_df[f'lstm_predicted_{target_column}'], 
             label='LSTM', color='red', alpha=0.7)
    
    plt.title(f'{target_column} - Actual vs Predicted Values')
    plt.xlabel('Timestamp')
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_dir = 'app/data/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    plot_file = f'{output_dir}/predictions_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Plot saved to {plot_file}")

def plot_feature_importance(feature_names, feature_importance):

    # Create dataframe with feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance (XGBoost)')
    plt.tight_layout()
    
    # Save plot
    output_dir = 'app/data/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    plot_file = f'{output_dir}/feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Feature importance plot saved to {plot_file}")

def save_models(xgb_model, lstm_model, scaler, feature_names, sequence_length, target_column):

    # Create models directory
    models_dir = 'app/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save XGBoost model
    joblib.dump(xgb_model, f'{models_dir}/xgboost_model.pkl')
    
    # Save LSTM model with .keras extension
    lstm_model.save(f'{models_dir}/lstm_model.keras')
    
    # Save scaler
    joblib.dump(scaler, f'{models_dir}/scaler.pkl')
    
    # Save feature names and other metadata
    metadata = {
        'feature_names': feature_names,
        'sequence_length': sequence_length,
        'target_column': target_column,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    joblib.dump(metadata, f'{models_dir}/model_metadata.pkl')
    
    print(f"\nModels and metadata saved to {models_dir}/")

def main():
    """Main function to run the energy price forecasting pipeline."""
    # Parameters
    sequence_length = 24  # Increased from 12 to 24 for better temporal patterns
    
    # Step 1: Load the dataset
    df = load_data()
    
    # Step 2: Prepare data
    X, y, timestamps, target_column = prepare_data(df)
    
    # Step 3: Split data into train and test sets
    X_train, X_test, y_train, y_test, train_timestamps, test_timestamps = split_data(X, y, timestamps)
    
    # Step 4: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 5: Train XGBoost model
    xgb_model, xgb_predictions, xgb_metrics, feature_importance = train_xgboost_model(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Step 6: Prepare data for LSTM
    X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm = prepare_lstm_data(
        X_train_scaled, y_train, X_test_scaled, y_test, sequence_length
    )
    
    # Step 7: Train LSTM model
    lstm_model, lstm_predictions, lstm_metrics, lstm_history = train_lstm_model(
        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm
    )
    
    # Step 8: Compare models
    compare_models(xgb_metrics, lstm_metrics)
    
    # Step 9: Save predictions
    predictions_df = save_predictions(
        y_test, xgb_predictions, lstm_predictions, test_timestamps, target_column
    )
    
    # Step 10: Plot predictions
    plot_predictions(predictions_df, target_column)
    
    # Step 11: Plot feature importance
    plot_feature_importance(X_train.columns, feature_importance)
    
    # Step 12: Save models for later use with FastAPI
    save_models(xgb_model, lstm_model, scaler, X_train.columns.tolist(), 
                sequence_length, target_column)

if __name__ == "__main__":
    main() 