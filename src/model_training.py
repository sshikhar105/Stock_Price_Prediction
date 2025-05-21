
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pandas as pd
import joblib
from preprocess import preprocess_data
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from pathlib import Path # Added
import os # os will be removed later if not needed

def train_random_forest(X_train, y_train):
    param_grid_rf = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    time_series_split = TimeSeriesSplit(n_splits=5)
    # Using n_jobs=-1 to use all available cores for faster search
    grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), 
                                  param_grid_rf, 
                                  cv=time_series_split, 
                                  scoring='neg_mean_squared_error', 
                                  n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    print(f"Best Random Forest params: {grid_search_rf.best_params_}")
    return grid_search_rf.best_estimator_

def train_xgboost(X_train, y_train):
    # Ensure y_train is a 1D array
    if hasattr(y_train, 'values'):
        y_train_processed = y_train.values.ravel()
    else:
        y_train_processed = y_train.ravel()

    param_grid_xgb = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    time_series_split = TimeSeriesSplit(n_splits=5)
    # Using n_jobs=-1 to use all available cores for faster search
    grid_search_xgb = GridSearchCV(xgb.XGBRegressor(random_state=42), 
                                   param_grid_xgb, 
                                   cv=time_series_split, 
                                   scoring='neg_mean_squared_error', 
                                   n_jobs=-1)
    grid_search_xgb.fit(X_train, y_train_processed) # Use processed y_train
    print(f"Best XGBoost params: {grid_search_xgb.best_params_}")
    return grid_search_xgb.best_estimator_

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Define paths
    data_dir = BASE_DIR / "data"
    processed_data_dir = BASE_DIR / "processed_data"
    models_dir = BASE_DIR / "models"
    
    # Create directories if they don't exist
    data_dir.mkdir(parents=True, exist_ok=True) # Though typically data dir should exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    stock_data_path = data_dir / "aapl_stock_data.csv"
    
    # Load raw data
    stock_data = pd.read_csv(stock_data_path)
    
    # Preprocess data (now returns scaler as well)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data)
    
    # Save processed data and scaler
    # Using pickle for DataFrames/Series to preserve types and index
    pd.to_pickle(pd.DataFrame(X_train), processed_data_dir / "X_train.pkl")
    pd.to_pickle(pd.DataFrame(X_test), processed_data_dir / "X_test.pkl")
    pd.to_pickle(pd.Series(y_train), processed_data_dir / "y_train.pkl")
    pd.to_pickle(pd.Series(y_test), processed_data_dir / "y_test.pkl")
    joblib.dump(scaler, processed_data_dir / "scaler.pkl")
    print(f"Processed data and scaler saved to {processed_data_dir}")

    print("Training Random Forest with GridSearchCV...")
    rf_model = train_random_forest(X_train, y_train)
    print("Training XGBoost with GridSearchCV...")
    xg_model = train_xgboost(X_train, y_train) # y_train is passed as is, XGBoost function handles ravel if needed
    
    # Define model paths
    rf_model_path = models_dir / "random_forest_model.pkl"
    xg_model_path = models_dir / "xgboost_model.pkl"
    
    # Save models
    joblib.dump(rf_model, rf_model_path)
    joblib.dump(xg_model, xg_model_path)
    print(f"Tuned models saved successfully to {models_dir}")
