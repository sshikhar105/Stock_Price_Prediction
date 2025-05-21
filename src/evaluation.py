
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import joblib
# from preprocess import preprocess_data # No longer needed
import matplotlib.pyplot as plt
# import os # No longer needed
from pathlib import Path # Added

# Wrapper class for naive predictions
class NaiveModel:
    def __init__(self, predictions):
        self.predictions = predictions
    
    def predict(self, X_test):
        # The X_test parameter is not used for the naive model, 
        # but the method signature needs to match.
        # Ensure the returned predictions have the same length as y_test (which X_test corresponds to)
        return self.predictions

def evaluate_model(model, X_test, y_test, is_naive=False, naive_predictions=None):
    if is_naive:
        predictions = naive_predictions
    else:
        predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return rmse, mae, r2

def plot_predictions(model, X_test, y_test, model_name, is_naive=False, naive_predictions=None):
    if is_naive:
        predictions = naive_predictions
    else:
        predictions = model.predict(X_test)
    
    # results_dir.mkdir(parents=True, exist_ok=True) # Ensure results_dir is created by caller
    
    plt.figure(figsize=(10, 6))
    # Use .values if y_test is a Series, otherwise assume it's a numpy array
    y_actual_plot = y_test.values if isinstance(y_test, pd.Series) else y_test
    predictions_plot = predictions.values if isinstance(predictions, pd.Series) else predictions
    
    plt.plot(y_actual_plot, label='Actual Prices')
    plt.plot(predictions_plot, label='Predicted Prices')
    plt.legend()
    plt.title(f'{model_name} - Actual vs Predicted Stock Prices')
    # results_dir will be passed to this function
    plt.savefig(results_dir / f"{model_name.lower().replace(' ', '_')}_predictions.png")
    plt.clf() # Clear the figure for the next plot

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    processed_data_dir = BASE_DIR / "processed_data"
    models_dir = BASE_DIR / "models"
    results_dir = BASE_DIR / "results"
    
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    # X_train is not directly needed for evaluation, but y_train is for naive baseline
    X_test_df = pd.read_pickle(processed_data_dir / "X_test.pkl")
    y_test_series = pd.read_pickle(processed_data_dir / "y_test.pkl")
    y_train_series = pd.read_pickle(processed_data_dir / "y_train.pkl")

    # Convert to NumPy arrays for consistency if models expect that
    # However, sklearn models generally handle pandas DataFrames/Series well.
    # For this script, X_test is used by model.predict(), y_test for metrics.
    X_test = X_test_df.values # Convert to numpy array as it was before scaling
    y_test = y_test_series # Keep as Series for indexing if needed, or .values for numpy array

    # Naive Baseline Model
    if not y_train_series.empty and not y_test.empty:
        y_pred_naive_list = [y_train_series.iloc[-1]] + y_test_series.iloc[:-1].tolist()
    elif not y_test.empty: # Edge case: y_train_series is empty, use first y_test as prediction for all
        y_pred_naive_list = [y_test_series.iloc[0]] * len(y_test_series)
    else: # both empty, should not happen with valid data
        y_pred_naive_list = []

    y_pred_naive = pd.Series(y_pred_naive_list, index=y_test_series.index)
    
    # The existing length check for y_pred_naive and y_test should still work with y_test_series
    if len(y_pred_naive) != len(y_test_series) and not y_test_series.empty:
        # Simplified adjustment: if lengths mismatch, and y_test_series is not empty,
        # re-create y_pred_naive to match y_test_series length,
        # using the first naive prediction for all entries.
        # This is a fallback, the primary logic should handle lengths correctly.
        if y_pred_naive_list: # if any prediction was made
            y_pred_naive = pd.Series([y_pred_naive_list[0]] * len(y_test_series), index=y_test_series.index)
        else: # if no prediction could be made (e.g. y_train empty and y_test empty)
            y_pred_naive = pd.Series([0] * len(y_test_series), index=y_test_series.index)


    print("Naive Baseline Evaluation:")
    # Pass y_test_series to evaluate_model
    naive_rmse, naive_mae, naive_r2 = evaluate_model(None, None, y_test_series, is_naive=True, naive_predictions=y_pred_naive)
    print(f"RMSE: {naive_rmse}, MAE: {naive_mae}, R2: {naive_r2}")

    # Load models
    rf_model_path = models_dir / "random_forest_model.pkl"
    xg_model_path = models_dir / "xgboost_model.pkl"
    rf_model = joblib.load(rf_model_path)
    xg_model = joblib.load(xg_model_path)
    
    print("Random Forest Evaluation:")
    # Pass X_test (numpy array) and y_test_series
    rf_rmse, rf_mae, rf_r2 = evaluate_model(rf_model, X_test, y_test_series)
    print(f"RMSE: {rf_rmse}, MAE: {rf_mae}, R2: {rf_r2}")
    
    print("XGBoost Evaluation:")
    # Pass X_test (numpy array) and y_test_series
    xg_rmse, xg_mae, xg_r2 = evaluate_model(xg_model, X_test, y_test_series)
    print(f"RMSE: {xg_rmse}, MAE: {xg_mae}, R2: {xg_r2}")
    
    # Plot predictions - pass results_dir to plot_predictions
    # The first two arguments to plot_predictions (model, X_test) are not used for naive.
    plot_predictions(None, None, y_test_series, "Naive Baseline", results_dir, is_naive=True, naive_predictions=y_pred_naive)
    plot_predictions(rf_model, X_test, y_test_series, "Random Forest", results_dir)
    plot_predictions(xg_model, X_test, y_test_series, "XGBoost", results_dir)
