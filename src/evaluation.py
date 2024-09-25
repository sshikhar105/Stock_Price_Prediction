
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import joblib
from preprocess import preprocess_data
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return rmse, mae, r2

def plot_predictions(model, X_test, y_test):
    predictions = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.legend()
    plt.title('Actual vs Predicted Stock Prices')
    plt.show()

if __name__ == "__main__":
    stock_data = pd.read_csv('../data/aapl_stock_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(stock_data)
    
    rf_model = joblib.load('../models/random_forest_model.pkl')
    xg_model = joblib.load('../models/xgboost_model.pkl')
    
    print("Random Forest Evaluation:")
    rf_rmse, rf_mae, rf_r2 = evaluate_model(rf_model, X_test, y_test)
    print(f"RMSE: {rf_rmse}, MAE: {rf_mae}, R2: {rf_r2}")
    
    print("XGBoost Evaluation:")
    xg_rmse, xg_mae, xg_r2 = evaluate_model(xg_model, X_test, y_test)
    print(f"RMSE: {xg_rmse}, MAE: {xg_mae}, R2: {xg_r2}")
    
    # Plot predictions
    plot_predictions(rf_model, X_test, y_test)
    plot_predictions(xg_model, X_test, y_test)
