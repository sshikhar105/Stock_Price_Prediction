
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pandas as pd
import joblib
from preprocess import preprocess_data

def train_random_forest(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, y_train):
    xg_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xg_model.fit(X_train, y_train)
    return xg_model

if __name__ == "__main__":
    stock_data = pd.read_csv('../data/aapl_stock_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(stock_data)
    
    rf_model = train_random_forest(X_train, y_train)
    xg_model = train_xgboost(X_train, y_train)
    
    # Save models
    joblib.dump(rf_model, '../models/random_forest_model.pkl')
    joblib.dump(xg_model, '../models/xgboost_model.pkl')
