
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Create Lagged Features
    data['Close_lag_1'] = data['Close'].shift(1)
    data['Close_lag_3'] = data['Close'].shift(3)
    data['Close_lag_5'] = data['Close'].shift(5)

    # Create Moving Average Features
    data['Close_MA_7'] = data['Close'].rolling(window=7).mean()
    data['Close_MA_30'] = data['Close'].rolling(window=30).mean()

    # Handle NaNs
    data.dropna(inplace=True)

    # Define Features (X) and Target (y)
    X = data[['Open', 'High', 'Low', 'Volume', 'Close_lag_1', 'Close_lag_3', 'Close_lag_5', 'Close_MA_7', 'Close_MA_30']]
    y = data['Close']
    
    # Train-test split - ensure y is also sliced according to X's new index after dropna
    # X and y are now aligned after dropna, so direct splitting is fine.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler
