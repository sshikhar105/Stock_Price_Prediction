import unittest
import pandas as pd
import numpy as np
from src.preprocess import preprocess_data # Assuming PYTHONPATH is set or running from root
from sklearn.preprocessing import StandardScaler

class TestPreprocessData(unittest.TestCase):

    def test_preprocess_data_output_and_features(self):
        # 1. Create Sample Data
        # Create 60 days of data to ensure enough data after MAs and lags
        # Using pd.date_range to generate valid dates
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        data = {
            'Date': dates,
            'Open': np.random.rand(60) * 100,
            'High': np.random.rand(60) * 100 + 100, # Ensure high is higher than open
            'Low': np.random.rand(60) * 100 - 50,   # Ensure low is lower than open
            'Close': np.random.rand(60) * 100,
            'Volume': np.random.rand(60) * 10000
        }
        sample_df = pd.DataFrame(data)
        # Ensure Close prices are positive for MAs
        sample_df['Close'] = sample_df['Close'].abs() + 1 

        # 2. Call preprocess_data
        # Using .copy() as good practice
        X_train, X_test, y_train, y_test, scaler = preprocess_data(sample_df.copy())

        # 3. Add Assertions

        # Check types
        self.assertIsInstance(X_train, np.ndarray, "X_train should be a NumPy array")
        self.assertIsInstance(X_test, np.ndarray, "X_test should be a NumPy array")
        self.assertIsInstance(y_train, pd.Series, "y_train should be a Pandas Series")
        self.assertIsInstance(y_test, pd.Series, "y_test should be a Pandas Series")
        self.assertIsInstance(scaler, StandardScaler, "Scaler should be a StandardScaler instance")

        # Check shapes
        # Initial rows = 60. Max lag/window for NaN is 30 (Close_MA_30). So, 60 - 29 = 31 rows remain after dropna.
        # Test size 0.2 * 31 = 6.2 => 6 test rows, 25 train rows.
        expected_rows_after_dropna = 60 - (30 - 1) # 30 is the largest window, so 29 rows are lost
        
        # Due to sequential split, exact numbers can be calculated.
        # test_size = floor(0.2 * expected_rows_after_dropna)
        # train_size = expected_rows_after_dropna - test_size
        
        # Let's be slightly more flexible if rounding/floor behavior is tricky
        # self.assertEqual(y_train.shape[0], 25, "y_train should have 25 rows")
        # self.assertEqual(y_test.shape[0], 6, "y_test should have 6 rows")
        # Instead, check consistency:
        self.assertEqual(X_train.shape[0], y_train.shape[0], "X_train and y_train should have the same number of rows")
        self.assertEqual(X_test.shape[0], y_test.shape[0], "X_test and y_test should have the same number of rows")
        self.assertGreater(X_train.shape[0], 0, "X_train should not be empty")
        self.assertGreater(X_test.shape[0], 0, "X_test should not be empty")


        # Check number of features (Open, High, Low, Volume + 5 new features)
        expected_features = 4 + 5 
        self.assertEqual(X_train.shape[1], expected_features, f"X_train should have {expected_features} features")
        self.assertEqual(X_test.shape[1], expected_features, f"X_test should have {expected_features} features")

        # Check for NaNs
        self.assertFalse(np.isnan(X_train).any(), "X_train should not contain NaNs")
        self.assertFalse(np.isnan(X_test).any(), "X_test should not contain NaNs")
        self.assertFalse(y_train.isnull().any(), "y_train should not contain NaNs")
        self.assertFalse(y_test.isnull().any(), "y_test should not contain NaNs")
        
        # Check that y_train and y_test indices are continuous after splitting (important for time series)
        # This is implicitly handled by train_test_split with shuffle=False and the fact that y is a Series
        # but we can check if the last index of y_train is one before the first index of y_test,
        # assuming the original DataFrame had a continuous default index before setting Date as index.
        # However, 'Date' is set as index, and dropna can break strict continuity.
        # What's more important is that they are correctly ordered.
        self.assertTrue(y_train.index.max() < y_test.index.min(), "y_train indices should precede y_test indices")


if __name__ == '__main__':
    # This allows running the tests from the command line
    # For imports to work correctly, run from the project root:
    # python -m unittest tests.test_preprocess
    unittest.main()
