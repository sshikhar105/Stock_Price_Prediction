# Stock Price Prediction

This project uses historical stock price data to predict future prices using Random Forest and XGBoost models. It includes data collection, preprocessing with feature engineering, model training with hyperparameter tuning, and comprehensive evaluation against a naive baseline.

## Prerequisites

- Python 3.x
- Install the dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## How to Run

### 1. **Download stock data**
   - The default stock data used in this project is for Apple (AAPL) stock from Yahoo Finance. To download it, run:
     ```bash
     python src/data_collection.py
     ```
   - This saves the raw data to `data/aapl_stock_data.csv`.
   - **Note**: To use data for a different stock, update the stock ticker in `src/data_collection.py`.

### 2. **Preprocess data and train models**
   - Once the raw data is downloaded, run the preprocessing and model training script:
     ```bash
     python src/model_training.py
     ```
   - **Data Preprocessing involves:**
     - Converting the 'Date' column and setting it as the index.
     - Engineering new features:
       - Lagged 'Close' prices (1, 3, and 5 days).
       - Moving averages of 'Close' prices (7-day and 30-day).
     - Handling NaNs introduced by these operations by dropping rows.
     - Performing a sequential train-test split (data is not shuffled to respect time series nature).
     - Scaling features using `StandardScaler` (fitted on training data only).
   - The script saves the processed data (`X_train`, `X_test`, `y_train`, `y_test`) and the `scaler` object to the `processed_data/` directory.
   - **Model Training involves:**
     - Training Random Forest and XGBoost regressors.
     - Performing hyperparameter tuning for both models using `GridSearchCV` with `TimeSeriesSplit` for cross-validation, which is suitable for time-series data.
     - The best parameters found during tuning are printed to the console.
     - The tuned models are saved to the `models/` directory (e.g., `random_forest_model.pkl`, `xgboost_model.pkl`).

### 3. **Evaluate the models**
   - To evaluate the models and visualize the actual vs predicted stock prices:
     ```bash
     python src/evaluation.py
     ```
   - This script now loads the processed data (`X_test`, `y_test`, `y_train`) from the `processed_data/` directory and the trained models from the `models/` directory.
   - **Evaluation includes:**
     - Calculating Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² score for both trained models.
     - Evaluating a **naive baseline model** (predicting the previous day's close) for comparison.
     - Printing metrics for all models.
     - Generating and saving plots comparing actual vs. predicted stock prices for each model to the `results/` directory (e.g., `results/random_forest_predictions.png`).

### 4. **Running Tests**
   - Unit tests are provided for the preprocessing functionality. To run them:
     ```bash
     python -m unittest discover tests
     ```
   - Ensure you are in the project root directory when running this command.

## Results

The "Results" section in the original README, including example metrics and plots, serves as an illustration. Actual results may vary based on the dataset, features, and tuning. The evaluation script will print the latest metrics and save updated plots.

### Example Plots
Plots are saved in the `results/` directory.

   - **Random Forest Actual vs Predicted Prices**:
     ![Random Forest Plot](results/Random_Forest_predictions.png)

   - **XGBoost Actual vs Predicted Prices**:
     ![XGBoost Plot](results/XGBoost_predictions.png)

   - **Naive Baseline Actual vs Predicted Prices**: (Example filename, actual might vary)
     ![Naive Baseline Plot](results/naive_baseline_predictions.png)


## Directory Structure

- `data/`: Stores downloaded raw stock data (e.g., `aapl_stock_data.csv`).
- `processed_data/`: Stores processed data (e.g., `X_train.pkl`, `y_test.pkl`) and the scaler (`scaler.pkl`).
- `models/`: Stores trained and tuned machine learning models.
- `results/`: Stores generated graphs, such as model performance and actual vs predicted prices.
- `src/`: Contains the Python source code for data collection, preprocessing, model training, and evaluation.
- `tests/`: Contains unit tests for the project.

## License
This project is open source and available under the [MIT License](LICENSE). (Assuming a LICENSE file exists or will be added - if not, this line can be removed or updated).
