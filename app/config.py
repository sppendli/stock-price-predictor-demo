"""
Configuration settings for Stock Price Predictor Streamlit App.
Contains all configuration dictionaries that replace YAML config files.
"""

EXPERIMENT_CONFIG = {
    "experiment_name": "stock_price_prediction",
    "version": "1.0.0",
    "model": {
        "name": "StockPricePredictor",
        "class": "sklearn.ensemble.RandomForestRegressor",
        "type":"linear_regression",
        "artifact_path": "test_model",
        "params":{
            "n_estimators": 1000,
            "max_depth": 8,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "max_features": "sqrt"
        }
    },
    "data": {
        "path": "data/AMZN_Test_Data.csv",
        "date_col": "Date",
        "test_size": 0.2,
        "version": "2024-08"
    },
    "validation": {
        "strategy": "expanding_window",
        "n_splits": 5,
        "train_size": 10, # days
        "test_size": 5,  # days
        "forecast_horizon": 5
    },
    "target": "close_7d_future",
    "promotion_thresholds": {
        "mae": 2.5,
        "rmse": 3.5,
        "r2": 0.85
    }
}

MLFLOW_CONFIG = {
    "tracking": {
        "uri": "http://127.0.0.1:5000",
        "experiment_name": "Baseline",
        "registry_uri": "sqlite:///models/mlflow/registry.db",
        "run_name": "RF Regressor with Fcst Horizon",
        "description": "Stock price prediction experiment"
    },
    "autolog": {
        "enabled": True,
        "log_input_examples": True,
        "log_model_signatures": True,
        "log_models" : True
    },
    "model_registry": {
        "stage": "Production"
    }
}