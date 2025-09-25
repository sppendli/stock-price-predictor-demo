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

FEATURE_CONFIG = {
    "version": 1.0,
    "feature_definitions": {
        "moving_averages":{
            "windows": [7, 14, 21],
            "columns": ["Close"]
        },
        "relative_strength_index": {
            "window": 14,
            "column": "Close"
        },
        "stochastic_oscillator": {
            "k_window": 14,
            "d_window": 3,
            "high_column": "High",
            "low_column": "Low",
            "close_column": "Close"
        },
        "bollinger_bands": {
            "window": 20,
            "column": "Close"
        },
        "average_true_range": {
            "window": 14,
            "high_column": "High",
            "low_column": "Low",
            "close_column": "Close"
        },
        "moving_average_convergence_divergence": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "column": "Close"
        },
        "on_balance_volume": {
            "close_column": "Close",
            "volume_column": "Volume"
        },
        "volume_weighted_average_price": {
            "high_column": "High",
            "low_column": "Low",
            "close_column": "Close",
            "volume_column": "Volume"
        },
        "rate_of_change": {
            "window": 14,
            "column": "Close"
        },
        "commodity_channel_index": {
            "window": 20,
            "high_column": "High",
            "low_column": "Low",
            "close_column": "Close"
        }
    },
    "temporal_features": {},
    "feature_selection": {
        "exclude":["Open", "High", "Low"],
        "target": "close_7d_future"
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