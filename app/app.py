"""
Stock Price Predictor Streamlit Application.

This module provides a Streamlit-based web interface for visualizing stock price predictions,
running trading simulations, and interacting with the stock price predictor framework.
It integrates MLflow for model management, feature engineering pipelines, and a trading simulator.
"""


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import mlflow
import mlflow.pyfunc
from typing import Optional, Dict, Any
import logging
import os
from pathlib import Path
import warnings
warnings.simplefilter("ignore")

from stock_price_predictor.data_processing.feature_engineering import FeaturePipeline
from stock_price_predictor.features.registry import registry
from stock_price_predictor.utils.config_loader import load_config
from stock_price_predictor.simulations.simulations import TradingSimulator, ThresholdStrategy, MomentumStrategy, PositionType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    """
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = False
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = None
    if 'target_col' not in st.session_state:
        st.session_state.target_col = None
    if 'forecast_horizon' not in st.session_state:
        st.session_state.forecast_horizon = 5

try:
    mlflow_config = load_config("mlflow_config", temp_file=False)
    mlflow.set_tracking_uri(mlflow_config["tracking"]["uri"])
    logger.info(f"MLflow tracking URI set to {mlflow_config['tracking']['uri']}")
except Exception as e:
    logger.error(f"Failed to load MLflow config: {str(e)}")
    st.error("Failed to configure MLflow. Please check your configuration.")

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """
    Load the production model from MLflow registry.

    Returns
    -------
    mlflow.pyfunc.PyFuncModel
        Loaded model instance, or None if loading fails.

    Raises
    ------
    Exception
        If model loading fails.
    """
    try:
        model_path = Path(__file__).parent / "model"
        model = mlflow.pyfunc.load_model(model_path.as_posix())
        logger.info(f"Successfully loaded model: {model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        return None
    
@st.cache_data
def load_data(file_path: str, tickers: list) -> pd.DataFrame:
    """
    Load and preprocess CSV data for selected tickers.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    tickers : list
        List of ticker symbols to filter.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with 'Date' as index.

    Raises
    ------
    Exception
        If CSV loading or processing fails.
    """
    try:
        data = pd.read_csv(file_path)
        if 'Ticker' in data.columns:
            data = data[data['Ticker'].isin(tickers)].set_index('Date')
            data.drop(columns=['Ticker'], inplace=True)
        else:
            data = data.set_index('Date')

        data.sort_index(inplace=True)
        
        logger.info(f"Successfully loaded CSV data: {len(data)} rows")
        return data
    
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        st.error(f"Failed to load CSV file: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def process_data(
        data: pd.DataFrame, 
        forecast_horizon: int = 5, 
        config_path: str = "feature_config"
    ) -> pd.DataFrame:
    """
    Process data using the feature engineering pipeline.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with raw stock data.
    forecast_horizon : int, optional
        Number of days to predict ahead (default is 5).
    config_path : str, optional
        Path to feature configuration file (default is "feature_config").

    Returns
    -------
    tuple
        Processed features (X) and target (y) as DataFrames.

    Raises
    ------
    Exception
        If feature pipeline processing fails.
    """
    try:
        feature_pipeline = FeaturePipeline(config_path="feature_config")
        experiment_config = load_config("experiment_config", temp_file=False)
        processed_data = feature_pipeline.transform(df=data, features_to_apply=None, forecast_horizon=forecast_horizon)
        target_col = experiment_config['target']
        X = processed_data.drop(columns=target_col)
        y = processed_data[target_col]

        logger.info(f"Successfully processed data using feature pipeline")
        return X, y
    
    except Exception as e:
        logger.error(f"Error in feature pipeline: {str(e)}")
        st.error(f"Feature pipeline failed: {str(e)}")
        return None, None
    
@st.cache_data
def make_predictions(data: pd.DataFrame, forecast_horizon: int):
    """
    Generate predictions using the loaded model.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with processed features.
    forecast_horizon : int
        Number of days to predict ahead.

    Returns
    -------
    np.ndarray
        Array of predictions.

    Raises
    ------
    Exception
        If prediction generation fails.
    """
    try:
        model = load_model()
        if model is None:
            return np.array([])
        
        X, y = process_data(data, forecast_horizon)
        if X is None or X.empty:
            return np.array([])
        
        predictions = model.predict(X.tail(forecast_horizon))

        logger.info(f"Made predictions using the registered model: {len(predictions)} forecasts")
        return predictions

    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        st.error(f"Prediction failed: {str(e)}")
        return np.array([])
    
def generate_rolling_predictions(
        data: pd.DataFrame, 
        forecast_horizon: int, 
        window_size: int = 50
    ):
    """
    Generate rolling predictions over a sliding window.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with stock data.
    forecast_horizon : int
        Number of days to predict ahead.
    window_size : int, optional
        Size of the rolling window (default is 50).

    Returns
    -------
    pd.Series
        Series of rolling predictions with dates as index.

    Raises
    ------
    Exception
        If rolling prediction generation fails.
    """
    try:
        model = load_model()
        if model is None:
            return np.array([])
        
        predictions, dates = [], []

        for i in range(window_size, len(data) - forecast_horizon):
            window_data = data.iloc[i-window_size:i]

            X, y = process_data(window_data, forecast_horizon)
            if X is None or X.empty:
                if i < window_size + 3:  # Only show first few failures
                    st.write(f"DEBUG ROLLING: No features generated for window {i}")
                continue
            if X is not None and not X.empty:
                pred = model.predict(X.tail(forecast_horizon))

                if hasattr(pred, '__len__') and len(pred) > 0:
                    pred_array = np.array(pred).flatten()
                    predictions.append(pred_array[-1])
                else:
                    predictions.append(np.full(forecast_horizon, float(pred)))
                dates.append(data.index[i])
        
        return pd.Series(predictions, index=dates)

    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        st.error(f"Prediction failed: {str(e)}")
        return pd.Series(dtype=float)
    
def prepare_simulation_data(
        data: pd.DataFrame, 
        target_col: str, 
        forecast_horizon: int
    ) -> pd.DataFrame:
    """
    Prepare data for trading simulations.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with stock data.
    target_col : str
        Name of the target column.
    forecast_horizon : int
        Number of days to predict ahead.

    Returns
    -------
    pd.DataFrame
        Prepared simulation DataFrame.

    Raises
    ------
    Exception
        If data preparation fails.
    """
    try:
        predictions = generate_rolling_predictions(data, forecast_horizon)

        sim_data = data.loc[predictions.index].copy()
        sim_data['y_pred'] = predictions
        sim_data['y_true'] = data['Close']
        sim_data['y_test'] = sim_data[target_col]

        sim_data = sim_data.dropna(subset=['y_test'])
    
        sim_data.reset_index(inplace=True)
        if "Date" in sim_data.columns:
            sim_data.rename(columns={"Date": "date"}, inplace=True)
        elif "index" in sim_data.columns:
            sim_data.rename(columns={"index": "date"}, inplace=True)

        return sim_data

    except Exception as e:
        logger.error(f"Error preparing simulation data: {str(e)}")
        st.error(f"Failed to prepare simulation data: {str(e)}")
        return pd.DataFrame()
    
@st.cache_data
def run_trading_simulations(preds_df: pd.DataFrame, sim_config: dict):
    """
    Run trading simulations with configurable strategies.

    Parameters
    ----------
    preds_df : pd.DataFrame
        DataFrame with predictions and true values.
    sim_config : dict
        Configuration dictionary for simulation parameters.

    Returns
    -------
    dict
        Simulation results including equity curve and metrics.

    Raises
    ------
    Exception
        If simulation execution fails.
    """
    try:
        sim = TradingSimulator(
            initial_capital=sim_config['params'].get('initial_capital', 10000),
            commission=sim_config['params'].get('commission', 0.001),
            slippage=sim_config['params'].get('slippage', 0.0005),
            position_sizing=sim_config['params'].get('position_sizing', 'fixed'),
            max_position_size=sim_config['params'].get('max_position_size', 0.95),
            risk_free_rate=sim_config['params'].get('risk_free_rate', 0.02)
        )

        strategy_type = sim_config['strategy'].get('type', 'threshold')
        if strategy_type == "threshold":
            strategy = ThresholdStrategy(
                long_threshold=sim_config['strategy'].get('long_threshold', 0.01),
                short_threshold=sim_config['strategy'].get('short_threshold', -0.01),
                name="threshold_strategy"
            )
        elif strategy_type == "momentum":
            strategy = MomentumStrategy(
                lookback=sim_config['strategy'].get('lookback', 5),
                momentum_threshold=sim_config['strategy'].get('momentum_threshold', 0.01)
            )
        else:
            strategy = ThresholdStrategy(
                long_threshold=sim_config['strategy'].get('long_threshold', 0.01),
                short_threshold=sim_config['strategy'].get('short_threshold', -0.01),
                name="threshold_strategy"
            )

        results = sim.run_simulation(
            preds_df=preds_df,
            strategy=strategy,
            start_date=sim_config.get('start_date'),
            end_date=sim_config.get('end_date'),
            y_true_col=sim_config.get('y_true_col', 'y_true'),
            forecast_horizon=sim_config.get('forecast_horizon', 5)
        )

        results['simulator'] = sim
        results['initial_capital'] = sim_config.get('initial_capital', 10000)
        results['final_equity'] = sim.equity_curve['equity'].iloc[-1]
        results['equity'] = sim.equity_curve
        results['trades'] = sim.trades
        results['performance_metrics'] = sim.performance_metrics.to_dict()

        logger.info(f"Trading simulation completed successfully")
        return results

    except Exception as e:
        logger.error(f"Error running trading simulation: {str(e)}")
        st.error(f"Simulation failed: {str(e)}")
        return None
    
def filter_by_date(data: pd.DataFrame, start_date: str, end_date:str) -> pd.DataFrame:
    """
    Filter DataFrame by a date range.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with 'Date' index.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.

    Raises
    ------
    Exception
        If date filtering fails.
    """
    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        mask = (data.index >= start_date) & (data.index <= end_date)
        filtered_data = data.loc[mask]

        if filtered_data.empty:
            st.warning(f"No data found in the selected data range({start_date.date()} to {end_date.date()})")
            st.info(f"Avaialbele date range: {data.index.min().date()} to {data.index.max().date()}")

        return filtered_data

    except Exception as e:
        logger.error(f"Error filtering data: {str(e)}")
        st.error(f"Error filtering data by date: {str(e)}")
        return pd.DataFrame()

def create_price_chart(
        data: pd.DataFrame, 
        predictions: Optional[np.ndarray] = None, 
        target_col: str = None
    ):
    """
    Create an interactive price chart with predictions.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with price data.
    predictions : np.ndarray, optional
        Array of predicted values (default is None).
    target_col : str, optional
        Name of the target column (default is None).

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()

    if target_col in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[target_col],
            mode='lines',
            name=f"Historical {target_col}",
            line=dict(color='blue', width=2)
        ))

    ma_cols = [col for col in data.columns if 'MA' in col or 'SMA' in col]
    colors = ['orange', 'red', 'purple', 'brown']
    for i, ma_col in enumerate(ma_cols[:4]):
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[ma_col],
            mode='lines',
            name=ma_col,
            line=dict(color=colors[i % len(colors)], width=1),
            opacity=0.7
        ))

    if predictions is not None and len(predictions) > 0:
        last_date = data.index[-1]
        pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predictions), freq='D')

        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=predictions,
            mode='lines+markers',
            name='Predictions',
            line=dict(color='green', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
    fig.update_layout(
        title="Stock Price Predictions",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        showlegend=True,
        height=500
    )

    return fig

def create_simulation_charts(results: dict):
    """
    Create charts for simulation results.

    Parameters
    ----------
    results : dict
        Dictionary containing simulation results.

    Returns
    -------
    tuple
        Tuple of (portfolio_fig, trades_fig) as go.Figure objects.
    """
    if not results or 'equity' not in results:
        return None, None
    
    portfolio_fig, trades_fig = None, None

    if 'equity' in results:
        equity_curve = results['equity'].copy()

        if 'date' not in equity_curve.columns:
            if isinstance(equity_curve.index, pd.DatetimeIndex):
                equity_curve = equity_curve.reset_index().rename(columns={'index':'date'})
            else:
                equity_curve = equity_curve.reset_index().rename(columns={equity_curve.index.name or 'index': 'date'})
        
        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(go.Scatter(
            x=equity_curve['date'],
            y=equity_curve['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='green', width=2)
        ))

        if 'benchmark_values' in results:
            portfolio_fig.add_trace(go.scatter(
                x=equity_curve['date'],
                y=results['benchmark_values'],
                mode='lines',
                name='Buy & Hold Benchmark',
                line=dict(color='blue', width=2, dash='dash')
            ))

        portfolio_fig.update_layout(
            title="📈 Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            hovermode='x unified',
            height=400
        )
    
    if 'trades' in results and len(results['trades']) > 0:
        trades_df = pd.DataFrame(results['trades'])
        trades_fig = go.Figure()

        buy_trades = trades_df[trades_df['position_type'] == PositionType.LONG]
        if not buy_trades.empty:
            trades_fig.add_trace(go.Scatter(
                x=buy_trades['entry_date'],
                y=buy_trades['entry_price'],
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))

        sell_trades = trades_df[trades_df['position_type'] == PositionType.SHORT]
        if not sell_trades.empty:
            trades_fig.add_trace(go.Scatter(
                x=sell_trades['entry_date'],
                y=sell_trades['entry_price'],
                mode='markers',
                name='Sell',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))

        trades_fig.update_layout(
            title="🎯 Trading Signals",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=400
        )
        
    return portfolio_fig, trades_fig
    
def create_metrics_summary(
        data: pd.DataFrame, 
        predictions: Optional[np.ndarray] = None, 
        target_col: str = None
    ) -> Dict[str, Any]:
    """
    Create a summary of stock metrics and predictions.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with price data.
    predictions : np.ndarray, optional
        Array of predicted values (default is None).
    target_col : str, optional
        Name of the target column (default is None).

    Returns
    -------
    Dict[str, Any]
        Dictionary of calculated metrics.
    """
    if data.empty or target_col not in data.columns:
        return {}

    data = data.dropna()
    current_price = data[target_col].iloc[-1]
    daily_return = data[target_col].pct_change().iloc[-1]
    volatility = data[target_col].rolling(window=20).std().iloc[-1]

    metrics = {
        "Current Price": f"${current_price:.2f}",
        "Daily Return": f"{daily_return*100:.2f}%",
        "20-Day Volatility": f"${volatility:.2f}",
        "Period High": f"{data[target_col].max():.2f}",
        "Period Low": f"{data[target_col].min():.2f}"
    }

    if predictions is not None and len(predictions) > 0:
        pred_return = (predictions[-1] - current_price) / current_price
        metrics["Predicted Next Price"] = f"${predictions[-1]:.2f}"
        metrics["Predicted Return"] = f"{pred_return*100:.2f}%"

    return metrics

def display_simulation_metrics(results: dict):
    """
    Display simulation performance metrics in a Streamlit layout.

    Parameters
    ----------
    results : dict
        Dictionary containing simulation results.
    """
    if not results:
        return
    
    initial_value = results.get('initial_capital', 0)
    final_value = results.get('final_equity', 0)
    total_return = ((final_value - initial_value) / initial_value * 100) if initial_value > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Initial Capital", f"${initial_value:,.2f}")
        st.metric("Final Value", f"${final_value:,.2f}")

    with col2:
        st.metric("Total Return", f"{total_return:.2f}%",
                  delta=f"{total_return:.2f}%" if total_return != 0 else None)
        if 'max_drawdown' in results['performance_metrics']:
            st.metric("Max Drawdown", f"{results['performance_metrics']['max_drawdown']:.2f}%")
        
    with col3:
        if 'sharpe_ratio' in results['performance_metrics']:
            st.metric("Sharpe Ratio", f"{results['performance_metrics']['sharpe_ratio']:.3f}")
        if 'total_trades' in results['performance_metrics']:
            st.metric("Total Trades", str(results['performance_metrics']['total_trades']))

    with col4:
        if 'win_rate' in results['performance_metrics']:
            st.metric("Win Rate", f"{results['win_rate']:.1f}%")
        if 'benchmark_return' in results:
            excess_return = results['performance_metrics'].get('total_return', 0) - results['benchmark_return']
            st.metric("vs Benchmark", f"{excess_return:+.2f}%")

def display_csv_info(data: pd.DataFrame):
    """
    Display summary information about loaded CSV data.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with stock data.
    """
    if not data.empty:
        st.success("✅ CSV loaded successfully!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Date Range", f"{data.index.min()} to {data.index.max()}")
        with col3:
            st.metric("Features", len(data.columns))

def display_pipeline_info():
    """
    Display information about the production pipeline architecture.
    """
    st.subheader("🏗️ Production Pipeline Architecture")

    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**🔧 Core Components**")
        st.write("- **Feature Pipeline**: Advanced feature engineering")
        st.write("- **Feature Registry**: Centralized feature management")
        st.write("- **Config Loader**: Dynamic configuration management")
        st.write("- **MLflow Integration**: Model versioning & deployment")
        st.write("- **Trading Simulator**: Strategy backtesting engine")

    with col2:
        st.success("**⚡ Pipeline Benefits**")
        st.write("- **Modular Architecture**: Easily Extensible")
        st.write("- **Configuration Driven**: No hard-coded parameters")
        st.write("- **Production Ready**: Error handling & logging")
        st.write("- **Scalable Design**: Handles multiple assets")
        st.write("- **Backtesting**: Real-world trading simulation")

def trading_simulation_section(
        data: pd.DataFrame, 
        predictions: np.ndarray, 
        target_col: str, 
        forecast_horizon: int
    ):
    """
    Display and manage trading simulation section.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with stock data.
    predictions : np.ndarray
        Array of predicted values.
    target_col : str
        Name of the target column.
    forecast_horizon : int
        Number of days to predict ahead.
    """
    st.subheader("🎮 Trading Strategy Simulation")
    st.write("Test how your predictions would perform in a simulated trading environment")

    with st.expander("⚙️ Simulation Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**💰 Capital & Risk Management**")
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000,
                key="sim_initial_capital"
            )

            commission = st.number_input(
                "Commission (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                format="%.3f",
                key="sim_commission"
            ) / 100

            slippage = st.number_input(
                "Slippage (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                format="%.3f",
                key="sim_slippage"
            ) / 100

            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=10,
                max_value=100,
                value=95,
                key="sim_max_position_size"
            ) / 100

        with col2:
            st.write("**📊 Strategy Parameters**")

            strategy_type = st.selectbox(
                "Strategy Type",
                options=['threshold', 'momentum'],
                index=0,
                help="Select the trading strategy to use",
                key="sim_strategy_type"
            )

            if strategy_type == 'threshold':
                long_threshold = st.number_input(
                    "Long Threshold (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key="sim_long_threshold"
                ) / 100

                short_threshold = st.number_input(
                    "Short Threshold (%)",
                    min_value=-10.0,
                    max_value=-0.1,
                    value=-1.0,
                    step=0.1,
                    key="sim_short_threshold"
                ) / 100

            elif strategy_type == 'momentum':
                lookback = st.number_input(
                    "Lookback Period (days)",
                    min_value=2,
                    max_value=30,
                    value=5,
                    step=1,
                    key="sim_lookback"
                )

                momentum_threshold = st.number_input(
                    "Momentum Threshold (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key="sim_momentum_threshold"
                ) / 100

    if st.button("🚀 Run Trading Simulation", type="primary"):
        with st.spinner("Running trading simulation..."):
            try:
                sim_data = prepare_simulation_data(data, target_col, forecast_horizon)

                if sim_data.empty:
                    st.error("Failed to prepare simulation data.")
                    return
                
                sim_config = {}
                
                sim_config['params'] = {
                    'initial_capital': initial_capital,
                    'commission': commission,
                    'slippage': slippage,
                    'position_sizing': 'fixed',
                    'max_position_size': max_position_size,
                    'risk_free_rate': 0.02,
                    'forecast_horizon': forecast_horizon,
                    'y_true_col': 'y_true'
                }

                sim_config['strategy'] = {
                    'type': strategy_type
                }

                if sim_config['strategy']['type'] == "threshold":
                    sim_config['strategy'].update({
                        'long_threshold': long_threshold,
                        'short_threshold': short_threshold
                    })

                elif sim_config['strategy']['type'] == "momentum":
                    sim_config['strategy'].update({
                        'lookback': lookback,
                        'momentum_threshold': momentum_threshold
                    })

                results = run_trading_simulations(sim_data, sim_config)

                if results:
                    st.success("✅ Simulation completed successfully!")

                    display_simulation_metrics(results)

                    portfolio_fig, trades_fig = create_simulation_charts(results)

                    col1, col2 = st.columns(2)
                    with col1:
                        if portfolio_fig is not None:
                            st.plotly_chart(portfolio_fig, use_container_width=True)
                    with col2:
                        if trades_fig:
                            st.plotly_chart(trades_fig, use_container_width=True)

                    with st.expander("📊 Detailed Performance Metrics"):
                        if 'performance_metrics' in results:
                            metrics = results['performance_metrics']
                            metrics_dict = {
                                'Total Return (%)': f"{metrics["total_return"] * 100:.2f}%",
                                'Annualized Return (%)': f"{metrics["annualized_return"] * 100:.2f}%",
                                'Sharpe Ratio': f"{metrics["sharpe_ratio"]:.3f}",
                                'Sortino Ratio': f"{metrics["sortino_ratio"]:.3f}",
                                'Max Drawdown (%)': f"{metrics["max_drawdown"] * 100:.2f}%",
                                'Calmar Ratio': f"{metrics["calmar_ratio"]:.3f}",
                                'Win Rate (%)': f"{metrics["win_rate"] * 100:.1f}%",
                                'Profit Factor': f"{metrics["profit_factor"]:.2f}",
                                'Total Trades': str(metrics["total_trades"]),
                                'Winning Trades': str(metrics["winning_trades"]),
                                'Losing Trades': str(metrics["losing_trades"]),
                                'Avg Trade Duration (days)': f"{metrics["avg_trade_duration"]:.1f}"
                            }

                            for i, (key, value) in enumerate(metrics_dict.items()):
                                if i % 3 == 0:
                                    cols = st.columns(3)
                                with cols[i % 3]:
                                    st.metric(key, value)

                    if 'trades' in results and len(results['trades']) > 0:
                        with st.expander("📋 Trading Log"):
                            trades_data = []
                            for trade in results['trades']:
                                trades_data.append({
                                    'Entry Date': trade.entry_date.strftime('%Y-%m-%d'),
                                    'Exit Date': trade.exit_date.strftime('%Y-%m-%d'),
                                    'Position': trade.position_type.name,
                                    'Entry Price': f"${trade.entry_price:.2f}",
                                    'Exit Price': f"${trade.exit_price:.2f}",
                                    'Quantity': f"{trade.quantity:.2f}",
                                    'P&L': f"${trade.pnl:.2f}",
                                    'Return %': f"{trade.return_pct * 100:.2f}%",
                                    'Duration (days)': str(trade.duration_days)
                                })
                            trades_df = pd.DataFrame(trades_data)
                            if trades_df is not None and not trades_df.empty:
                                st.dataframe(trades_df, use_container_width=True)
                    
                    with st.expander("📈 Equity Curve Details"):
                        if 'equity' in results:
                            equity_curve = results['equity'].copy()
                            equity_curve.reset_index(inplace=True)

                            display_columns = ['date', 'equity', 'current_price', 'signal', 'cash', 'current_position']
                            available_columns = [col for col in display_columns if col in equity_curve.columns]

                            if available_columns:
                                st.dataframe(equity_curve[available_columns].tail(20), use_container_width=True)

                    with st.expander("📊 Strategy Analysis"):
                        st.write("**Strategy Performance Summary:**")
                        if 'performance_metrics' in results:
                            metrics = results['performance_metrics']
                            perf_text = f"""
                            **Trading Strategy:** {strategy_type.title()} Strategy
                            **Initial Capital:** ${initial_capital:,.2f}
                            **Final Portfolio Value:** ${results.get('final_equity', 0):,.2f}
                            **Total Return:** {metrics["total_return"] * 100:.2f}%
                            **Annualized Return:** {metrics["annualized_return"] * 100:.2f}%
                            **Max Drawdown:** {metrics["max_drawdown"] * 100:.2f}%
                            **Sharpe Ratio:** {metrics["sharpe_ratio"]:.3f}
                            **Total Trades:** {metrics["total_trades"]}
                            **Win Rate:** {metrics["win_rate"] * 100:.1f}%
                            **Transaction Costs:** Commission: {commission*100:.2f}%, Slippage: {slippage*100:.2f}%
                            """
                            st.text(perf_text)
                else:
                    st.error("Simulation failed. Please check your data and settings.")

            except Exception as e:
                st.error(f"Simulation error: {str(e)}")
                logger.error(f"Simulation error: {str(e)}")
    
def main():
    """
    Main function to run the Streamlit application.
    """
    initialize_session_state()

    st.title("📈 Stock Price Predictor")
    st.markdown("**Powered by Custom ML Framework, MLflow Model Registry & Trading Simulation Engine**")
    st.markdown("---")

    st.sidebar.header("⚙️ Configuration")

    st.sidebar.subheader("📁 Data Source")
    data_source  = st.sidebar.radio(
        "Choose Data Source",
        ("Use Default Dataset", "Upload a CSV File")
    )

    if data_source == "Upload a CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload a CSV file owth columns: Date, Ticker, Open, High, Low, Close, Volume"
        )

    available_tickers = st.sidebar.multiselect(
        "Select Tickers",
        options=['AAPL', 'GOOGL', 'AMZN', 'TSLA', 'MSFT', 'NVDA'],
        default=['AMZN'],
        help="Select which stock tickers to analyze"
    )

    with st.sidebar.expander("📋 CSV Format Requirements"):
        st.write("""
        **Required columns:**
        - Date, Ticker, Open, High, Low, Close, Volume

        **Sample format:**
        ```
        Date, Ticker, Open, High, Low, Close, Volume
        2024-01-01, AMZN, 150.00, 155.00, 19.00, 154.00, 1000000
        ```
        """)

    data = pd.DataFrame()

    if data_source == "Upload a CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV File",
            type=['csv'],
            help="Upload a CSV file with columns: Date, Ticker, Open, High, Low, Close, Volume"
        )

        if uploaded_file is not None:
            temp_file = f"temp_{uploaded_file.name}"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())        
            data = load_data(temp_file, available_tickers)

            os.remove(temp_file)
    else:
        default_path = Path(__file__).parent / "data" / "AMZN_Test_Data.csv"
        if default_path.exists():
            data = load_data(default_path, available_tickers)
        else:
            st.error("Default dataset not found. Please upload a CSV file.")

    if not data.empty:
        display_csv_info(data)

    if not data.empty:
        st.sidebar.subheader("📅 Data Range")
        data.index = pd.to_datetime(data.index)
        min_date = data.index.min().date()
        max_date = data.index.max().date()

        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=max(min_date, max_date - timedelta(days=365)),
                min_value=min_date,
                max_value=max_date,
                help="Filter data from this date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                help="Filter data to this date"
            )

        st.sidebar.subheader("🔮 Prediction Settings")
        forecast_horizon = st.sidebar.slider(
            "Forecast Horizon",
            min_value=1,
            max_value=30,
            value=5,
            help="Number of days to predict ahead"
        )

        display_pipeline_info()

        if st.sidebar.button("🚀 Predict", type="primary"):
            with st.spinner("Processing data through production pipeline..."):
                filtered_data = filter_by_date(data, str(start_date), str(end_date))

                if not filtered_data.empty:
                    experiment_config = load_config("experiment_config")
                    target_col = experiment_config.get('target')
                    
                    filtered_data[target_col] = filtered_data["Close"].shift(-forecast_horizon)
                    predictions = make_predictions(filtered_data, forecast_horizon)

                    st.session_state.filtered_data = filtered_data
                    st.session_state.predictions = predictions
                    st.session_state.target_col = target_col
                    st.session_state.forecast_horizon = forecast_horizon
                    st.session_state.predictions_made = True
                
        if st.session_state.predictions_made and st.session_state.predictions is not None:
            filtered_data = st.session_state.filtered_data
            predictions = st.session_state.predictions
            target_col = st.session_state.target_col
            forecast_horizon = st.session_state.forecast_horizon
            
            col1, col2 = st.columns([3, 1])

            with col1:
                chart = create_price_chart(filtered_data, predictions, target_col)
                st.plotly_chart(chart, use_container_width= True)

            with col2:
                st.subheader("📋 Summary")
                metrics = create_metrics_summary(filtered_data, predictions, target_col)

                for key, value in metrics.items():
                    st.metric(key, value)

            if predictions is not None  and len(predictions) > 0:
                trading_simulation_section(filtered_data, predictions, target_col, forecast_horizon)

            st.subheader("📊 Feature Engineering Pipeline")
            X, y = process_data(data=filtered_data, forecast_horizon=forecast_horizon)

            if X is not None and not X.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.info(f"**Generated Features: {X.shape[1]}**")
                    feature_categories = {
                        'Price Features': [col for col in X.columns if any(x in col.lower() for x in ['open', 'high', 'low', 'close'])],
                        'Technical Indicators': [col for col in X.columns if any(x in col.lower() for x in ['ma', 'rsi', 'bb', 'macd'])],
                        'Volume Features': [col for col in X.columns if 'volume' in col.lower()],
                        'Lag Features': [col for col in X.columns if 'lag' in col.lower()],
                        'Other Features': []
                    }
                    used_features = set()
                    for category, features in feature_categories.items():
                        used_features.update(features)
                    feature_categories['Other Features'] = [col for col in X.columns if col not in used_features]

                    for category, features in feature_categories.items():
                        if features:
                            st.write(f"**{category}:** {len(features)}")

                with col2:
                    st.success(f"**Pipeline Output Shape**")
                    st.write(f"- **Features (X):** {X.shape}")
                    st.write(f"- **Target (y):** {y.shape if y is not None else 'N/A'}")
                    st.write(f"- **Forecast Horizon:** {forecast_horizon} days")
                    st.write(f"- **Target Column:** {target_col}") 

            with st.expander("🔍 Feature Pipeline Details"):
                if X is not None and not X.empty:
                    st.write("**Generated Features:**")                           
                    feature_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Type': ['Numeric'] * len(X.columns),
                        'Latest Value': X.iloc[-1].values
                    })
                    st.dataframe(feature_df, use_container_width=True)

            with st.expander("📄 Raw Data Sample"):
                st.dataframe(filtered_data.tail(20), use_container_width=True)

            with st.expander("🤖 Production Pipeline Details"):
                st.write("**Model Registry:** stock-price-predictor-baseline (Production Stage)")
                st.write("**Pipeline Architecture:**")
                st.code("""
                📊 Stock Price Predictor Framework
                ├── 🔧 Data Processing
                │   ├── CSV Loader with multi-ticker support
                │   ├── Date filtering and validation
                │   └── Data quality checks
                ├── ⚙️ Feature Engineering Pipeline
                │   ├── FeaturePipeline (config-driven)
                │   ├── Feature Registry integration
                │   ├── Dynamic feature generation
                │   └── Target variable preparation
                ├── 🎯 Prediction Pipeline
                │   ├── MLflow model loading (Production stage)
                │   ├── Feature preprocessing
                │   ├── Batch prediction capability
                │   └── Error handling & fallbacks
                ├── 🎮 Trading Simulation Engine
                │   ├── TradingSimulator with configurable parameters
                │   ├── ThresholdStrategy & MomentumStrategy
                │   ├── Performance metrics calculation
                │   └── Risk management controls
                └── 📈 Visualization & Analytics
                    ├── Interactive price charts
                    ├── Portfolio performance tracking
                    ├── Trading signals visualization
                    └── Pipeline monitoring
                """)
                st.write("**Configuration Management:**")
                st.write("- **Feature Config:** Defines feature engineering parameters")
                st.write("- **Experiment Config:** Controls model and target settings")
                st.write("- **MLflow Config:** Model registry and tracking settings")
                st.write("- **Dynamic Loading:** No hard-coded configurations")

                st.write(f"**Data Summary:**")
                st.write(f"- **Source:** Uploaded CSV({len(filtered_data)} records)")
                st.write(f"- **Tickers:** {', '.join(available_tickers)}")
                st.write(f"- **Date Range:** {start_date} to {end_date}")
                st.write(f"- **Framework:** Custom Stock Price Predictor Package")

    else:
        st.info("👆 Please upload a CSV file to get started")

        display_pipeline_info()

        st.subheader("📊 Sample Multi-Ticker CSV Format")
        sample_data = {
            'Date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
            'Ticker': ['AMZN', 'AAPL', 'AMZN', 'AAPL'],
            'Open': [150.00, 180.00, 152.00, 182.00],
            'High': [155.00, 185.00, 156.00, 186.00],
            'Low': [149.00, 179.00, 151.00, 181.00],
            'Close': [154.00, 184.00, 153.00, 183.00],
            'Volume': [1000000, 2000000, 1200000, 2200000]
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

        st.subheader("🎮 Trading Strategy Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**📊 Simulation Capabilities**")
            st.write("- **Threshold Strategy**: Configurable buy/sell thresholds")
            st.write("- **Momentum Strategy**: Trend-following based on prediction momentum")
            st.write("- **Transaction Costs**: Commission and slippage modeling")
            st.write("- **Risk Management**: Position sizing controls")
            st.write("- **Performance Metrics**: Comprehensive risk-adjusted metrics")
            
        with col2:
            st.success("**📈 Analysis Features**")
            st.write("- **Portfolio Tracking**: Real-time value monitoring")
            st.write("- **Benchmark Comparison**: vs Buy & Hold performance")
            st.write("- **Trade Visualization**: Buy/sell signal charts")
            st.write("- **Detailed Reporting**: Complete trading log with P&L")
            st.write("- **Risk Metrics**: Sharpe, Sortino, Calmar ratios")

    st.markdown("---")
    st.markdown(
        "🏗️ **Powered by Custom ML Framework** | "
        "📊 **MLflow Model Registry** | "
        "🎮 **Advanced Trading Simulation** | "
        "⚡ **Production-Ready Pipeline**"
    )
    st.markdown(
        "💡 **Note:** This system showcases enterprise-grade ML engineering practices. "
        "Predictions and simulations should not be used for actual trading decisions."
    )


if __name__ == "__main__":
    main()