# 📈 Stock Price Predictor

An **interactive web demo** for stock price prediction and trading strategy backtesting.  
Showcasing advanced machine learning, technical analysis, and realistic simulation — packaged into a simple, intuitive Streamlit app.

🌐 [**Launch the Demo**](https://psplabs-stock-price-predictor.streamlit.app) - Interactive stock prediction and backtesting

This demo highlights how a production-ready ML framework can be adapted into a lightweight, shareable application for exploring predictive modeling in finance.

## ✨ Features

### 📊 Stock Price Prediction
- ML-driven forecasts using advanced algorithms + technical indicators.
- Visual overlays of RSI, MACD, Bollinger Bands, and Moving Averages.
- Configurable forecast horizon (1–30 days).

### 💹 Trading Strategy Backtesting
- Simulation with transaction costs and slippage included.
- Threshold-based and momentum-based strategies.
- Performance metrics: Sharpe ratio, drawdown, volatility, win rate.
- Interactive portfolio charts for analysis.

## 🛠️ Technology Stack
<table border="0">
<tr>
<td align="center">
<img src="assets/python-original.svg"width="40"/><br>Python</td>
 <td align="center"><img src="assets/pandas-original.svg" width="40"/><br>Pandas</td>
 <td align="center"><img src="assets/numpy-original.svg" width="40"/><br>NumPy</td>
 <td align="center"><img src="assets/scikit_learn_logo_small.svg.png" width="72"/><br>Scikit-learn</td>
 <td align="center"><img src="assets/MLflow-logo-final-black.png" width="104"/><br>MLflow</td>
 <td align="center"><img src="assets/pydantic.png" width="40"/><br>Pydantic</td>
 <td align="center"><img src="assets/plotly-original.svg" width="40"/><br>Plotly</td>
<td align="center"><img src="assets/streamlit-original.svg" width="40"/><br>Streamlit</td> 
</tr>
<table>

## 🏗️ Repository Structure
```
stock-price-predictor-demo/
├── 📱 app/
│   ├── app.py                                            # Main Streamlit application
│   ├── model/                                            # Pre-trained model artifacts
│   │   └── model.pkl
│   └── data/                                             # Sample datasets
│       └── sample_stocks_data.csv
├── wheels/
│   └── stock_price_predictor-1.0.0-py3-none-any.whl      # Framework package
├── requirements.txt                                      # Aplication dependencies
├── 📖 README.md                                          # Demo documentation
└── 🪙 LICENSE                                            # License
```

---

# 💻 How to Use

### Prediction
1. **Select stock** (AAPL, AMZN, TSLA, etc.)
2. **Configure parameters**: prediction horizon & indicators
3. **Run forecast**: ML model outputs price predictions
4. **Explore charts**: visual overlays of technical analysis

### Backtesting
1. **Choose strategy**: Threshold or Momentum
2. **Set parameters**: risk management, position sizing
3. **Run simulation**: realistic trade execution with costs
4. **Analyze results**: portfolio performance, drawdown, returns

## 📊 Key Metrics

| Metric            | What it tells you |
|-------------------|-------------------|
| **Total Return**  | Overall portfolio gain/loss |
| **Sharpe Ratio**  | Risk-adjusted return |
| **Max Drawdown**  | Largest drop from peak |
| **Win Rate**      | % of profitable trades |
| **Volatility**    | Portfolio risk level |


## 🎯 Why This Demo Matters

- **End-to-End ML Pipeline**: from raw data to prediction
- **Professional-Grade Indicators**: widely used in finance
- **Realistic Backtesting**: accounts for trading frictions
- **Interactive Visualization**: data-driven insights at a glance
- **Clean UI/UX**: accessible for both developers and analysts

## 🔧 Under the Hood

This demo leverages a custom ML framework with:
- **Modular Architecture** for scalability
- **MLflow Tracking** for experiments and model versioning
- **Feature Engineering** with advanced technical indicators
- **Simulations & Backtesting** for realistic strategy evaluation under market conditions  

## ⚠️ Demo Limitations

This demo is intended to showcase the core capabilities of the forecasting framework  and is **not a full-fledged trading platform**. To ensure clarity and focus, the current limitations include:
1. **Single Asset Focus**
   - The model currently supports **AMZN stock only**.
   - Other assets, indices, or ETFs are not included in this version.
2. **Short Forecast Horizon**
   - Forecasts are limited to a **15-day window**.
   - Predictions beyond 15 days are not supported and may produce inaccurate results.
3. **Historical Data Only**
   - The demo relies on **historical market data** for feature generation and model predictions.
   - Real-time trading integration is not implemented in this version.
4. **Simplified Simulation**
   - Backtesting and simulation features are **limited to illustrative purposes**.
   - Results should not be used for real-world investment decisions.
5. **No Risk Management**
   - This demo does **not include portfolio optimization, risk assessment, or capital allocation**.
   - Any suggested actions are purely for educational and demonstration purposes.
6. **Experimental Model**
   - This model represents an **early prototype**.
   - Accuracy is optimized for short-term AMZN predictions and may vary in other contexts.

> ⚡ **Note:** These constraints are intentional to highlight model performance in a **focused, controlled setting**. Future versions will expand assets, horizons, and functionality while maintaining reliability.

---
  
*For educational and research purposes only. Not financial advice.*