# 🚀 AI Crypto Prediction - BTC-USD Trading Bot

A machine learning project that predicts Bitcoin (BTC-USD) price movements using technical indicators and generates automated trading signals with interactive visualizations.

## 📋 Overview

This project uses a **Random Forest Classifier** to predict whether Bitcoin's price will go UP or DOWN based on historical price data and technical indicators. The model analyzes 3 years of daily price data and generates trading signals with an interactive dashboard.

## 🎯 Features

- **Historical Data Processing**: Automatic download of 3 years of BTC-USD daily price data
- **Technical Indicators**: 
  - Moving Averages (MA10, MA20)
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
  - Daily Returns
  - Volume Analysis

- **Machine Learning Model**: Random Forest Classifier with 100 estimators
- **Trading Signals**: Automated BUY (📈) and SELL (📉) signals
- **Interactive Dashboard**: Plotly-based visualization with multiple technical analysis charts
- **Performance Metrics**: Accuracy score and detailed classification reports

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the project**:
```bash
cd Ai-Crypto-Prediction
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
   ```bash
   .\venv\Scripts\Activate.ps1
   ```
   - **Windows (CMD)**:
   ```bash
   venv\Scripts\activate.bat
   ```
   - **macOS/Linux**:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🏃 Usage

Run the script:
```bash
python script.py
```

### Output

The script will:
1. Download 3 years of historical BTC-USD data
2. Calculate technical indicators
3. Train the Random Forest model
4. Display model accuracy and classification metrics
5. Generate next-day price prediction (UP ↑ or DOWN ↓)
6. Create interactive trading signals
7. Generate and display `trading_dashboard.html` in your browser

## 📊 Dashboard Features

The interactive dashboard (`trading_dashboard.html`) displays:

1. **Price Chart** with:
   - Candlestick price action
   - Moving Averages (MA10, MA20)
   - BUY signals (🟢 green triangles)
   - SELL signals (🔴 red triangles)

2. **Volume Bar Chart** - Color-coded by price direction
3. **RSI Indicator** - With overbought (70) and oversold (30) levels
4. **MACD Indicator** - For trend momentum analysis

## 📁 Project Structure

```
Ai-Crypto-Prediction/
├── script.py              # Main prediction script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── venv/                 # Virtual environment (created during setup)
└── trading_dashboard.html # Generated interactive chart (output)
```

## 🧠 Model Details

### Features Used
- `ma10`: 10-day Moving Average
- `ma20`: 20-day Moving Average
- `rsi`: Relative Strength Index
- `macd`: MACD indicator value
- `volume`: Trading volume
- `return`: Daily percentage return

### Target Variable
- `1`: Price increases next day (UP 📈)
- `0`: Price decreases next day (DOWN 📉)

### Model Configuration
- **Algorithm**: Random Forest Classifier
- **Estimators**: 100 trees
- **Train/Test Split**: 80/20 (chronological, no shuffle)
- **Test Size**: ~20% of data (most recent ~200 days)

## 📈 Interpreting Results

### Accuracy Score
Shows the percentage of correct predictions on the test set. A score above 52-55% is considered above random for this binary classification task.

### Classification Report
- **Precision**: Of predicted UPs/DOWNs, how many were correct
- **Recall**: Of actual UPs/DOWNs, how many were predicted correctly
- **F1-Score**: Harmonic mean of precision and recall

### Trading Signals
- **OPEN (BUY)**: 🟢 Green triangle - Signal to enter a long position
- **CLOSE (SELL)**: 🔴 Red triangle - Signal to exit the position

## ⚠️ Important Disclaimers

- **Educational Purpose**: This project is for learning and demonstration only
- **Not Financial Advice**: Do not use for actual trading without professional review
- **Historical Data**: Past performance does not guarantee future results
- **Market Risk**: Cryptocurrency markets are highly volatile and unpredictable
- **Model Limitations**: ML models can fail during unusual market conditions
- **Backtesting Bias**: The model is tested on the same data it learns from

## 🔧 Requirements

See `requirements.txt`:
- `pandas` - Data manipulation
- `numpy` - Numerical computation
- `yfinance` - Data download
- `scikit-learn` - Machine learning
- `ta` - Technical analysis indicators
- `plotly` - Interactive visualizations

## 🐛 Troubleshooting

### Issue: `yfinance` connection errors
- Try running the script again
- Check your internet connection
- Consider using a proxy if behind a firewall

### Issue: Memory errors with large datasets
- Reduce the period in `ticker.history()` from "3y" to "1y" or "6mo"

### Issue: Indicators show NaN values
- This is normal for the first ~20 rows due to moving average calculations
- The script automatically removes these with `dropna()`

## 📚 Learning Resources

- [Technical Analysis Indicators](https://en.wikipedia.org/wiki/Technical_analysis)
- [Random Forest Algorithm](https://scikit-learn.org/stable/modules/ensemble.html#forests)
- [Plotly Documentation](https://plotly.com/python/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)

## 🎓 Developed For

**Course**: Introduction to AI  
**Subject**: Algorithmic Trading & Machine Learning  
**Date**: 2026

---

**Note**: This is a mini-project for educational purposes. Always conduct your own research and consult with financial advisors before making any trading decisions.
