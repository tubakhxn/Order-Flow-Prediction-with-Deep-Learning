# Order Flow Prediction with Deep Learning

## Overview
This project is an advanced research-grade Streamlit application for predicting short-term price pressure in cryptocurrency markets using deep learning. It analyzes order book snapshots and trade data, or generates synthetic order book data, to construct features and train a neural network (CNN or LSTM) to forecast upward or downward price movements.

## Features
- **Data Input:** Upload your own order book CSV or use the built-in synthetic data generator.
- **Feature Engineering:** Automatically extracts bid/ask depth tensors, order book imbalance, volume delta, and future returns.
- **Deep Learning Model:** Select between CNN and LSTM architectures (PyTorch) for prediction.
- **Training:** Rolling window training with real-time accuracy and confidence metrics.
- **Visualizations:** Interactive Plotly charts including:
  - 3D order book surface
  - Order flow imbalance heatmap
  - Prediction probability time series (with glow effects)
  - Feature importance/activation visualization
- **UI:** Cinematic dark quant theme, wide layout, large metric cards, and sidebar controls.

## How to Run
1. Clone or fork this repository:
   ```
   git clone https://github.com/tubakhxn/order-flow-prediction.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Start the Streamlit app:
   ```
   streamlit run app.py
   ```
4. Open your browser at [http://localhost:8501](http://localhost:8501)

## How to Fork
- Click the "Fork" button on the top right of the GitHub repository page.
- Clone your forked repo to your local machine:
  ```
  git clone https://github.com/YOUR-USERNAME/order-flow-prediction.git
  ```
- Make your changes and push to your fork.

## Creator
**Developer:** tubakhxn

---
For questions or contributions, please open an issue or pull request on GitHub.
