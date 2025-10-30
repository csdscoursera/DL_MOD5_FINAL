# Deep Learning Module 5 Final Project (DL_MOD5_FINAL)
**Course:** Deep Learning

This repository contains the final project for the Deep Learning course (Module 5). The project is a comprehensive Jupyter Notebook that implements a deep learning pipeline to forecast stock prices.

---

## Project Objective
The goal of this project is to build, compare, and optimize several Recurrent Neural Network (RNN) models to predict the next day's closing price of Apple Inc. (`AAPL`) stock.

---

## Methodology
The notebook is structured to follow a complete data science workflow:

1.  **Data Collection**: 5 years of `AAPL` stock data (2019-2024) are downloaded using the `yfinance` library.
2.  **Exploratory Data Analysis (EDA)**: The data is visualized to identify trends, correlations, and distributions.
3.  **Data Preprocessing**: Data is cleaned, normalized using `MinMaxScaler`, and transformed into 60-day sequences for time-series forecasting.
4.  **Model Comparison**: Three different architectures are built and compared:
    * Baseline **Simple RNN**
    * **LSTM** (Long Short-Term Memory)
    * **GRU** (Gated Recurrent Unit)
5.  **Hyperparameter Tuning**: The `Keras Tuner` library is used to find the optimal hyperparameters (units, dropout rate, learning rate) for the best-performing model (GRU).

---

## Results
After a comparative analysis, the **Final Tuned GRU** model was the clear winner, achieving the best performance on the unseen test set.

* **Final Model RMSE:** **$3.55**
* **Baseline LSTM RMSE:** $4.19
* **Baseline RNN RMSE:** $6.95

The final model's predictions are plotted against the actual test data, showing it successfully learned and tracked the stock's price momentum.

---

## How to Run
This project is contained in a single Jupyter Notebook (`.ipynb`).

### 1. Recommended Environment
The easiest way to run this project is on **Google Colab**, which provides a free GPU for training.

### 2. Dependencies
If running locally, you must install the following libraries:

```bash
pip install tensorflow yfinance keras-tuner pandas matplotlib seaborn scikit-learn
