# Deep Learning Module 5 Final Project (DL_MOD5_FINAL)

**Course: Deep Learning**

This repository contains the final project for the Deep Learning course (Module 5). The project is a comprehensive Jupyter Notebook that implements a deep learning pipeline to forecast stock prices.

## Project Objective

The goal of this project is to build, compare, and optimize several Recurrent Neural Network (RNN) models to predict the next day's closing price of Apple Inc. (AAPL) stock.

The primary finding of the project was that a simple **Naive Baseline** model (RMSE: \$2.09) outperformed all deep learning models, demonstrating that historical price data alone was not a sufficient predictor for this task.

## Methodology

The notebook is structured to follow a complete data science workflow:

1.  **Data Collection**: 5 years of `AAPL` stock data (2019-2024) are downloaded using the `yfinance` library.
2.  **Exploratory Data Analysis (EDA)**: The data is visualized to identify trends, correlations (notably the multicollinearity of O-H-L-C prices), and distributions (like the right-skew of `Volume`).
3.  **Data Preprocessing**: Data is cleaned, normalized using `MinMaxScaler` (fit *only* on training data to prevent data leakage), and transformed into 60-day sequences.
4.  **Model Comparison**: Three different architectures are built and compared against a "Naive Baseline" (persistence) model:
    * Naive Baseline (Benchmark)
    * Baseline Simple RNN
    * Baseline LSTM (Long Short-Term Memory)
    * Baseline GRU (Gated Recurrent Unit)
5.  **Hyperparameter Tuning**: The `Keras Tuner` library is used to find the optimal hyperparameters for the GRU model, using a **chronological validation split** to prevent data leakage during tuning.
6.  **Robust Evaluation**: A final evaluation is performed using a 5-Fold **Time-Series Cross-Validation** to get a reliable measure of the tuned model's true performance.

## Results & Key Finding

The primary finding of this project was that the deep learning models, when trained on price history alone, **failed to outperform the Naive Baseline**. The "common-sense" persistence model (predicting tomorrow's price is the same as today's) was the most accurate.

This highlights a classic "Efficient Market Hypothesis" scenario, where past price movement alone does not contain enough information to accurately predict future movements.

#### Final RMSE Scores:

| Model | RMSE (Lower is better) | Analysis |
| :--- | :--- | :--- |
| **Naive Baseline (Winner)** | **\$2.09** | The most accurate model. |
| **Tuned GRU (5-Fold CV Avg)** | \$3.03 | The most robust DL model, but still inferior to the baseline. |
| **Baseline Simple RNN** | \$3.10 | Surprisingly outperformed the single-split tuned GRU. |
| **Final Tuned GRU (Single Split)** | \$3.85 | The 80/20 test split gave a pessimistic score. |
| **Baseline LSTM** | \$7.13 | Performed the worst, likely due to poor initial hyperparameters. |

The final plots in the notebook clearly visualize this finding, showing the Naive Baseline's predictions tracking the *actual* price more closely than the trained GRU model.

## How to Run

This project is contained in a single Jupyter Notebook (`.ipynb`).

1.  **Recommended Environment**
    The easiest way to run this project is on **Google Colab**, which provides a free GPU for training and has all dependencies pre-installed.
2.  **Dependencies**
    If running locally, you must install the following libraries:
    ```
    pip install tensorflow yfinance keras-tuner pandas matplotlib seaborn scikit-learn
    ```
