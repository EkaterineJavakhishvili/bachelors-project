Financial Market Prediction using Ensemble Learning & NLP

📌 Abstract
This project serves as the practical implementation for a Bachelor's Thesis on "Multi-Source Ensemble Learning for Financial Market Movement Prediction"

The system constructs an AI-powered Trading Agent that fuses two distinct data streams for a target asset:

    - Technical Analysis: Internal market momentum, RSI, MACD, and lag features derived directly from the asset's price history.

    - Natural Language Processing (NLP): Sentiment scores derived from financial news headlines using FinBERT (Financial Bidirectional Encoder Representations from Transformers).

Methodologically, the project compares individual regression models (Linear, Random Forest, Gradient Boosting) against a weighted Ensemble Strategy to minimize variance and improve predictive stability.

📂 Project Structure

bachelors-project/
├── data/
│   ├── raw/                 # Raw downloads (CSV)
│   ├── processed/           # Cleaned data ready for modeling
├── models/
│   └── price_agent/         # Serialized models (.pkl)
├── reports/
│   └── figures/             # Generated charts (Backtests, RMSE plots)
├── src/
│   ├── data/                # ETL Pipeline (Ingest & Prepare)
│   ├── models/              # Training scripts (RF, GB, Linear, Ensemble)
│   ├── news/                # NLP & Sentiment Analysis (FinBERT)
│   ├── prices/              # Feature Engineering & Backtesting
│   └── utils.py             # Helper functions
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

⚙️ Installation
1. Clone the Repository
git clone https://github.com/EkaterineJavakhishvili/bachelors-project.git
cd bachelors-project

2. Set Up Virtual Environment (Recommended)
It is best practice to run scientific code in an isolated environment to avoid version conflicts.

# MacOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
This project relies on torch (for BERT), scikit-learn, pandas, and yfinance.

pip install -r requirements.txt

🚀 Usage Pipeline (The Runbook)
To replicate the thesis results, execute the modules in the following strict order.

Phase 1: Data Acquisition (ETL)
Download raw market data for the target asset (e.g., AAPL).

# 1. Download Raw Data
python -m src.data.ingest --ticker AAPL
Financial News Data: https://www.kaggle.com/datasets/frankossai/apple-stock-aapl-historical-financial-news-data

# 2. Clean & Convert to Optimized Parquet
python -m src.data.prepare --ticker AAPL

Phase 2: Natural Language Processing
Process the financial news dataset using the FinBERT transformer model to generate a daily sentiment score (-1 to +1). Note: This process requires significant computational resources.

python -m src.news.sentiment --ticker AAPL

Phase 3: Feature Engineering
Merge the Price Data, Technical Indicators (RSI, MACD, Lags), and Sentiment Scores into a single "Golden Dataset."

python -m src.prices.features --ticker AAPL
    Output: data/processed/features/AAPL_features.parquet

Phase 4: Model Training
Train the three component models of the ensemble using the full historical dataset.

# 1. Linear Regression
python -m src.models.train_linear --ticker AAPL

# 2. Gradient Boosting
python -m src.models.train_gb --ticker AAPL

# 3. Random Forest
python -m src.models.train_rf --ticker AAPL

    Artifacts: Models are saved to models/price_agent/*.pkl.

Phase 5: Evaluation & Simulation
A. Comparative Study (Ensemble Leaderboard)
Evaluate the models on unseen test data. This script generates the RMSE/R² metrics and a plot comparing the Ensemble Prediction vs. Actual Price.

python -m src.models.train_ensemble --ticker AAPL
    Figure Saved: reports/figures/ensemble_results_AAPL.png

B. Trading Backtest (Profitability Check)
Simulate a trading strategy using the trained AI agent over the full simulation period.

    - Strategy: Go Long if AI predicts Price > Today, else Hold Cash.
    - Benchmark: Buy & Hold.

# Run simulation on full dataset
python -m src.prices.backtest --ticker AAPL
    Figure Saved: reports/figures/backtest_result.png

📊 Results Summary
The final system was evaluated on Apple Inc. (AAPL) stock data.

Metric	Random Forest	Gradient Boosting	Linear Baseline	Ensemble
RMSE (Price)	$3.61	$3.80	$3.67	$3.63
R² (Price)	0.9848	0.9831	0.9843	0.9846

    - Prediction Accuracy: The Random Forest model achieved the highest individual accuracy (R^2 of 0.9848), effectively capturing the asset's momentum and trend.

    - Trading Performance: In the backtest simulation, the AI strategy generated a return of +43.85%, successfully identifying the primary bull trend while managing downside risk during market corrections.

📜 License
This project is open-source and available under the MIT License.

👨‍💻 Author
Ekaterine Javakhishvili Bachelor's Degree Candidate | Caucasus Univesity Specialization in Computer Science