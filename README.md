# Equity Market Regime Timing

**Predicting S&P 500 drawdowns from macro, financial, and behavioral data — with a Hidden Markov Model for regime detection and gradient boosting for the timing signal.**

This is the code behind our bachelor's thesis at Universidad de Navarra (Business Administration + Data Analytics), graded 9.7/10. The question it sets out to answer is simple: can a structured, multidimensional framework anticipate periods of stress in the S&P 500 well enough to be useful for market timing? Over a demanding 2025 out-of-sample test, the answer turned out to be yes.

**Authors:** Santiago Uranga and Juan Manuel Escudero · **Supervisor:** Borja Balparda de Marco

---

## What it does

Every week, the model looks at the current macro-financial environment and predicts whether the S&P 500 is likely to suffer a significant drop over the following four weeks. When it flags risk, the strategy moves to a defensive allocation; otherwise it stays fully invested. The whole thing is built to be honest about time: nothing from the future ever leaks into a decision made in the past.

Two ideas do most of the work:

- A **Hidden Markov Model** infers the market's hidden regime (calm, volatile, or crisis) from six stress indicators, and hands the downstream model *probabilities* rather than a hard label — so "70% crisis, 30% volatile" carries more information than just "crisis."
- A **CatBoost** classifier combines those regime signals with a compact set of macro and behavioral features to produce the weekly timing call.

## Results (2025, fully out-of-sample)

The test year was deliberately hard: it included the April tariff selloff and a volatile recovery — exactly the kind of regime the model was designed to navigate. A passive buy-and-hold of the S&P 500 returned 15.26% over the period.

| Strategy (defensive allocation) | Total return | vs. benchmark | Sharpe | CAPM alpha (ann.) |
|---|---|---|---|---|
| Hold cash during risk-off | 27.98% | +12.72% | 2.01 | 32.43% (p = 0.022) |
| **Hold T-bills during risk-off** (most realistic) | **32.03%** | **+16.77%** | **1.90** | **30.66% (p = 0.034)** |
| Short the S&P 500 during risk-off | 48.39% | +33.13% | 2.61 | 49.79% (p = 0.005) |

Across every variant the strategy's beta to the market was **negative** (roughly −0.28 to −0.32), which is the key point: the outperformance didn't come from quietly taking on more market exposure, but from genuinely stepping aside during the worst of the drawdown. The CAPM alpha stays statistically significant in all three cases.

## How it's built

**Data.** Roughly two decades of weekly data (April 2005 – November 2025), pulled from FRED, Yahoo Finance, and Google Trends, and organized into eleven thematic blocks: liquidity, interest rates and the yield curve, volatility, FX, credit stress, sector rotation, valuation, commodities, real activity, inflation, and search-based behavioral signals.

**Feature engineering and selection.** Lags, multi-horizon changes, rolling statistics, and z-scores expand the raw series into ~1,250 candidate predictors. A four-stage funnel (variance filter → correlation filter → mutual-information/correlation relevance → a recursive CatBoost wrapper) narrows that down to 30 robust features. Selection runs strictly on pre-2025 data, with 2025 held out untouched.

**Regime model.** A three-state Gaussian HMM (via `hmmlearn`) fitted only on data up to 2024, using *filtered* probabilities (forward recursion, no look-ahead) so the regime signal is valid in real time. It contributes six features: the three state probabilities, regime entropy, persistence, and a hard label.

**Models compared.** Logistic Regression (Elastic Net), Balanced Random Forest, a compact neural network, XGBoost, and CatBoost — all tuned with time-series cross-validation and evaluated on balanced accuracy plus actual trading performance. CatBoost won, in part because its ordered boosting respects the time ordering of the data.

**Most influential predictors.** Financial-conditions indices (NFCI) dominate, followed by interest-rate volatility, a few Google Trends signals (searches like "sell jewelry" and "credit card debt"), and the HMM regime features.

## Repository structure

```
.
├── notebooks/
│   ├── 01_data_collection.ipynb        # pull raw series from FRED / Yahoo / Google Trends
│   ├── 02_preliminary_analysis.ipynb   # ACF/PACF, feature engineering, feature selection
│   ├── 03_hmm_regimes.ipynb            # Hidden Markov Model + regime features
│   ├── models/                         # classifier training
│   │   ├── logistic_regression.ipynb
│   │   ├── balanced_random_forest.ipynb
│   │   ├── neural_network.ipynb
│   │   ├── xgboost.ipynb
│   │   └── catboost.ipynb
│   └── trading_strategy/               # signal -> backtest for each model
│       ├── logreg_strategy.ipynb
│       ├── rf_strategy.ipynb
│       ├── nn_strategy.ipynb
│       ├── xgboost_strategy.ipynb
│       └── catboost_strategy.ipynb
├── data/
│   ├── weekly_df.csv                   # engineered weekly dataset
│   ├── reduced_vars.csv                # 30 selected features
│   ├── reduced_vars_with_hmm.csv       # + 6 HMM regime features (final modeling set)
│   └── catboost_preds.csv              # final model predictions
├── requirements.txt
└── README.md
```

## Running it

```bash
git clone https://github.com/urangasanti/equity-market-regime-timing.git
cd equity-market-regime-timing
pip install -r requirements.txt
```

Data collection needs a free FRED API key ([get one here](https://fred.stlouisfed.org/docs/api/api_key.html)). Set it as an environment variable before running the first notebook:

```bash
export FRED_API_KEY=your_key_here      # Windows: setx FRED_API_KEY "your_key_here"
```

The notebooks are meant to run in order (01 → 02 → 03 → models → strategy). If you just want to reproduce the results, the processed datasets in `data/` let you skip straight to the modeling notebooks.

## Limitations

Worth being upfront about: the out-of-sample test covers a single year, which is a short window to draw firm conclusions from, even a demanding one. The backtest assumes clean weekly execution and doesn't model transaction costs or slippage. Some inputs aren't updated at weekly frequency, so in a live setting a few signals would arrive with a lag. And, like any model trained on history, it rests on the assumption that the relationships it learned will keep holding — which isn't guaranteed.

## License

Released under the MIT License. If you reference this work, please cite:

> Uranga, S., & Escudero, J. M. (2026). *A Data-Driven Exploration of Economic, Financial, and Social Indicators to Assess Equity Market.* Bachelor's Thesis, Universidad de Navarra.
