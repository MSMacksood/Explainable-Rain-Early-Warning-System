"""Monsoon-aware, explainable rainfall early-warning system for Sri Lanka.

Modules
-------
data_prep : ingestion, Phase-0 validation, cleaning
features  : monsoon-aware feature engineering and selection
train     : 5 model families, PanelTimeSeriesSplit CV, Optuna tuning
evaluate  : metrics matrix, statistical tests, fairness/error decomposition
explain   : SHAP / permutation-importance explainability artifacts
"""

__version__ = "1.0.0"
