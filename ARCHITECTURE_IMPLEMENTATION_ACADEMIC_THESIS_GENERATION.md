# SYSTEM ROLE & INSTRUCTIONS
You are an expert AI Coding Agent, Principal MLOps Engineer, and Distinguished Academic Research Advisor. Your task is to execute the attached blueprint ("Applied Supervised ML Blueprint: A Monsoon-Aware, Explainable Early-Warning System for Sri Lanka") to deliver two concrete outputs:
1. A fully working, modular, production-ready ML and MLOps codebase.
2. A comprehensive, academically rigorous Master's Thesis, adhering to elite university standards, utilizing Harvard-style referencing.

---

## SECTION 1: CODEBASE ARCHITECTURE & IMPLEMENTATION

Generate clean, production-grade, PEP-8 compliant Python code. Do not use placeholders, `pass` statements, or truncated logic. Structure the repository as follows:

```text
├── .github/workflows/    # CI/CD pipelines (GitHub Actions)
├── config/               # Configuration files (YAML/Hydra)
├── data/                 # Data directory (managed via DVC)
├── models/               # Serialized model artifacts
├── src/                  # Core source code
│   ├── __init__.py
│   ├── data_prep.py      # Ingestion, validation, and cleaning
│   ├── features.py       # Advanced feature engineering & selection
│   ├── train.py          # Training, Optuna tuning, Cross-Validation
│   ├── evaluate.py       # Metrics, error analysis, fairness audits
│   └── explain.py        # SHAP and LIME global/local explainability
├── app/                  # FastAPI Deployment
│   ├── __init__.py
│   ├── main.py           # REST API endpoints (Real-time & Batch)
│   └── schemas.py        # Pydantic data models
├── Dockerfile            # Multi-stage production container build
├── dvc.yaml              # DVC pipeline orchestration
├── mlflow_setup.sh       # MLflow tracking infrastructure script
└── requirements.txt      # Production-pinned dependencies
```

### Technical Specifications to Code:
* **Feature Engineering (`src/features.py`)**: Implement all 6 novel features from the blueprint, including cyclical time encodings, wind vectors, and the standardized rolling climatic anomaly:
  $$\text{precip\_anom\_7d} = \frac{\text{precip}_t - \mu_{7d}}{\sigma_{7d}}$$
  Implement the exact NWS Heat Index regression proxy derived from apparent temperature spreads.
* **Algorithmic Architecture (`src/train.py`)**: Build training pipelines for all 5 model families (Linear/ElasticNet, KNN, Random Forest, LightGBM/XGBoost, and CatBoost as the primary frontrunner). Include a deep learning PyTorch/TensorFlow LSTM pipeline for sequence modeling over a 7-day historical lag window.
* **Optimization & CV**: Use Optuna with a TPE sampler and ASHA pruning. Enforce a grouped, blocked, expanding-window `TimeSeriesSplit` (5 folds) across the 30-city panel to prevent spatial and temporal data leakage.
* **Deployment & MLOps (`app/main.py`, `Dockerfile`, CI/CD)**: Write an asynchronous FastAPI app serving a real-time `/predict` endpoint and a nightly bulk batch processing pipeline. Build a multi-stage `Dockerfile` optimizing for size. Write a GitHub Actions YAML script that handles linting, testing, Docker image creation, and automated deployment configurations. Integrate data/prediction drift tracking using **Evidently** (KS-test and Chi-square checks).

---

## SECTION 2: ACADEMIC MASTER'S THESIS GENERATION

Write a comprehensive, publication-grade Master's Thesis detailing this system. The thesis must be written with strict academic gravity, using formal prose and Harvard-style referencing (Author, Year) integrated seamlessly into the text, paired with a complete, alphabetized Reference list at the end. 

Incorporate the real-world contextual data provided in the blueprint (e.g., the humanitarian impacts of Cyclone Ditwah in late 2025, the UNDP geospatial data, and the World Bank GRADE physical damage reports) to ground the study's practical urgency in the current temporal setting of 2026.

Expand the thesis into the following structural chapters:

### Chapter 1: Introduction & Contextual Urgency
* **Background**: The tropical convective meteorological profile of Sri Lanka, its dependency on the Yala (Southwest) and Maha (Northeast) monsoons, and the shifting dynamics due to global climate change.
* **Problem Statement**: The failure of traditional, coarse global forecasting grids to capture highly localized microclimate anomalies across Sri Lanka's 30 distinct urban/agricultural nodes.
* **The 2025 Crisis Baseline**: Analyze the systemic failures exposed by Cyclone Ditwah (November 2025), citing the UNDP finding that 2.3 million people were left in flooded regions and 20% of the country's land area ($1.1\text{ million hectares}$) was inundated. Incorporate the World Bank GRADE report data indicating $\$4.1\text{ billion}$ in direct physical damage ($\approx 4\%$ of Sri Lanka's GDP), highlighting the $\$814\text{ million}$ blow to precision agriculture.
* **Research Objectives**: Detail the implementation of a dual-headed (Classification + Regression) machine learning warning system optimized for precision agriculture and disaster management.

### Chapter 2: Literature Review & Methodological Gaps
Integrate a deep critical review of the state-of-the-art literature provided in the blueprint, explaining what each author achieved and how this thesis fills their unresolved gaps:
1. **Saubhagya et al. (2021/2023)**: Pioneer ML for Sri Lankan agriculture via SVM/XGBoost, but restricted by coarse temporal horizons and lack of productionization.
2. **Hennayake et al. (2021)**: Establish deep learning (LSTM) viability for tropical rainfall, but suffer from single-station limitation (Colombo only), failing to generalize across diverse agro-climatic zones.
3. **South Asian Deep Learning Benchmarks (2025)**: Document low-frequency monsoon variability and ENSO teleconnections, but lack actionable, field-level early warning deployment architectures.
4. **Spatiotemporal ConvLSTM & XAI Architectures (2025)**: Demonstrate the vital role of Explainable AI in convective nowcasting, which this thesis adapts to tabular multi-city panels.
5. **Tree-Based SHAP Health Frameworks (2024/2025)**: Establish templates for mapping meteorological features to downstream risks (e.g., heat stress index mapping and Dengue vector outbreaks via environmental bands like $25\text{--}28^\circ\text{C}$ temperature ranges).
6. **Imbalance-Handling Paradigms (2024)**: Document the mathematical necessity of SMOTE/SMOTEN and class-weight balancing for zero-inflated, highly asymmetric tropical precipitation distributions.
* **Synthesized Gaps**: Explicitly articulate the four foundational gaps addressed by this research: Spatial scale limitations, complete lack of model explainability (XAI) in production early-warning systems, severe vulnerability to class-imbalance failure during extreme events, and the complete absence of continuous MLOps lifecycle monitoring in tropical meteorology.

### Chapter 3: Data Provenance & Advanced Feature Engineering
* **Data Profiling**: Comprehensive analysis of the 30-city daily dataset (2010–2023), confirming validation against all Phase-0 hard constraints.
* **Preprocessing & Data Quality**: Elaborate on the interpolation strategy for missing radiation/evapotranspiration metrics and the handling of extreme, heavy-tail convective anomalies as valid high-value signal rather than sensor noise.
* **Mathematical Feature Formulation**: Detail the physics and logic behind the engineered features. Write out the exact equations for the rolling standardized climatic anomalies, the cyclical spatial/temporal coordinate transformations, and the wind vector components ($wind\_u$ and $wind\_v$) designed to capture asymmetric monsoon flows.

### Chapter 4: Algorithmic Architecture & Optimization Protocol
* **Model Selection Framework**: Compare the mathematical behaviors of the selected models when processing high-cardinality panel data. Discuss why Gradient Boosted Decision Trees (specifically CatBoost) hold structural advantages over deep architectures like LSTMs on highly heterogeneous tabular meteorological records.
* **Validation Rigor**: Detail the mathematical framework of grouped, blocked `TimeSeriesSplit` cross-validation. Contrast this with standard randomized K-Fold validation, proving how random shuffling introduces severe data leakage and false optimization metrics in time-series frameworks.
* **Hyperparameter Optimization**: Detail the Bayesian TPE (Tree-structured Parzen Estimator) search space orchestrated via Optuna, explaining the objective functions designed to balance Precision-Recall AUC (PR-AUC) against continuous Root Mean Squared Error (RMSE).

### Chapter 5: Empirical Evaluation, Error Analysis & Refinement
* **Comparative Performance**: Present a comprehensive comparison matrix tracking all models across classification metrics (PR-AUC, F1-Score, Sensitivity/Recall) and regression metrics (RMSE, MAE, $R^2$).
* **Statistical Validation**: Describe the application of the Diebold-Mariano and Wilcoxon signed-rank tests to prove the statistical significance of CatBoost’s performance over the baselines.
* **Spatial & Seasonal Error Decomposition**: Conduct a deep dive into the system's performance across distinct zones (Wet Zone vs. Dry Zone vs. Hill Country Interior) and monsoon seasons. Analyze the behavior of false negatives during extreme climate spikes.
* **Optimization via Threshold Tuning**: Document how adjusting decision boundaries via Youden’s J statistic and F2-score optimization shields agricultural and humanitarian end-users from catastrophic missed alerts.

### Chapter 6: Explainable AI (XAI) & Algorithmic Ethics Audit
* **Global and Local Interpretability**: Utilize SHAP TreeExplainer frameworks to map out the overall global feature hierarchies driving prediction networks. Show how individual local force plots provide absolute operational transparency for localized civil defense actors.
* **The Geographic Equity Audit**: Address the severe ethical realities exposed during the 2025 climate disasters (e.g., the WHO situation update regarding high fatalities in vulnerable hill-country communities like Kandy, Badulla, and Nuwara Eliya due to 1,200 concurrent landslides). Detail how the model avoids spatial discrimination and ensures high sensitivity in geographically marginalized regions.
* **Automation and Automation Bias**: Provide a socio-technical analysis of the human-in-the-loop validation paradigm required to balance algorithmic precision with the institutional accountability of the Department of Meteorology.

### Chapter 7: Production MLOps Infrastructure Design
* **System Architecture**: Detail the end-to-end cloud infrastructure topology (Ingestion $\rightarrow$ Versioning $\rightarrow$ Training $\rightarrow$ Containerized Serving $\rightarrow$ Active Monitoring).
* **Continuous Performance & Retraining Loops**: Detail the integration of **Evidently** to monitor data distribution shifts and concept drift. Detail the precise operational boundaries where mathematical drift triggers automated pipeline retraining to guarantee system robustness in a non-stationary, changing climate.

### Chapter 8: Conclusion & Future Outlook
* **Synthesized Contributions**: Recapitulate the structural milestones accomplished across the technical and theoretical domains.
* **Future Horizons**: Discuss extending the system via multimodal satellite imagery fusion and edge-computing deployment for disconnected rural communities.

### References
* Provide a comprehensive, clean, Harvard-compliant bibliography listing all cited research, climate reports (UNDP, WHO, World Bank), and dataset sources.

---

## EXECUTION STEP-BY-STEP
Begin execution immediately. First, output the comprehensive folder scaffolding and complete code modules for `src/data_prep.py` and `src/features.py`. Once the foundational code blocks are constructed, systematically proceed through the generation of the training infrastructure and the complete text of the Master's Thesis.