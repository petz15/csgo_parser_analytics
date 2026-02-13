# CS:GO Economic Analytics

Research project analyzing Counter-Strike: Global Offensive economic decision-making using statistical modeling, Agent-Based Models (ABM), and machine learning.

## Project Overview

This repository contains analysis and modeling tools for CS:GO round-by-round economic strategies, exploring how teams optimize equipment spending decisions based on game state information.

### Components

1. **Statistical Analysis** - Contest Success Function (CSF) models analyzing player performance
   - Symmetric and asymmetric Tullock CSF models
   - Maximum Likelihood Estimation (MLE)
   - Logistic regression models with ROC curves and goodness-of-fit metrics
   - See: [non_parametric_methods.ipynb](non_parametric_methods.ipynb)

2. **Descriptive Statistics** - Creating visualizations for the Master Thesis
   - CSV exports for reproducibility
   - See: [descriptive_statistics_v2.ipynb](descriptive_statistics_v2.ipynb)

3. **Agent-Based Modeling** - Creating ABM distributions, parametrization, initialization etc.
   - See: [abm_distributions.ipynb](abm_distributions.ipynb)

4. **Machine Learning** - Reinforcement learning agents trained on ABM data
   - DQN, PPO, REINFORCE, Tree, XGBoost, and Logistic strategies
   - Imitation learning from simulated game outcomes
   - Export to Go-compatible formats
   - See: [ML_TRAINING_README.md](ML_TRAINING_README.md)

## Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL database (for descriptive statistics) or CSV files
- Required packages: pandas, numpy, scipy, sklearn, torch, xgboost, matplotlib, seaborn, openpyxl, python-dotenv

### Installation
```bash
pip install pandas numpy scipy scikit-learn torch xgboost matplotlib seaborn openpyxl psycopg2-binary python-dotenv
```

### Data

- The three notebooks (descriptive_statistics_v2, abm_distributions and non_parametric_methods) use the same DB query as basis for their calculations. Although in the code, each exports its own CSV, they are the same. 
- For ML training, use ABM simulations with configuration --csv 4. Multiple files can be added to the folder, make sure they have distinct names

### Basic Usage

**Run Statistical Analysis:**
```bash
jupyter notebook non_parametric_methods.ipynb
```

**Export Descriptive Statistics:**
```bash
jupyter notebook descriptive_statistics_v2.ipynb
# Exports to descriptive_statistics_YYYYMMDD.csv
```

**Train ML Models:**
```bash
python train_from_abm_data.py --results-folder matchup_021_anti_allin_v3_vs_expected_value --strategy dqn --episodes 500
```

## Project Structure

```
├── non_parametric_methods.ipynb      # CSF statistical analysis
├── descriptive_statistics_v2.ipynb   # Database queries & exports
├── ml_model.py                        # RL strategy implementations
├── train_from_abm_data.py            # ML training script
├── export_models.py                   # Model export to Go format
├── export_for_go.py                   # Additional Go exports
├── data_exports/                      # CSV/Excel outputs
├── models_from_abm/                   # Trained model files (.pt)
├── ml_go_models/                      # Go-compatible model exports
└── matchup_*/                         # ABM simulation results
```

## Output Files

- **Statistical Tables**: Excel/CSV format with multiple sheets (model comparisons, diagnostics, robustness checks)
- **Trained Models**: PyTorch `.pt` files with training metadata in `models_from_abm/`
- **Go Exports**: JSON files with model weights and architecture in `ml_go_models/`

## Key Features

- ✅ Publication-ready statistical tables with significance indicators (p < 0.005)
- ✅ Comprehensive model diagnostics (ROC-AUC, Brier Score, Hosmer-Lemeshow)
- ✅ Reproducible analysis via CSV data exports
- ✅ Multiple RL algorithms for economic strategy learning
- ✅ Cross-platform model deployment (Python → Go)

## Documentation

- [ML_TRAINING_README.md](ML_TRAINING_README.md) - Detailed machine learning training guide
- Inline documentation in Jupyter notebooks
- Docstrings in Python modules

## License

Academic research project.
