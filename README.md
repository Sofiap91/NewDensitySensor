# New Density Sensor - Python ML Pipeline

Machine learning pipeline for predicting shear strength from Vector Network Analyzer (VNA) frequency measurements using a two-stage classification + regression approach.

## Overview

This project uses electromagnetic frequency data (400MHz-4GHz) from VNA measurements to predict mud shear strength at different depths (20cm, 50cm, 80cm). The two-stage model:

1. **Stage 1 (Classifier):** RandomForest binary classifier separates water (shear=0) from mud (shear>0)
2. **Stage 2 (Regressor):** Chosen regression model predicts shear strength for mud samples only

## Quick Start

### 1. Train Models

```bash
# Train regression models for all depths
python3 models/train_elasticnet.py  # ElasticNet (linear with L1+L2 regularization)
python3 models/train_pls.py         # Partial Least Squares (best for spectral data)
python3 models/train_gpr.py         # Gaussian Process Regression (with uncertainty)

# Train two-stage classifiers for all depths
python3 models/train_mud_classifier.py
```

### 2. Make Predictions

```bash
# Use existing preprocessed data with specified model
python3 main.py --depth 20 --model elasticnet
python3 main.py --depth 50 --model pls
python3 main.py --depth 80 --model gpr

```

### 3. Visualize Results

```bash
# Create GPS heatmaps from output prediction files
python3 visualize_predictions.py predictions_20cm_elasticnet_20251216_143022.csv
python3 visualize_predictions.py predictions_50cm_pls_20251216_143055.csv
```

## Available Models

- **elasticnet** - Linear regression with L1+L2 regularization (default)
- **pls** - Partial Least Squares (recommended for spectral/frequency data)
- **gpr** - Gaussian Process Regression (provides uncertainty estimates)

## File Structure

```
Input/                          # Raw data (VNA + Vane Shear)
Output/                         # Preprocessed training data
models/
  ├── elasticnet/              # ElasticNet models
  ├── pls/                     # PLS models
  ├── gpr/                     # GPR models
  ├── mud_classifier/          # Binary classifiers
  └── train_*.py               # Training scripts
main.py                        # Prediction pipeline
scripts/
  ├── visualize_predictions.py       # GPS heatmap visualization
data_processor.py              # Data preprocessing
```

## Data

- **Input:** Raw data for preprocessing. Includes:
  - VNA frequency measurements (724 features, 400MHz-4GHz) + GPS coordinates
  - Manual Vane shear test results with GPS location
- **Output:**
  - Data after preprocessing for training the models. Include a file per depth that removed Null values and a complete data file