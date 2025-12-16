"""
Common Training Functions for Density Sensor Models
===================================================

This module contains shared functionality used across all model training scripts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt


def load_training_data(depth_cm: int, 
                      data_folder: str = "Output",
                      mud_only: bool = False,
                      mud_threshold: float = 0.1) -> tuple:
    """
    Load preprocessed training data for a specific depth
    
    Args:
        depth_cm: Depth to load data for (20, 50, or 80)
        data_folder: Folder containing training data CSV files
        mud_only: If True, filter out water samples (shear <= threshold)
        mud_threshold: Threshold to separate water from mud samples
        
    Returns:
        tuple: (df, X, y, feature_columns, metadata)
            - df: Full dataframe
            - X: Feature matrix
            - y: Target vector
            - feature_columns: List of feature column names
            - metadata: Dictionary with data statistics
    """
    data_file = Path(data_folder) / f"training_data_{depth_cm}cm.csv"

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    target_column = f'shear_{depth_cm}cm'

    # Filter for mud-only if requested (for two-stage training)
    if mud_only:
        original_count = len(df)
        mud_mask = df[target_column] > mud_threshold
        df = df[mud_mask].copy()
        water_count = original_count - len(df)
        print(f"\n🔹 Training on MUD SAMPLES ONLY")
        print(f"  Filtered out {water_count} water samples (shear ≤ {mud_threshold})")
        print(f"  Training on {len(df)} mud samples (shear > {mud_threshold})")
        metadata = {'n_water_filtered': int(water_count)}
    else:
        metadata = {}

    # Identify feature columns (all frequency-related columns)
    feature_columns = [col for col in df.columns if col.startswith('freq_')]

    print(f"Found {len(feature_columns)} frequency features")

    # Extract features and target
    X = df[feature_columns].values
    y = df[target_column].values

    # Store metadata
    metadata['n_samples'] = len(df)
    metadata['n_features'] = len(feature_columns)
    metadata['target_mean'] = float(np.mean(y))
    metadata['target_std'] = float(np.std(y))
    metadata['target_min'] = float(np.min(y))
    metadata['target_max'] = float(np.max(y))

    return df, X, y, feature_columns, metadata


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate standard regression metrics
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metrics: mae, rmse, r2, mape, max_error
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE only for non-zero values
    nonzero_mask = y_true > 0.1
    if np.sum(nonzero_mask) > 0:
        mape = mean_absolute_percentage_error(y_true[nonzero_mask], y_pred[nonzero_mask])
    else:
        mape = np.nan
    
    max_error = np.max(np.abs(y_true - y_pred))
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape) if not np.isnan(mape) else None,
        'max_error': float(max_error)
    }


def print_metrics(metrics: dict):
    """
    Print regression metrics in a formatted way
    
    Args:
        metrics: Dictionary with metric values
    """
    print(f"  MAE (Mean Absolute Error):  {metrics['mae']:.2f}")
    print(f"  RMSE (Root Mean Squared):   {metrics['rmse']:.2f}")
    print(f"  R² (Coefficient of Determ): {metrics['r2']:.3f}")
    if metrics['mape'] is not None:
        print(f"  MAPE (Mean Abs % Error):    {metrics['mape']*100:.1f}%")
    print(f"  Max Error:                  {metrics['max_error']:.2f}")


def interpret_metrics(metrics: dict, target_std: float):
    """
    Provide interpretation of model performance metrics
    
    Args:
        metrics: Dictionary with metric values
        target_std: Standard deviation of target variable
    """
    print("\n" + "-"*70)
    print("Model Quality Assessment:")
    print("-"*70)
    
    r2 = metrics['r2']
    mae = metrics['mae']
    
    # R² interpretation
    if r2 > 0.9:
        print("✓ Excellent R² (>0.9) - Model explains >90% of variance")
    elif r2 > 0.8:
        print("✓ Good R² (>0.8) - Model explains >80% of variance")
    elif r2 > 0.6:
        print("⚠ Fair R² (>0.6) - Model explains >60% of variance")
    else:
        print("✗ Poor R² (<0.6) - Model explains <60% of variance")
    
    # MAE interpretation
    if mae < target_std * 0.3:
        print("✓ Excellent MAE - Predictions are very accurate")
    elif mae < target_std * 0.5:
        print("✓ Good MAE - Predictions are reasonably accurate")
    elif mae < target_std * 0.7:
        print("⚠ Fair MAE - Predictions have moderate error")
    else:
        print("⚠ High MAE - Predictions have significant error")
    
    print(f"\nAverage prediction error: ±{mae:.1f} units")


def plot_predictions(y_true: np.ndarray, 
                    y_pred: np.ndarray, 
                    title: str,
                    save_path: Path):
    """
    Create prediction vs actual and residual plots
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot: Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Shear Strength')
    axes[0].set_ylabel('Predicted Shear Strength')
    axes[0].set_title(f'{title}\nPredicted vs Actual')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot: Errors
    residuals = y_pred - y_true
    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Shear Strength')
    axes[1].set_ylabel('Residual (Predicted - Actual)')
    axes[1].set_title(f'{title}\nResidual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved prediction plot: {save_path}")
    plt.close()


def save_model_metadata(metadata: dict, 
                       model_type: str,
                       depth_cm: int,
                       feature_columns: list,
                       models_folder: Path):
    """
    Save model metadata to JSON file
    
    Args:
        metadata: Dictionary with model metadata
        model_type: Type of model (e.g., 'elasticnet', 'pls', 'gpr')
        depth_cm: Depth the model was trained for
        feature_columns: List of feature column names
        models_folder: Folder to save metadata in
    """
    metadata['depth_cm'] = depth_cm
    metadata['model_type'] = model_type
    metadata['feature_columns'] = feature_columns
    metadata['trained_at'] = datetime.now().isoformat()
    
    metadata_file = models_folder / f"{model_type}_{depth_cm}cm_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_file}")


def print_section_header(title: str, width: int = 70):
    """
    Print a formatted section header
    
    Args:
        title: Section title
        width: Width of the header
    """
    print("\n" + "="*width)
    print(title)
    print("="*width)


def print_subsection_header(title: str, width: int = 70):
    """
    Print a formatted subsection header
    
    Args:
        title: Subsection title
        width: Width of the header
    """
    print("\n" + "-"*width)
    print(title)
    print("-"*width)
