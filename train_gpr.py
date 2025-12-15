"""
Gaussian Process Regression (GPR) for Density Sensor
=====================================================

Gaussian Process Regression is excellent for:
- Small datasets (our case: ~100-130 samples)
- Non-linear relationships
- Uncertainty quantification (provides confidence intervals)
- Capturing complex patterns without overfitting

GPR models the target as a distribution rather than a single prediction,
giving us both a prediction AND confidence in that prediction - very useful
for engineering applications where uncertainty matters.

Note: GPR is computationally expensive (O(n³)), so it works best with
smaller datasets like ours.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error
)

import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class GPRTrainer:
    """Trains Gaussian Process Regression models for shear strength prediction"""
    
    def __init__(self, depth_cm: int, data_folder: str = "Output", models_folder: str = "models/gpr"):
        """
        Initialize the trainer
        
        Args:
            depth_cm: Depth to train for (20, 50, or 80)
            data_folder: Folder containing preprocessed CSV files
            models_folder: Folder to save trained models
        """
        self.depth_cm = depth_cm
        self.data_folder = Path(data_folder)
        self.models_folder = Path(models_folder)
        self.models_folder.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_columns = None
        self.target_column = f'shear_{depth_cm}cm'
        self.metadata = {}
        
    def load_data(self) -> tuple:
        """
        Load preprocessed data for the specified depth
        
        Returns:
            X: Feature matrix
            y: Target vector
        """
        data_file = self.data_folder / f"training_data_{self.depth_cm}cm.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        
        print(f"Loaded {len(df)} samples")
        
        # Identify feature columns (all frequency-related columns)
        self.feature_columns = [col for col in df.columns if col.startswith('freq_')]
        
        print(f"Found {len(self.feature_columns)} frequency features")
        
        # Extract features and target
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        
        # Store metadata
        self.metadata['n_samples'] = len(df)
        self.metadata['n_features'] = len(self.feature_columns)
        self.metadata['target_mean'] = float(np.mean(y))
        self.metadata['target_std'] = float(np.std(y))
        self.metadata['target_min'] = float(np.min(y))
        self.metadata['target_max'] = float(np.max(y))
        
        print(f"\nTarget statistics ({self.target_column}):")
        print(f"  Mean: {self.metadata['target_mean']:.2f}")
        print(f"  Std:  {self.metadata['target_std']:.2f}")
        print(f"  Min:  {self.metadata['target_min']:.2f}")
        print(f"  Max:  {self.metadata['target_max']:.2f}")
        
        return X, y
    
    def create_pipeline(self, n_components: int = 20) -> Pipeline:
        """
        Create a scikit-learn pipeline with preprocessing and GPR
        
        Pipeline steps:
        1. StandardScaler: Normalize features (mean=0, std=1)
        2. PCA: Reduce dimensions (GPR is expensive with many features)
        3. GaussianProcessRegressor: Non-linear regression with uncertainty
        
        Args:
            n_components: Number of PCA components (must be < n_samples)
            
        Returns:
            Pipeline object
        """
        # Define the kernel: combination of RBF (smooth variations) and WhiteKernel (noise)
        # This is a good general-purpose kernel for sensor data
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
                 WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        
        steps = [
            ('scaler', StandardScaler()),  # Normalize features
            ('pca', PCA(n_components=n_components)),  # Reduce dimensions for speed
            ('gpr', GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,  # Try multiple initializations
                alpha=1e-10,  # Small regularization
                normalize_y=True  # Normalize target for better numerics
            ))
        ]
        
        return Pipeline(steps)
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2,
        n_pca_components: int = 15
    ):
        """
        Train the GPR model
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data to use for testing (0-1)
            n_pca_components: Number of PCA components (keep low for GPR speed)
        """
        print("\n" + "="*70)
        print(f"Training Gaussian Process Regression Model for {self.depth_cm}cm Depth")
        print("="*70)
        print("\nNote: GPR is computationally intensive. Using PCA to reduce dimensions.")
        print(f"Reducing {X.shape[1]} features to {n_pca_components} components for speed.")
        
        # Split data into training and test sets
        print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Ensure n_components is less than n_samples
        max_components = min(n_pca_components, X_train.shape[0] - 1, X_train.shape[1])
        
        print(f"\nTraining GPR with {max_components} PCA components...")
        print("This may take a few minutes...")
        
        # Create and train pipeline
        pipeline = self.create_pipeline(n_components=max_components)
        pipeline.fit(X_train, y_train)
        
        self.model = pipeline
        
        # Report on dimensionality reduction
        pca = self.model.named_steps['pca']
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"\nPCA reduced features from {X.shape[1]} to {max_components}")
        print(f"Explained variance: {explained_variance:.2%}")
        
        self.metadata['pca_components'] = int(max_components)
        self.metadata['pca_variance_explained'] = float(explained_variance)
        
        # Get the trained kernel
        gpr = self.model.named_steps['gpr']
        print(f"\nOptimized kernel: {gpr.kernel_}")
        
        # Evaluate on training and test sets
        print("\n" + "="*70)
        print("Model Evaluation")
        print("="*70)
        
        # Make predictions (GPR returns mean and std)
        y_train_pred, y_train_std = self.model.predict(X_train, return_std=True)
        y_test_pred, y_test_std = self.model.predict(X_test, return_std=True)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)
        
        # Add uncertainty information
        train_metrics['mean_uncertainty'] = float(np.mean(y_train_std))
        test_metrics['mean_uncertainty'] = float(np.mean(y_test_std))
        
        # Display results
        print("\nTraining Set Performance:")
        self._print_metrics(train_metrics)
        
        print("\nTest Set Performance:")
        self._print_metrics(test_metrics)
        
        # Store metrics
        self.metadata['train_metrics'] = train_metrics
        self.metadata['test_metrics'] = test_metrics
        
        # Interpret results
        self._interpret_results(test_metrics)
        
        # Create visualizations
        self._plot_predictions(y_test, y_test_pred, y_test_std, f"{self.depth_cm}cm Test Set")
        
        return self.model
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate regression metrics"""
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
    
    def _print_metrics(self, metrics: dict):
        """Print metrics in a nice format"""
        print(f"  MAE (Mean Absolute Error):  {metrics['mae']:.2f}")
        print(f"  RMSE (Root Mean Squared):   {metrics['rmse']:.2f}")
        print(f"  R² (Coefficient of Determ): {metrics['r2']:.3f}")
        if metrics['mape'] is not None:
            print(f"  MAPE (Mean Abs % Error):    {metrics['mape']*100:.1f}%")
        print(f"  Max Error:                  {metrics['max_error']:.2f}")
        if 'mean_uncertainty' in metrics:
            print(f"  Mean Uncertainty (±σ):      {metrics['mean_uncertainty']:.2f}")
    
    def _interpret_results(self, metrics: dict):
        """Provide interpretation of the results"""
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
        target_std = self.metadata['target_std']
        if mae < target_std * 0.3:
            print("✓ Excellent MAE - Predictions are very accurate")
        elif mae < target_std * 0.5:
            print("✓ Good MAE - Predictions are reasonably accurate")
        elif mae < target_std * 0.7:
            print("⚠ Fair MAE - Predictions have moderate error")
        else:
            print("⚠ High MAE - Predictions have significant error")
        
        print(f"\nAverage prediction error: ±{mae:.1f} units")
        
        if 'mean_uncertainty' in metrics:
            print(f"Average uncertainty estimate: ±{metrics['mean_uncertainty']:.1f} units")
            print("\n💡 GPR provides uncertainty estimates - useful for engineering decisions!")
    
    def _plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, title: str):
        """Create visualization with uncertainty bands"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Predicted vs Actual with error bars
        axes[0].errorbar(y_true, y_pred, yerr=y_std, fmt='o', alpha=0.6, 
                        elinewidth=1, capsize=3, markersize=5)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Shear Strength')
        axes[0].set_ylabel('Predicted Shear Strength')
        axes[0].set_title(f'{title}\nPredicted vs Actual (GPR with ±σ)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Residual plot
        residuals = y_pred - y_true
        axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Shear Strength')
        axes[1].set_ylabel('Residual (Predicted - Actual)')
        axes[1].set_title(f'{title}\nResidual Plot')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Uncertainty vs Error
        abs_errors = np.abs(residuals)
        axes[2].scatter(y_std, abs_errors, alpha=0.6, edgecolors='k', linewidths=0.5)
        axes[2].plot([0, y_std.max()], [0, y_std.max()], 'r--', lw=2, 
                     label='Perfect Calibration')
        axes[2].set_xlabel('Predicted Uncertainty (σ)')
        axes[2].set_ylabel('Absolute Error')
        axes[2].set_title(f'{title}\nUncertainty Calibration')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.models_folder / f"gpr_{self.depth_cm}cm_predictions.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved prediction plot: {plot_file}")
        plt.close()
    
    def save_model(self):
        """Save the trained model and metadata"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Save the model
        model_file = self.models_folder / f"gpr_{self.depth_cm}cm.pkl"
        joblib.dump(self.model, model_file)
        print(f"\nSaved model: {model_file}")
        
        # Save metadata
        self.metadata['depth_cm'] = self.depth_cm
        self.metadata['model_type'] = 'Gaussian_Process_Regression'
        self.metadata['feature_columns'] = self.feature_columns
        self.metadata['trained_at'] = datetime.now().isoformat()
        
        metadata_file = self.models_folder / f"gpr_{self.depth_cm}cm_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved metadata: {metadata_file}")
    
    def load_model(self):
        """Load a previously trained model"""
        model_file = self.models_folder / f"gpr_{self.depth_cm}cm.pkl"
        metadata_file = self.models_folder / f"gpr_{self.depth_cm}cm_metadata.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        self.model = joblib.load(model_file)
        print(f"Loaded model: {model_file}")
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            self.feature_columns = self.metadata['feature_columns']
            print(f"Loaded metadata: {metadata_file}")
    
    def predict(self, X: np.ndarray, return_std: bool = False):
        """
        Make predictions using the trained model
        
        Args:
            X: Feature matrix
            return_std: If True, return uncertainty estimates
            
        Returns:
            predictions (and optionally standard deviations)
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        if return_std:
            return self.model.predict(X, return_std=True)
        else:
            return self.model.predict(X)


def main():
    """Main training function"""
    
    print("="*70)
    print("Gaussian Process Regression Model Training for Density Sensor")
    print("="*70)
    print("\nGPR is excellent for small datasets and provides uncertainty estimates.")
    print("Training will be slower than linear models due to computational complexity.\n")
    
    # Configuration
    DEPTHS = [20, 50, 80]  # Train models for all depths
    DATA_FOLDER = "Output"
    MODELS_FOLDER = "models/gpr"
    N_PCA_COMPONENTS = 15  # Keep low for GPR speed (n³ complexity)
    
    # Train a model for each depth
    for depth in DEPTHS:
        print(f"\n{'='*70}")
        print(f"TRAINING MODEL FOR {depth}cm DEPTH")
        print(f"{'='*70}\n")
        
        try:
            # Initialize trainer
            trainer = GPRTrainer(
                depth_cm=depth,
                data_folder=DATA_FOLDER,
                models_folder=MODELS_FOLDER
            )
            
            # Load data
            X, y = trainer.load_data()
            
            # Train model
            trainer.train(
                X, y,
                test_size=0.2,
                n_pca_components=N_PCA_COMPONENTS
            )
            
            # Save the trained model
            trainer.save_model()
            
            print(f"\n{'='*70}")
            print(f"✓ Successfully trained and saved model for {depth}cm")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"\n✗ Error training model for {depth}cm: {str(e)}\n")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nModels saved in: {MODELS_FOLDER}/")
    print("Files created:")
    print("  - gpr_XXcm.pkl (trained model)")
    print("  - gpr_XXcm_metadata.json (model info)")
    print("  - gpr_XXcm_predictions.png (predictions with uncertainty)")


if __name__ == "__main__":
    main()
