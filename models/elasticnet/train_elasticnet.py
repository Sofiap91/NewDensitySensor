"""
ElasticNet Model Training for Density Sensor
============================================

ElasticNet is a linear regression model with both L1 (Lasso) and L2 (Ridge) 
regularization. It's good for:
- High-dimensional data (many features)
- Correlated features (like our frequency measurements)
- Feature selection (L1 penalty)
- Preventing overfitting (L2 penalty)

The model tries to find the best linear combination of features to predict shear strength.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
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


class ElasticNetTrainer:
    """Trains ElasticNet models for shear strength prediction"""
    
    def __init__(self, depth_cm: int, data_folder: str = "Output", models_folder: str = "models/elasticnet"):
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
        # Pattern: freq_XXX_real, freq_XXX_imag, freq_XXX_magnitude, freq_XXX_phase
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
    
    def create_pipeline(self, use_pca: bool = True, pca_variance: float = 0.95) -> Pipeline:
        """
        Create a scikit-learn pipeline with preprocessing and ElasticNet
        
        Pipeline steps:
        1. StandardScaler: Normalize features (mean=0, std=1)
        2. PCA (optional): Reduce dimensionality while keeping variance
        3. ElasticNet: Linear regression with L1+L2 regularization
        
        Args:
            use_pca: Whether to use PCA for dimensionality reduction
            pca_variance: Amount of variance to retain (0-1)
            
        Returns:
            Pipeline object
        """
        steps = [
            ('scaler', StandardScaler()),  # Normalize features
        ]
        
        if use_pca:
            steps.append(('pca', PCA(n_components=pca_variance)))  # Reduce dimensions
        
        steps.append(('elasticnet', ElasticNet(max_iter=10000)))  # The actual model
        
        return Pipeline(steps)
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2,
        use_pca: bool = True,
        tune_hyperparameters: bool = True
    ):
        """
        Train the ElasticNet model
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data to use for testing (0-1)
            use_pca: Whether to use PCA
            tune_hyperparameters: Whether to search for best hyperparameters
        """
        print("\n" + "="*70)
        print(f"Training ElasticNet Model for {self.depth_cm}cm Depth")
        print("="*70)
        
        # Split data into training and test sets
        print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Create pipeline
        pipeline = self.create_pipeline(use_pca=use_pca)
        
        if tune_hyperparameters:
            print("\nTuning hyperparameters with GridSearchCV...")
            print("This will try different combinations of parameters to find the best model")
            
            # Define parameter grid to search
            # alpha: regularization strength (higher = more regularization)
            # l1_ratio: balance between L1 and L2 (0=Ridge, 1=Lasso, 0.5=equal mix)
            param_grid = {
                'elasticnet__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
            
            if use_pca:
                # Also search for optimal PCA variance
                param_grid['pca__n_components'] = [0.90, 0.95, 0.99]
            
            # GridSearchCV will try all combinations and pick the best
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,  # 5-fold cross-validation
                scoring='neg_mean_absolute_error',  # Metric to optimize
                n_jobs=-1,  # Use all CPU cores
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            
            print("\nBest hyperparameters found:")
            for param, value in grid_search.best_params_.items():
                print(f"  {param}: {value}")
            
            self.metadata['best_params'] = grid_search.best_params_
            self.metadata['cv_score'] = -grid_search.best_score_  # Convert back to positive MAE
            
        else:
            # Use default parameters
            print("\nTraining with default parameters...")
            self.model = pipeline
            self.model.fit(X_train, y_train)
        
        # Check if PCA was used and report
        if use_pca and 'pca' in self.model.named_steps:
            pca = self.model.named_steps['pca']
            n_components = pca.n_components_
            explained_variance = np.sum(pca.explained_variance_ratio_)
            print(f"\nPCA reduced features from {X.shape[1]} to {n_components}")
            print(f"Explained variance: {explained_variance:.2%}")
            
            self.metadata['pca_components'] = int(n_components)
            self.metadata['pca_variance_explained'] = float(explained_variance)
        
        # Evaluate on training and test sets
        print("\n" + "="*70)
        print("Model Evaluation")
        print("="*70)
        
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)
        
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
        
        # Create visualization
        self._plot_predictions(y_test, y_test_pred, f"{self.depth_cm}cm Test Set")
        
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
    
    def _plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, title: str):
        """Create a scatter plot of predictions vs actual values"""
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
        
        # Save plot
        plot_file = self.models_folder / f"elasticnet_{self.depth_cm}cm_predictions.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved prediction plot: {plot_file}")
        plt.close()
    
    def save_model(self):
        """Save the trained model and metadata"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Save the model
        model_file = self.models_folder / f"elasticnet_{self.depth_cm}cm.pkl"
        joblib.dump(self.model, model_file)
        print(f"\nSaved model: {model_file}")
        
        # Save metadata
        self.metadata['depth_cm'] = self.depth_cm
        self.metadata['model_type'] = 'ElasticNet'
        self.metadata['feature_columns'] = self.feature_columns
        self.metadata['trained_at'] = datetime.now().isoformat()
        
        metadata_file = self.models_folder / f"elasticnet_{self.depth_cm}cm_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved metadata: {metadata_file}")
    
    def load_model(self):
        """Load a previously trained model"""
        model_file = self.models_folder / f"elasticnet_{self.depth_cm}cm.pkl"
        metadata_file = self.models_folder / f"elasticnet_{self.depth_cm}cm_metadata.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        self.model = joblib.load(model_file)
        print(f"Loaded model: {model_file}")
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            self.feature_columns = self.metadata['feature_columns']
            print(f"Loaded metadata: {metadata_file}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        return self.model.predict(X)


def main():
    """Main training function"""
    
    print("="*70)
    print("ElasticNet Model Training for Density Sensor")
    print("="*70)
    
    # Configuration
    DEPTHS = [20, 50, 80]  # Train models for all depths
    DATA_FOLDER = "Output"
    MODELS_FOLDER = "models/elasticnet"
    
    # Train a model for each depth
    for depth in DEPTHS:
        print(f"\n{'='*70}")
        print(f"TRAINING MODEL FOR {depth}cm DEPTH")
        print(f"{'='*70}\n")
        
        try:
            # Initialize trainer
            trainer = ElasticNetTrainer(
                depth_cm=depth,
                data_folder=DATA_FOLDER,
                models_folder=MODELS_FOLDER
            )
            
            # Load data
            X, y = trainer.load_data()
            
            # Train model with hyperparameter tuning
            trainer.train(
                X, y,
                test_size=0.2,
                use_pca=True,
                tune_hyperparameters=True
            )
            
            # Save the trained model
            trainer.save_model()
            
            print(f"\n{'='*70}")
            print(f"✓ Successfully trained and saved model for {depth}cm")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"\n✗ Error training model for {depth}cm: {str(e)}\n")
            continue
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nModels saved in: {MODELS_FOLDER}/")
    print("Files created:")
    print("  - elasticnet_XXcm.pkl (trained model)")
    print("  - elasticnet_XXcm_metadata.json (model info)")
    print("  - elasticnet_XXcm_predictions.png (visualization)")


if __name__ == "__main__":
    main()
