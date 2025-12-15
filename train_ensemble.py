"""
Ensemble Tree-Based Models for Density Sensor
==============================================

This script trains three powerful ensemble methods:
1. Random Forest: Averages many decision trees (robust, handles non-linearity)
2. XGBoost: Gradient boosting with advanced regularization (industry standard)
3. LightGBM: Fast gradient boosting optimized for efficiency

Tree-based models are excellent for:
- Non-linear relationships (no assumptions about data distribution)
- High-dimensional data (automatic feature selection)
- Handling outliers and discontinuities (like our zero-value problem)
- No need for feature scaling
- Feature importance analysis

These models may handle the water/mud discontinuity better than 
regression models like ElasticNet, PLS, or GPR.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error
)

import matplotlib.pyplot as plt
import seaborn as sns

# Try to import XGBoost and LightGBM (they may need installation)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip3 install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip3 install lightgbm")

# Set random seed for reproducibility
np.random.seed(42)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class EnsembleTrainer:
    """Trains ensemble tree-based models for shear strength prediction"""
    
    def __init__(self, depth_cm: int, model_type: str, data_folder: str = "Output", 
                 models_folder: str = "models/ensemble"):
        """
        Initialize the trainer
        
        Args:
            depth_cm: Depth to train for (20, 50, or 80)
            model_type: 'random_forest', 'xgboost', or 'lightgbm'
            data_folder: Folder containing preprocessed CSV files
            models_folder: Folder to save trained models
        """
        self.depth_cm = depth_cm
        self.model_type = model_type.lower()
        self.data_folder = Path(data_folder)
        self.models_folder = Path(models_folder)
        self.models_folder.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_columns = None
        self.target_column = f'shear_{depth_cm}cm'
        self.metadata = {}
        
        # Validate model type
        valid_types = ['random_forest', 'xgboost', 'lightgbm']
        if self.model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}")
        
        # Check availability
        if self.model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Run: pip3 install xgboost")
        if self.model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Run: pip3 install lightgbm")
    
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
    
    def create_model(self):
        """Create the appropriate model based on model_type"""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == 'lightgbm':
            return lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
    
    def get_param_grid(self):
        """Get hyperparameter grid for the model type"""
        if self.model_type == 'random_forest':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        elif self.model_type == 'xgboost':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0]
            }
        
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0]
            }
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2,
        use_grid_search: bool = True
    ):
        """
        Train the ensemble model
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data to use for testing (0-1)
            use_grid_search: Whether to use GridSearchCV for hyperparameter tuning
        """
        model_name = self.model_type.replace('_', ' ').title()
        print("\n" + "="*70)
        print(f"Training {model_name} Model for {self.depth_cm}cm Depth")
        print("="*70)
        
        # Split data into training and test sets
        print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        if use_grid_search:
            print(f"\nPerforming hyperparameter tuning with GridSearchCV...")
            print("This may take several minutes...")
            
            base_model = self.create_model()
            param_grid = self.get_param_grid()
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            
            print(f"\nBest parameters found:")
            for param, value in grid_search.best_params_.items():
                print(f"  {param}: {value}")
            
            self.metadata['best_params'] = grid_search.best_params_
            self.metadata['cv_score'] = float(grid_search.best_score_)
            print(f"\nBest CV R² score: {grid_search.best_score_:.3f}")
        
        else:
            print(f"\nTraining {model_name} with default parameters...")
            self.model = self.create_model()
            self.model.fit(X_train, y_train)
        
        # Evaluate on training and test sets
        print("\n" + "="*70)
        print("Model Evaluation")
        print("="*70)
        
        # Make predictions
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
        self._interpret_results(test_metrics, train_metrics)
        
        # Feature importance
        self._analyze_feature_importance()
        
        # Create visualizations
        self._plot_predictions(y_test, y_test_pred, f"{self.depth_cm}cm Test Set")
        self._plot_feature_importance()
        
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
    
    def _interpret_results(self, test_metrics: dict, train_metrics: dict):
        """Provide interpretation of the results"""
        print("\n" + "-"*70)
        print("Model Quality Assessment:")
        print("-"*70)
        
        test_r2 = test_metrics['r2']
        train_r2 = train_metrics['r2']
        test_mae = test_metrics['mae']
        
        # R² interpretation
        if test_r2 > 0.9:
            print("✓ Excellent Test R² (>0.9) - Model explains >90% of variance")
        elif test_r2 > 0.8:
            print("✓ Good Test R² (>0.8) - Model explains >80% of variance")
        elif test_r2 > 0.6:
            print("✓ Fair Test R² (>0.6) - Model explains >60% of variance")
        elif test_r2 > 0.3:
            print("⚠ Moderate Test R² (>0.3) - Model explains >30% of variance")
        else:
            print("✗ Poor Test R² (<0.3) - Model explains <30% of variance")
        
        # Overfitting check
        r2_gap = train_r2 - test_r2
        if r2_gap > 0.3:
            print(f"⚠ Significant overfitting detected (Train R²={train_r2:.3f}, Test R²={test_r2:.3f})")
        elif r2_gap > 0.15:
            print(f"⚠ Moderate overfitting (Train R²={train_r2:.3f}, Test R²={test_r2:.3f})")
        else:
            print(f"✓ Good generalization (Train R²={train_r2:.3f}, Test R²={test_r2:.3f})")
        
        # MAE interpretation
        target_std = self.metadata['target_std']
        if test_mae < target_std * 0.3:
            print(f"✓ Excellent MAE - Predictions are very accurate")
        elif test_mae < target_std * 0.5:
            print(f"✓ Good MAE - Predictions are reasonably accurate")
        elif test_mae < target_std * 0.7:
            print(f"⚠ Fair MAE - Predictions have moderate error")
        else:
            print(f"⚠ High MAE - Predictions have significant error")
        
        print(f"\nAverage prediction error: ±{test_mae:.1f} units")
    
    def _analyze_feature_importance(self):
        """Extract and analyze feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # Get top 20 features
            indices = np.argsort(importances)[::-1][:20]
            top_features = [(self.feature_columns[i], importances[i]) for i in indices]
            
            self.metadata['top_features'] = [
                {'feature': feat, 'importance': float(imp)} 
                for feat, imp in top_features
            ]
            
            print(f"\n" + "-"*70)
            print("Top 20 Most Important Features:")
            print("-"*70)
            for i, (feat, imp) in enumerate(top_features, 1):
                print(f"{i:2d}. {feat:30s}  {imp:.4f}")
    
    def _plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, title: str):
        """Create prediction visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Predicted vs Actual
        axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Shear Strength')
        axes[0].set_ylabel('Predicted Shear Strength')
        axes[0].set_title(f'{title}\nPredicted vs Actual ({self.model_type.replace("_", " ").title()})')
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
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.models_folder / f"{self.model_type}_{self.depth_cm}cm_predictions.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved prediction plot: {plot_file}")
        plt.close()
    
    def _plot_feature_importance(self):
        """Plot feature importance"""
        if not hasattr(self.model, 'feature_importances_'):
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importances[indices])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_columns[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top 20 Features - {self.model_type.replace("_", " ").title()} ({self.depth_cm}cm)')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        plot_file = self.models_folder / f"{self.model_type}_{self.depth_cm}cm_importance.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Saved feature importance plot: {plot_file}")
        plt.close()
    
    def save_model(self):
        """Save the trained model and metadata"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Save the model
        model_file = self.models_folder / f"{self.model_type}_{self.depth_cm}cm.pkl"
        joblib.dump(self.model, model_file)
        print(f"\nSaved model: {model_file}")
        
        # Save metadata
        self.metadata['depth_cm'] = self.depth_cm
        self.metadata['model_type'] = self.model_type
        self.metadata['feature_columns'] = self.feature_columns
        self.metadata['trained_at'] = datetime.now().isoformat()
        
        metadata_file = self.models_folder / f"{self.model_type}_{self.depth_cm}cm_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved metadata: {metadata_file}")
    
    def load_model(self):
        """Load a previously trained model"""
        model_file = self.models_folder / f"{self.model_type}_{self.depth_cm}cm.pkl"
        metadata_file = self.models_folder / f"{self.model_type}_{self.depth_cm}cm_metadata.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        self.model = joblib.load(model_file)
        print(f"Loaded model: {model_file}")
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            self.feature_columns = self.metadata['feature_columns']
            print(f"Loaded metadata: {metadata_file}")
    
    def predict(self, X: np.ndarray):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        return self.model.predict(X)


def main():
    """Main training function"""
    
    print("="*70)
    print("Ensemble Tree-Based Models Training for Density Sensor")
    print("="*70)
    print("\nTraining Random Forest, XGBoost, and LightGBM models.")
    print("These models are excellent for non-linear relationships and")
    print("may handle the water/mud discontinuity better than linear models.\n")
    
    # Configuration
    DEPTHS = [20, 50, 80]
    MODEL_TYPES = ['random_forest']
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        MODEL_TYPES.append('xgboost')
    else:
        print("⚠ XGBoost not available. Install with: pip3 install xgboost\n")
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        MODEL_TYPES.append('lightgbm')
    else:
        print("⚠ LightGBM not available. Install with: pip3 install lightgbm\n")
    
    DATA_FOLDER = "Output"
    MODELS_FOLDER = "models/ensemble"
    USE_GRID_SEARCH = True  # Set to False for faster training
    
    # Train models for each combination of depth and model type
    results = []
    
    for model_type in MODEL_TYPES:
        for depth in DEPTHS:
            print(f"\n{'='*70}")
            print(f"TRAINING {model_type.upper().replace('_', ' ')} FOR {depth}cm DEPTH")
            print(f"{'='*70}\n")
            
            try:
                # Initialize trainer
                trainer = EnsembleTrainer(
                    depth_cm=depth,
                    model_type=model_type,
                    data_folder=DATA_FOLDER,
                    models_folder=MODELS_FOLDER
                )
                
                # Load data
                X, y = trainer.load_data()
                
                # Train model
                trainer.train(
                    X, y,
                    test_size=0.2,
                    use_grid_search=USE_GRID_SEARCH
                )
                
                # Save the trained model
                trainer.save_model()
                
                # Store results for comparison
                results.append({
                    'model_type': model_type,
                    'depth': depth,
                    'test_r2': trainer.metadata['test_metrics']['r2'],
                    'test_mae': trainer.metadata['test_metrics']['mae']
                })
                
                print(f"\n{'='*70}")
                print(f"✓ Successfully trained {model_type} for {depth}cm")
                print(f"{'='*70}\n")
                
            except Exception as e:
                print(f"\n✗ Error training {model_type} for {depth}cm: {str(e)}\n")
                import traceback
                traceback.print_exc()
                continue
    
    # Print comparison summary
    if results:
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('test_r2', ascending=False)
        
        print("\nRanked by Test R² Score:")
        print("-"*70)
        for i, row in df_results.iterrows():
            print(f"{row['model_type']:15s} {row['depth']:2d}cm  "
                  f"R²={row['test_r2']:6.3f}  MAE={row['test_mae']:6.2f}")
        
        best = df_results.iloc[0]
        print("\n" + "="*70)
        print(f"🏆 BEST MODEL: {best['model_type']} at {best['depth']}cm")
        print(f"   Test R² = {best['test_r2']:.3f}")
        print(f"   Test MAE = {best['test_mae']:.2f}")
        print("="*70)
    
    print(f"\n\nAll models saved in: {MODELS_FOLDER}/")


if __name__ == "__main__":
    main()
