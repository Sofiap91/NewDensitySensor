"""
PLS (Partial Least Squares) Regression for Density Sensor
==========================================================

PLS Regression is specifically designed for spectral data and is recommended
as the best starting point for this type of problem because it:

- Handles highly correlated features (like frequency measurements)
- Designed for spectroscopy and sensor data
- Learns latent components oriented toward the output
- Works well with small sample sizes
- Much more stable than neural networks with limited data
- Commonly used in NIR spectroscopy, radar, and dielectric sensing

PLS finds directions in the feature space that are maximally correlated
with the target variable, making it ideal for our VNA frequency data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error
)

import matplotlib.pyplot as plt
import seaborn as sns

# Import common training functions
from train_functions import (
    load_training_data,
    calculate_regression_metrics,
    print_metrics,
    interpret_metrics,
    plot_predictions,
    save_model_metadata,
    print_section_header
)

# Set random seed for reproducibility
np.random.seed(42)


class PLSTrainer:
    """Trains PLS Regression models for shear strength prediction"""
    
    def __init__(self, depth_cm: int, data_folder: str = "Output", models_folder: str = "models/pls"):
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
        df, X, y, feature_columns, metadata = load_training_data(
            depth_cm=self.depth_cm,
            data_folder=self.data_folder
        )

        self.feature_columns = feature_columns
        self.metadata.update(metadata)

        return X, y


    def create_pipeline(self, n_components: int = 10) -> Pipeline:
        """
        Create a scikit-learn pipeline with preprocessing and PLS

        Pipeline steps:
        1. StandardScaler: Normalize features (mean=0, std=1)
        2. PLSRegression: Find latent components for prediction

        Args:
            n_components: Number of PLS components to use

        Returns:
            Pipeline object
        """
        steps = [
            ('scaler', StandardScaler()),  # Normalize features
            ('pls', PLSRegression(n_components=n_components, scale=False))  # Already scaled
        ]

        return Pipeline(steps)


    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2,
        tune_hyperparameters: bool = True
    ):
        """
        Train the PLS model

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data to use for testing (0-1)
            tune_hyperparameters: Whether to search for best number of components
        """
        print("\n" + "="*70)
        print(f"Training PLS Regression Model for {self.depth_cm}cm Depth")
        print("="*70)

        # Split data into training and test sets
        print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        # Create pipeline
        pipeline = self.create_pipeline()

        if tune_hyperparameters:
            print("\nTuning number of PLS components with GridSearchCV...")
            print("Finding optimal number of latent variables for prediction")

            # Search for optimal number of components
            # Rule of thumb: n_components should be much less than n_samples
            max_components = min(30, X_train.shape[0] - 1, X_train.shape[1])

            param_grid = {
                'pls__n_components': list(range(2, max_components, 2))
            }

            print(f"Testing components from 2 to {max_components}")

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

            # Report on PLS components
            n_components = self.model.named_steps['pls'].n_components
            print(f"\nPLS using {n_components} latent components")
            print(f"Reduced from {X.shape[1]} features to {n_components} components")

            self.metadata['n_components'] = int(n_components)

        else:
            # Use default parameters
            print("\nTraining with default parameters (10 components)...")
            self.model = pipeline
            self.model.fit(X_train, y_train)

            n_components = self.model.named_steps['pls'].n_components
            self.metadata['n_components'] = int(n_components)

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

        # Plot component importance
        self._plot_component_importance()

        return self.model


    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate regression metrics"""
        # Handle 2D predictions from PLS (n_samples, 1) -> (n_samples,)
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        return calculate_regression_metrics(y_true, y_pred)


    def _print_metrics(self, metrics: dict):
        """Print metrics in a nice format"""
        print_metrics(metrics)


    def _interpret_results(self, metrics: dict):
        """Provide interpretation of the results"""
        interpret_metrics(metrics, self.metadata['target_std'])


    def _plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, title: str):
        """Create a scatter plot of predictions vs actual values"""
        # Handle 2D predictions from PLS
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        plot_file = self.models_folder / f"pls_{self.depth_cm}cm_predictions.png"
        plot_predictions(y_true, y_pred, title, plot_file)


    def _plot_component_importance(self):
        """Plot the explained variance of each PLS component"""
        pls_model = self.model.named_steps['pls']

        # Get X and Y variance explained by each component
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        n_components = pls_model.n_components
        components = range(1, n_components + 1)

        # X variance (features)
        x_var = np.var(pls_model.x_scores_, axis=0)
        x_var_ratio = x_var / x_var.sum()

        axes[0].bar(components, x_var_ratio, alpha=0.7, color='steelblue')
        axes[0].set_xlabel('PLS Component')
        axes[0].set_ylabel('Proportion of X Variance')
        axes[0].set_title(f'{self.depth_cm}cm: Feature Space Variance\nExplained by Each Component')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Cumulative variance
        cumsum_var = np.cumsum(x_var_ratio)
        axes[1].plot(components, cumsum_var, marker='o', linewidth=2, markersize=6, color='darkgreen')
        axes[1].axhline(y=0.95, color='r', linestyle='--', linewidth=1, label='95% threshold')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Variance Explained')
        axes[1].set_title(f'{self.depth_cm}cm: Cumulative Variance\n(Feature Space)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])

        plt.tight_layout()

        # Save plot
        plot_file = self.models_folder / f"pls_{self.depth_cm}cm_components.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Saved component plot: {plot_file}")
        plt.close()


    def save_model(self):
        """Save the trained model and metadata"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        # Save the model
        model_file = self.models_folder / f"pls_{self.depth_cm}cm.pkl"
        joblib.dump(self.model, model_file)
        print(f"\nSaved model: {model_file}")

        # Save metadata using common function
        save_model_metadata(
            self.metadata,
            model_type='PLS_Regression',
            depth_cm=self.depth_cm,
            feature_columns=self.feature_columns,
            models_folder=self.models_folder
        )


    def load_model(self):
        """Load a previously trained model"""
        model_file = self.models_folder / f"pls_{self.depth_cm}cm.pkl"
        metadata_file = self.models_folder / f"pls_{self.depth_cm}cm_metadata.json"

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

        predictions = self.model.predict(X)

        # Handle 2D output
        if predictions.ndim > 1:
            predictions = predictions.ravel()

        return predictions


def main():
    """Main training function"""

    print("="*70)
    print("PLS Regression Model Training for Density Sensor")
    print("="*70)
    print("\nPLS (Partial Least Squares) Regression is specifically designed for")
    print("spectral data like VNA measurements. It's the recommended starting point")
    print("for this type of sensor data analysis.\n")

    # Configuration
    DEPTHS = [20, 50, 80]  # Train models for all depths
    script_dir = Path(__file__).resolve().parent.parent  # Go up to NewDensitySensor/
    DATA_FOLDER = script_dir / "Output"
    MODELS_FOLDER = script_dir / "models" / "PLS"

    # Train a model for each depth
    for depth in DEPTHS:
        print(f"TRAINING MODEL FOR {depth}cm DEPTH")

        try:
            # Initialize trainer
            trainer = PLSTrainer(
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
                tune_hyperparameters=True
            )

            # Save the trained model
            trainer.save_model()

            print(f"\n{'='*70}")
            print("Training Complete!")
            print(f"\nModels saved in: {MODELS_FOLDER}/")

        except Exception as e:
            print(f"\n✗ Error training model for {depth}cm: {str(e)}\n")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
