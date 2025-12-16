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


class ElasticNetTrainer:
    """Trains ElasticNet models for shear strength prediction"""

    def __init__(self, depth_cm: int, data_folder: str = "Output", models_folder: str = "models/elasticnet", 
                 mud_only: bool = False, mud_threshold: float = 0.1):
        """
        Initialize the trainer

        Args:
            depth_cm: Depth to train for (20, 50, or 80)
            data_folder: Folder containing preprocessed CSV files
            models_folder: Folder to save trained models
            mud_only: If True, train only on mud samples (shear > threshold)
            mud_threshold: Threshold to separate water from mud samples
        """
        self.depth_cm = depth_cm
        self.data_folder = Path(data_folder)
        self.models_folder = Path(models_folder)
        self.models_folder.mkdir(parents=True, exist_ok=True)
        self.mud_only = mud_only
        self.mud_threshold = mud_threshold

        self.model = None
        self.feature_columns = None
        self.target_column = f'shear_{depth_cm}cm'
        self.metadata = {}
        self.metadata['mud_only'] = mud_only
        if mud_only:
            self.metadata['mud_threshold'] = mud_threshold


    def load_data(self) -> tuple:
        """
        Load preprocessed data for the specified depth

        Returns:
            X: Feature matrix
            y: Target vector
        """
        df, X, y, feature_columns, metadata = load_training_data(
            depth_cm=self.depth_cm,
            data_folder=self.data_folder,
            mud_only=self.mud_only,
            mud_threshold=self.mud_threshold
        )
        
        self.feature_columns = feature_columns
        self.metadata.update(metadata)
        
        return X, y


    def create_pipeline(self, use_pca: bool = True, pca_variance: float = 0.95) -> Pipeline:
        """
        Create a scikit-learn pipeline with preprocessing and ElasticNet

        Pipeline steps:
        1. StandardScaler: Normalize features (mean=0, std=1)
        2. PCA: Reduce dimensionality while keeping variance
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
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # print(f"Training samples: {len(X_train)}")
        # print(f"Test samples: {len(X_test)}")

        # Create pipeline
        pipeline = self.create_pipeline(use_pca=use_pca)

        if tune_hyperparameters:
            # print("This will try different combinations of parameters to find the best model")

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
        return calculate_regression_metrics(y_true, y_pred)


    def _print_metrics(self, metrics: dict):
        """Print metrics in a nice format"""
        print_metrics(metrics)


    def _interpret_results(self, metrics: dict):
        """Provide interpretation of the results"""
        interpret_metrics(metrics, self.metadata['target_std'])


    def _plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, title: str):
        """Create a scatter plot of predictions vs actual values"""
        plot_file = self.models_folder / f"elasticnet_{self.depth_cm}cm_predictions.png"
        plot_predictions(y_true, y_pred, title, plot_file)


    def save_model(self):
        """Save the trained model and metadata"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        # Save the model
        model_file = self.models_folder / f"elasticnet_{self.depth_cm}cm.pkl"
        joblib.dump(self.model, model_file)
        print(f"\nSaved model: {model_file}")

        # Save metadata using common function
        save_model_metadata(
            self.metadata,
            model_type='elasticnet',
            depth_cm=self.depth_cm,
            feature_columns=self.feature_columns,
            models_folder=self.models_folder
        )


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
    MUD_ONLY = True  # Set to True for two-stage model (recommended)
    MUD_THRESHOLD = 0.1  # Threshold to separate water from mud

    # Use absolute paths relative to the script location
    script_dir = Path(__file__).resolve().parent.parent  # Go up to NewDensitySensor/
    DATA_FOLDER = script_dir / "Output"
    MODELS_FOLDER = script_dir / "models" / "elasticnet"

    if MUD_ONLY:
        print(f"\nMUD-ONLY MODE ENABLED")
        print(f"Training only on samples with shear > {MUD_THRESHOLD}")
        print(f"This prevents train-test distribution mismatch in two-stage models.")
        print()

    print(f"\nScript directory: {script_dir}")
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Models folder: {MODELS_FOLDER}")
    print(f"Data folder exists: {DATA_FOLDER.exists()}")
    print(f"Models folder exists: {MODELS_FOLDER.exists()}\n")

    # Train a model for each depth
    for depth in DEPTHS:
        print(f"\n" + "="*70)
        print(f"TRAINING MODEL FOR {depth}cm DEPTH")
        print("="*70)

        try:
            # Initialize trainer
            trainer = ElasticNetTrainer(
                depth_cm=depth,
                data_folder=DATA_FOLDER,
                models_folder=MODELS_FOLDER,
                mud_only=MUD_ONLY,
                mud_threshold=MUD_THRESHOLD
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

            # print(f"\n{'='*70}")
            # print(f"Successfully trained and saved model for {depth}cm")

            print("\n" + "="*70)
            print("Training Complete!")
            print(f"\nModels saved in: {MODELS_FOLDER}/")

        except Exception as e:
            print(f"\nError training model for {depth}cm: {str(e)}\n")
            continue


if __name__ == "__main__":
    main()
