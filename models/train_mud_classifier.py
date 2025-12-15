"""
Two-Stage Prediction: Binary Classification + Regression
=========================================================

Stage 1: Binary classifier determines if sample is water (0) or mud (>0)
Stage 2: If mud, apply regression model to predict shear strength
         If water, output 0

This approach handles the discontinuity problem where electromagnetic
properties change continuously but shear strength jumps discontinuously.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)


class TwoStageModel:
    """Two-stage model: classifier + regression"""

    def __init__(self, depth_cm: int, regression_model_type: str = 'elasticnet'):
        """
        Initialize the two-stage model

        Args:
            depth_cm: Depth to train for (20, 50, or 80)
            regression_model_type: Type of regression model to use for stage 2
        """
        self.depth_cm = depth_cm
        self.regression_model_type = regression_model_type
        self.classifier = None
        self.regressor = None
        self.feature_columns = None
        self.metadata = {}


    def load_data(self, data_folder: str = "Output"):
        """Load and prepare data"""
        data_file = Path(data_folder) / f"training_data_{self.depth_cm}cm.csv"

        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)

        # Identify feature columns
        self.feature_columns = [col for col in df.columns if col.startswith('freq_')]

        # Extract features and target
        X = df[self.feature_columns].values
        y = df[f'shear_{self.depth_cm}cm'].values

        # Create binary labels: 0 for water (shear ≤ 0.1), 1 for mud (shear > 0.1)
        threshold = 0.1
        y_binary = (y > threshold).astype(int)

        n_water = np.sum(y_binary == 0)
        n_mud = np.sum(y_binary == 1)

        print(f"Loaded {len(df)} samples with {len(self.feature_columns)} features")
        print(f"Binary classification:")
        print(f"  Water samples (shear ≤ {threshold}): {n_water} ({n_water/len(df)*100:.1f}%)")
        print(f"  Mud samples (shear > {threshold}): {n_mud} ({n_mud/len(df)*100:.1f}%)")

        self.metadata['n_samples'] = len(df)
        self.metadata['n_features'] = len(self.feature_columns)
        self.metadata['n_water'] = int(n_water)
        self.metadata['n_mud'] = int(n_mud)
        self.metadata['classification_threshold'] = threshold

        return df, X, y, y_binary


    def train_classifier(self, X, y_binary, test_size=0.2):
        """Train the binary classifier (Stage 1)"""
        print("\n" + "="*70)
        print(f"STAGE 1: Training Binary Classifier (Water vs Mud)")
        print("="*70)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=42, stratify=y_binary
        )

        print(f"\nTraining samples: {len(X_train)} (Water: {np.sum(y_train==0)}, Mud: {np.sum(y_train==1)})")
        print(f"Test samples: {len(X_test)} (Water: {np.sum(y_test==0)}, Mud: {np.sum(y_test==1)})")

        # Train classifier
        print("\nTraining Random Forest Classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.classifier.fit(X_train, y_train)

        # Evaluate
        y_train_pred = self.classifier.predict(X_train)
        y_test_pred = self.classifier.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        print(f"\nClassification Performance:")
        print(f"  Training Accuracy: {train_acc:.3f}")
        print(f"  Test Accuracy: {test_acc:.3f}")

        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5)
        print(f"  Cross-Val Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

        # Detailed test set metrics
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, y_test_pred, target_names=['Water', 'Mud']))

        print("\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"                Predicted Water  Predicted Mud")
        print(f"Actual Water    {cm[0,0]:15d}  {cm[0,1]:13d}")
        print(f"Actual Mud      {cm[1,0]:15d}  {cm[1,1]:13d}")

        # Store metrics
        self.metadata['classifier_metrics'] = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'confusion_matrix': cm.tolist()
        }

        return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred


    def load_regression_model(self):
        """Load the pre-trained regression model (Stage 2)"""
        print("\n" + "="*70)
        print(f"STAGE 2: Loading Regression Model ({self.regression_model_type})")
        print("="*70)

        model_folder = Path(f"models/{self.regression_model_type}")
        model_file = model_folder / f"{self.regression_model_type}_{self.depth_cm}cm.pkl"

        if not model_file.exists():
            raise FileNotFoundError(f"Regression model not found: {model_file}")

        print(f"Loading regression model: {model_file}")
        self.regressor = joblib.load(model_file)
        print("✓ Regression model loaded")


    def predict_two_stage(self, X):
        """
        Make predictions using two-stage approach
        
        Args:
            X: Feature matrix
            
        Returns:
            predictions: Final shear strength predictions
            classifications: Binary classifications (0=water, 1=mud)
        """
        if self.classifier is None or self.regressor is None:
            raise ValueError("Models not trained/loaded. Train classifier and load regressor first.")
        
        # Stage 1: Classify
        classifications = self.classifier.predict(X)

        # Stage 2: Regress only for mud samples
        predictions = np.zeros(len(X))
        mud_mask = classifications == 1

        if np.sum(mud_mask) > 0:
            predictions[mud_mask] = self.regressor.predict(X[mud_mask])
            # Ensure no negative predictions for mud
            predictions[mud_mask] = np.maximum(predictions[mud_mask], 0)

        return predictions, classifications


    def evaluate_two_stage(self, X, y_true, y_binary_true):
        """Evaluate the two-stage model"""
        print("\n" + "="*70)
        print("TWO-STAGE MODEL EVALUATION")
        print("="*70)

        predictions, classifications = self.predict_two_stage(X)

        # Overall metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        r2 = r2_score(y_true, predictions)

        print(f"\nOverall Regression Metrics:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²:   {r2:.3f}")

        # Metrics for mud samples only
        mud_mask = y_binary_true == 1
        if np.sum(mud_mask) > 0:
            mae_mud = mean_absolute_error(y_true[mud_mask], predictions[mud_mask])
            rmse_mud = np.sqrt(mean_squared_error(y_true[mud_mask], predictions[mud_mask]))
            r2_mud = r2_score(y_true[mud_mask], predictions[mud_mask])

            print(f"\nMud Samples Only (Regression Performance):")
            print(f"  MAE:  {mae_mud:.2f}")
            print(f"  RMSE: {rmse_mud:.2f}")
            print(f"  R²:   {r2_mud:.3f}")

            self.metadata['regression_metrics_mud'] = {
                'mae': float(mae_mud),
                'rmse': float(rmse_mud),
                'r2': float(r2_mud)
            }

        # Metrics for water samples (should all be 0)
        water_mask = y_binary_true == 0
        if np.sum(water_mask) > 0:
            mae_water = mean_absolute_error(y_true[water_mask], predictions[water_mask])
            print(f"\nWater Samples (should be 0):")
            print(f"  MAE: {mae_water:.2f}")

        self.metadata['overall_metrics'] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }

        return predictions, classifications


    def save_model(self, models_folder: str = "models/mud_classifier"):
        """Save the mud classifier model"""
        folder = Path(models_folder)
        folder.mkdir(parents=True, exist_ok=True)

        # Save classifier
        classifier_file = folder / f"classifier_{self.depth_cm}cm.pkl"
        joblib.dump(self.classifier, classifier_file)
        print(f"\nSaved classifier: {classifier_file}")

        # Save metadata
        self.metadata['depth_cm'] = self.depth_cm
        self.metadata['regression_model_type'] = self.regression_model_type
        self.metadata['feature_columns'] = self.feature_columns
        self.metadata['trained_at'] = datetime.now().isoformat()

        metadata_file = folder / f"mud_classifier_{self.depth_cm}cm_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved metadata: {metadata_file}")

        print(f"\nNote: Regression model already exists at models/{self.regression_model_type}/")


def main():
    """Train two-stage model"""
    
    print("="*70)
    print("Two-Stage Model Training: Binary Classification + Regression")
    print("="*70)

    # Configuration
    DEPTH_CM = 20
    REGRESSION_MODEL = 'elasticnet'

    # Initialize model
    model = TwoStageModel(depth_cm=DEPTH_CM, regression_model_type=REGRESSION_MODEL)

    # Load data
    df, X, y, y_binary = model.load_data()

    # Train classifier (Stage 1)
    X_train, X_test, y_train_binary, y_test_binary, _, _ = model.train_classifier(X, y_binary)

    # Get corresponding continuous targets
    y_train = y[y_binary == y_binary]  # This preserves order
    # Need to properly split y to match train/test split
    _, _, y_train_cont, y_test_cont = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_binary
    )

    # Load regression model (Stage 2)
    model.load_regression_model()

    # Evaluate on test set
    print("\n" + "="*70)
    print("Evaluating on TEST SET")

    predictions_test, classifications_test = model.evaluate_two_stage(
        X_test, y_test_cont, y_test_binary
    )

    # Save model
    model.save_model()

    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Predictions vs Actual
    axes[0].scatter(y_test_cont, predictions_test, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[0].plot([0, y_test_cont.max()], [0, y_test_cont.max()], 'r--', lw=2, label='Perfect')
    axes[0].set_xlabel('Actual Shear Strength')
    axes[0].set_ylabel('Predicted Shear Strength')
    axes[0].set_title(f'Two-Stage Model: Test Set ({DEPTH_CM}cm)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Color by classification
    colors = ['blue' if c == 0 else 'red' for c in classifications_test]
    axes[1].scatter(y_test_cont, predictions_test, c=colors, alpha=0.6, 
                   edgecolors='k', linewidths=0.5)
    axes[1].plot([0, y_test_cont.max()], [0, y_test_cont.max()], 'k--', lw=2, label='Perfect')
    axes[1].set_xlabel('Actual Shear Strength')
    axes[1].set_ylabel('Predicted Shear Strength')
    axes[1].set_title(f'Two-Stage Model: Colored by Classification\n(Blue=Water, Red=Mud)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'mud_classifier_model_{DEPTH_CM}cm.png', dpi=150, bbox_inches='tight')

    print("Two-Stage Model Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
