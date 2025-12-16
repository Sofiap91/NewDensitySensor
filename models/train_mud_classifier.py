"""
Binary Classifier Training: Water vs Mud
=========================================

Trains a RandomForest binary classifier to separate water (shear=0) from mud (shear>0).
This classifier is Stage 1 of the two-stage prediction approach.

Stage 2 (regression models) are trained separately and combined at prediction time in main.py.
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


class BinaryClassifier:
    """Binary classifier for water vs mud detection"""

    def __init__(self, depth_cm: int):
        """
        Initialize the classifier

        Args:
            depth_cm: Depth to train for (20, 50, or 80)
        """
        self.depth_cm = depth_cm
        self.classifier = None
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
        print(f"\nTraining Binary Classifier (Water vs Mud)")

        # Check if we have both classes
        unique_classes = np.unique(y_binary)
        has_both_classes = len(unique_classes) == 2

        if not has_both_classes:
            print("\nWarning: Only one class present in data!")
            if 0 not in unique_classes:
                print("All samples are MUD (no water samples)")
            else:
                print("All samples are WATER (no mud samples)")

        # Split data (disable stratify if only one class)
        if has_both_classes:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_binary, test_size=test_size, random_state=42, stratify=y_binary
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_binary, test_size=test_size, random_state=42
            )

        print(f"\nTraining samples: {len(X_train)} (Water: {np.sum(y_train==0)}, Mud: {np.sum(y_train==1)})")
        print(f"Test samples: {len(X_test)} (Water: {np.sum(y_test==0)}, Mud: {np.sum(y_test==1)})")

        # Train Random Forest classifier
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
        print(f"Training Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")

        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5)
        print(f"Cross-Val Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

        # Detailed test set metrics (only if both classes present)
        if has_both_classes and len(np.unique(y_test)) == 2:
            print("\nTest Set Classification Report:")
            print(classification_report(y_test, y_test_pred, target_names=['Water', 'Mud'], zero_division=0))

            print("\nConfusion Matrix (Test Set):")
            cm = confusion_matrix(y_test, y_test_pred)
            print(f"                Predicted Water  Predicted Mud")
            print(f"Actual Water    {cm[0,0]:15d}  {cm[0,1]:13d}")
            print(f"Actual Mud      {cm[1,0]:15d}  {cm[1,1]:13d}")
        else:
            print("\nNote: Skipping detailed classification report (single class only)")
            cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

        # Store metrics
        self.metadata['classifier_metrics'] = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'confusion_matrix': cm.tolist(),
            'has_both_classes': bool(has_both_classes)
        }

        return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred


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
        self.metadata['model_type'] = 'RandomForest_BinaryClassifier'
        self.metadata['feature_columns'] = self.feature_columns
        self.metadata['trained_at'] = datetime.now().isoformat()

        metadata_file = folder / f"classifier_{self.depth_cm}cm_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)


def main():
    """Train binary classifiers"""

    print("="*70)
    print("Binary Classifier Training: Water vs Mud")
    print("="*70)

    # Configuration
    DEPTHS = [20, 50, 80]

    # Train for each depth
    for DEPTH_CM in DEPTHS:
        print(f"TRAINING CLASSIFIER FOR {DEPTH_CM}cm DEPTH")

        try:
            # Initialize classifier
            classifier = BinaryClassifier(depth_cm=DEPTH_CM)

            # Load data
            df, X, y, y_binary = classifier.load_data()

            # Train classifier
            X_train, X_test, y_train_binary, y_test_binary, _, _ = classifier.train_classifier(X, y_binary)

            # Save model
            classifier.save_model()

            # Create visualization showing classification performance
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot 1: Training set classifications
            axes[0].scatter(range(len(y_train_binary)), y_train_binary,
                           c=['blue' if y == 0 else 'red' for y in y_train_binary],
                           alpha=0.6, edgecolors='k', linewidths=0.5, label='Actual')
            axes[0].set_xlabel('Sample Index')
            axes[0].set_ylabel('Class (0=Water, 1=Mud)')
            axes[0].set_title(f'Training Set Classification ({DEPTH_CM}cm)\n(Blue=Water, Red=Mud)')
            axes[0].set_ylim([-0.2, 1.2])
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Test set predictions vs actual
            y_test_pred = classifier.classifier.predict(X_test)
            colors_actual = ['blue' if y == 0 else 'red' for y in y_test_binary]
            colors_pred = ['cyan' if y == 0 else 'orange' for y in y_test_pred]

            x_pos = range(len(y_test_binary))
            axes[1].scatter(x_pos, y_test_binary, c=colors_actual, marker='o', s=100,
                           alpha=0.6, edgecolors='k', linewidths=1.5, label='Actual')
            axes[1].scatter(x_pos, y_test_pred, c=colors_pred, marker='x', s=100,
                           linewidths=2, label='Predicted')
            axes[1].set_xlabel('Sample Index')
            axes[1].set_ylabel('Class (0=Water, 1=Mud)')
            axes[1].set_title(f'Test Set: Predicted vs Actual ({DEPTH_CM}cm)')
            axes[1].set_ylim([-0.2, 1.2])
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'classifier_{DEPTH_CM}cm.png', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\nClassifier training complete for {DEPTH_CM}cm!")

        except Exception as e:
            print(f"\nError training classifier for {DEPTH_CM}cm: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nALL CLASSIFIERS TRAINED!")


if __name__ == "__main__":
    main()
