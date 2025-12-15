"""
Main Pipeline for Density Sensor Predictions
=============================================

This script orchestrates the full pipeline:
1. Preprocess raw VNA and Vane Shear data (optional - if you have new raw data)
2. Load trained models (classifier + regressor)
3. Make predictions on processed data
4. Export results to CSV

Usage:
    python3 main.py --mode predict          # Just predict using existing processed data
    python3 main.py --mode full             # Preprocess + predict
    python3 main.py --depth 20              # Specify depth (default: 20)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from data_processor import DataProcessor


class DensitySensorPipeline:
    """Main pipeline for density sensor predictions"""

    def __init__(self, depth_cm: int = 20, mode: str = 'predict'):
        """
        Initialize the pipeline

        Args:
            depth_cm: Depth to process (20, 50, or 80)
        """
        self.depth_cm = depth_cm
        self.classifier = None
        self.regressor = None
        self.data = None
        self.mode = mode

        self.data_processor = DataProcessor(
                input_folder="Input",
                output_folder="Output",
                site_filter="zz Brisbane"
            )

        print("="*70)
        print(f"Density Sensor Pipeline - {depth_cm}cm Depth")
        print("="*70)

        self.run()


    def step1_preprocess_data(self):
        """
        Step 1: Run data preprocessing
        
        This matches VNA measurements with Vane Shear ground truth data
        """
        print("\n" + "="*70)
        print("STEP 1: Data Preprocessing")
        print("="*70)

        try:
            # Run the preprocessing
            self.data_processor.run()

            # Check if output files were created
            output_file = Path("Output") / f"training_data_{self.depth_cm}cm.csv"
            if not output_file.exists():
                print(f"✗ Expected output file not found: {output_file}")
                return False

            print(f"✓ Preprocessing complete! Created {output_file}")
            return True

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return False


    def step2_load_models(self):
        """
        Step 2: Load trained models

        Loads the classifier (water vs mud) and regressor (shear strength)
        These are pre-trained .pkl files - no training happens here!
        """
        print("\n" + "="*70)
        print("STEP 2: Loading Trained Models")
        print("="*70)

        # Load classifier
        classifier_file = Path("models/mud_classifier") / f"classifier_{self.depth_cm}cm.pkl"
        if not classifier_file.exists():
            print(f"✗ Classifier not found: {classifier_file}")
            print("   Run train_mud_classifier.py first to create the model!")
            return False

        # print(f"\nLoading classifier: {classifier_file}")
        self.classifier = joblib.load(classifier_file)
        print("Classifier loaded")

        # Load regressor
        regressor_file = Path("models/elasticnet") / f"elasticnet_{self.depth_cm}cm.pkl"
        if not regressor_file.exists():
            print(f"Regressor not found: {regressor_file}")
            print("Run train_elasticnet.py first to create the model!")
            return False

        # print(f"Loading regressor: {regressor_file}")
        self.regressor = joblib.load(regressor_file)
        print("Regressor loaded")

        print("\nAll models loaded successfully!")
        return True


    def step3_load_data(self):
        """
        Step 3: Load preprocessed data

        This is the data we want to make predictions on
        """
        print("\n" + "="*70)
        print("STEP 3: Loading Preprocessed Data")
        print("="*70)

        data_file = Path("Output") / f"training_data_{self.depth_cm}cm.csv"

        if not data_file.exists():
            print(f"Data file not found: {data_file}")
            # print("Run with --mode full to preprocess data first")
            return False

        # print(f"\nLoading data from: {data_file}")
        self.data = pd.read_csv(data_file)

        # print(f"Loaded {len(self.data)} samples")
        print(f"Columns: {len(self.data.columns)}")

        return True


    def step4_make_predictions(self):
        """
        Step 4: Make predictions using the two-stage model
        
        Stage 1: Classify as Water (0) or Mud (>0)
        Stage 2: If Mud, predict shear strength; if Water, return 0
        """
        print("\n" + "="*70)
        print("STEP 4: Making Predictions")
        print("="*70)

        if self.classifier is None or self.regressor is None:
            print("Models not loaded!")
            return None

        if self.data is None:
            print("Data not loaded!")
            return None

        # Extract features
        feature_columns = [col for col in self.data.columns if col.startswith('freq_')]
        X = self.data[feature_columns].values

        print(f"\nPredicting on {len(X)} samples with {len(feature_columns)} features...")

        # Stage 1: Classification
        print("\n  Stage 1: Classifying samples (Water vs Mud)...")
        classifications = self.classifier.predict(X)

        n_water = np.sum(classifications == 0)
        n_mud = np.sum(classifications == 1)

        print(f"Water: {n_water} samples ({n_water/len(X)*100:.1f}%)")
        print(f"Mud:   {n_mud} samples ({n_mud/len(X)*100:.1f}%)")

        # Stage 2: Regression (only for mud)
        print("\n  Stage 2: Predicting shear strength for mud samples...")
        predictions = np.zeros(len(X))
        mud_mask = classifications == 1

        if np.sum(mud_mask) > 0:
            predictions[mud_mask] = self.regressor.predict(X[mud_mask])
            predictions[mud_mask] = np.maximum(predictions[mud_mask], 0)  # No negative values
            print(f"Predicted {np.sum(mud_mask)} mud samples")

        print(f"Water samples automatically set to 0 kPa")

        print("\nPredictions complete!")
        print(f"Range: {predictions.min():.2f} to {predictions.max():.2f} kPa")

        return predictions, classifications


    def step5_export_results(self, predictions, classifications):
        """
        Step 5: Export results to CSV

        Creates a comprehensive CSV with all predictions and metadata
        """
        print("\n" + "="*70)
        print("STEP 5: Exporting Results")
        print("="*70)

        # Create results dataframe
        results_df = pd.DataFrame({
            'measurement_datetime': self.data['measurement_datetime'],
            'latitude': self.data.get('latitude', self.data.get('vna_latitude', np.nan)),
            'longitude': self.data.get('longitude', self.data.get('vna_longitude', np.nan)),
            'vna_latitude': self.data.get('vna_latitude', np.nan),
            'vna_longitude': self.data.get('vna_longitude', np.nan),
            'depth_cm': self.depth_cm,
            'classification': ['Water' if c == 0 else 'Mud' for c in classifications],
            'classification_code': classifications,
            'predicted_shear_strength_kpa': predictions,
        })

        # Add actual values if available
        actual_col = f'shear_{self.depth_cm}cm'
        if actual_col in self.data.columns:
            results_df['actual_shear_strength_kpa'] = self.data[actual_col]
            results_df['prediction_error_kpa'] = predictions - self.data[actual_col]
            results_df['absolute_error_kpa'] = np.abs(predictions - self.data[actual_col])

        # Sort by datetime
        results_df = results_df.sort_values('measurement_datetime')

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"predictions_{self.depth_cm}cm_{timestamp}.csv"

        # Save to CSV
        results_df.to_csv(output_file, index=False)

        print(f"\nSaved predictions to: {output_file}")

        # Print summary
        print("\n" + "-"*70)
        print("Prediction Summary:")
        print("-"*70)

        if 'actual_shear_strength_kpa' in results_df.columns:
            from sklearn.metrics import mean_absolute_error, r2_score
            mae = mean_absolute_error(
                results_df['actual_shear_strength_kpa'], 
                results_df['predicted_shear_strength_kpa']
            )
            r2 = r2_score(
                results_df['actual_shear_strength_kpa'],
                results_df['predicted_shear_strength_kpa']
            )
    
            print(f"Mean Absolute Error: {mae:.2f} kPa")
            print(f"R² Score: {r2:.3f}")

        print(f"Prediction range: {predictions.min():.2f} to {predictions.max():.2f} kPa")

        return output_file


    def run(self):
        """
        Run the full pipeline
        
        Args:
            mode: 'predict' (skip preprocessing) or 'full' (include preprocessing)
        """
        print(f"\nRunning pipeline in '{self.mode}' mode")

        # Step 1: Preprocess (optional)
        if self.mode == 'full':
            if not self.step1_preprocess_data():
                print("\nPipeline failed at Step 1 (Preprocessing)")
                return False
        else:
            print("\nSkipping preprocessing (using existing data)")

        # Step 2: Load models
        if not self.step2_load_models():
            print("\nPipeline failed at Step 2 (Load Models)")
            return False

        # # Step 3: Load data
        if not self.step3_load_data():
            print("\nPipeline failed at Step 3 (Load Data)")
            return False

        # # Step 4: Make predictions
        predictions, classifications = self.step4_make_predictions()
        if predictions is None:
            print("\nPipeline failed at Step 4 (Predictions)")
            return False

        # # Step 5: Export results
        output_file = self.step5_export_results(predictions, classifications)

        return True


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Density Sensor Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python3 main.py                          # Predict with default settings (20cm)
            python3 main.py --depth 50               # Predict for 50cm depth
            python3 main.py --mode full              # Preprocess + predict
            python3 main.py --depth 80 --mode full   # Full pipeline for 80cm

            Modes:
            predict - Use existing preprocessed data (faster)
            full    - Run preprocessing then predict (use when you have new raw data)
            """
    )
    
    parser.add_argument(
        '--depth',
        type=int,
        choices=[20, 50, 80],
        default=20,
        help='Measurement depth in cm (default: 20)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['predict', 'full'],
        default='predict',
        help='Pipeline mode: "predict" (use existing data) or "full" (preprocess + predict)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = DensitySensorPipeline(depth_cm=args.depth, mode=args.mode)


if __name__ == "__main__":
    main()
