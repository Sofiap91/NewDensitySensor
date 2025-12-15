"""
Data Preprocessing Script for Density Sensor ML Training
=========================================================

This script matches VNA (Vector Network Analyzer) frequency measurements 
with Vane Shear test results to create training data for ML models.

Simple Usage:
    preprocessor = DataPreprocessor()
    preprocessor.run()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Single unified class to handle all data preprocessing.

    Simple usage:
        preprocessor = DataPreprocessor()
        preprocessor.run()  # That's it!
    """

    def __init__(self, 
                 input_folder: str = "Input",
                 output_folder: str = "Output",
                 site_filter: str = "zz Brisbane"):
        """
        Initialize the preprocessor

        Args:
            input_folder: Folder containing VNA transfer files and Vane Shear Report
            output_folder: Where to save processed data
            site_filter: Site name to filter Vane Shear data
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)

        self.site_filter = site_filter


    def run(self):
        """
        Run the complete preprocessing pipeline.
        This is the main method you call!
        """
        # Step 1: Load VNA data
        print("\n[Step 1/4] Loading VNA measurements...")
        vna_df = self._load_vna_data()

        # Step 2: Load Vane Shear data
        print("\n[Step 2/4] Loading Vane Shear measurements...")
        vane_shear_df = self._load_vane_shear_data()

        # Step 3: Match and combine
        print("\n[Step 3/4] Matching measurements...")
        training_df = self._match_and_combine(vna_df, vane_shear_df)

        # Step 4: Save outputs
        print("\n[Step 4/4] Saving datasets...")
        self._save_datasets(training_df)

        print("\n" + "="*70)
        print("Preprocessing Complete!")
        print("="*70)
        print(f"\nOutput files created in {self.output_folder}/:")
        print("  - training_data_complete.csv")
        print("  - training_data_20cm.csv")
        print("  - training_data_50cm.csv")
        print("  - training_data_80cm.csv")



    def _load_vna_data(self) -> pd.DataFrame:
        """Load and parse VNA transfer files"""
        transfer_files = sorted(self.input_folder.glob('transfer_*.csv'))

        if not transfer_files:
            raise FileNotFoundError(f"No transfer_*.csv files found in {self.input_folder}")

        print(f"Found {len(transfer_files)} transfer files")

        all_vna_data = []

        for file_path in transfer_files:
            print(f"  Loading {file_path.name}...")
            df = pd.read_csv(file_path)

            # Parse timestamps
            df['measurement_datetime'] = pd.to_datetime(df['measurement_date'])

            # Parse each row's frequency data
            for idx, row in df.iterrows():
                sensor_data = self._parse_sensor_data(row['Raw Sensor Data'])

                if sensor_data is not None:
                    vna_record = {
                        'measurement_datetime': row['measurement_datetime'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'distance_to_ground': row['DistanceToGround'],
                        **sensor_data
                    }
                    all_vna_data.append(vna_record)

        print(f"Loaded {len(all_vna_data)} VNA measurements")
        return pd.DataFrame(all_vna_data)


    def _parse_sensor_data(self, raw_data: str) -> Optional[Dict]:
        """Parse frequency data from Raw Sensor Data field"""
        if pd.isna(raw_data) or not isinstance(raw_data, str):
            return None

        frequencies, real_s11, imag_s11 = [], [], []

        for line in raw_data.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('!') or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) == 3:
                try:
                    frequencies.append(float(parts[0]))
                    real_s11.append(float(parts[1]))
                    imag_s11.append(float(parts[2]))
                except ValueError:
                    continue

        if not frequencies:
            return None

        return {
            'frequencies': np.array(frequencies),
            'real_s11': np.array(real_s11),
            'imag_s11': np.array(imag_s11),
            'n_frequencies': len(frequencies)
        }


    def _load_vane_shear_data(self) -> pd.DataFrame:
        """
        Load Vane Shear Report

        Returns:
            DataFrame with Vane Shear measurements
        """
        vane_shear_file = self.input_folder / "Vane Shear Report.csv"

        if not vane_shear_file.exists():
            raise FileNotFoundError(f"Vane Shear Report.csv not found in {self.input_folder}")

        print(f"Loading Vane Shear Report...")
        df = pd.read_csv(vane_shear_file)

        # Filter by site
        if self.site_filter:
            df = df[df['Site'] == self.site_filter].copy()
            print(f"Filtered to site '{self.site_filter}': {len(df)} measurements")

        # Parse timestamps
        df['datetime'] = pd.to_datetime(df['DateTime'])
        df['created_datetime'] = pd.to_datetime(df['Created'])

        # Extract depth and shear strength
        df['depth_cm'] = df['Depth'].astype(int) / 10  # mm to cm
        df['shear_strength'] = df['ShearStrenght'].astype(float)

        df = df.sort_values('datetime').reset_index(drop=True)

        print(f"Loaded {len(df)} Vane Shear measurements")
        print(f"Depth distribution: {df['depth_cm'].value_counts().sort_index().to_dict()}")

        return df


    def _match_and_combine(self, vna_df: pd.DataFrame, vane_shear_df: pd.DataFrame) -> pd.DataFrame:
        """Match VNA measurements with Vane Shear tests"""

        # Group Vane Shear measurements by location
        location_groups = self._group_by_location(vane_shear_df)
        print(f"Found {len(location_groups)} measurement locations")

        training_records = []
        matched_count = 0

        for group_idx, group in enumerate(location_groups):
            group_data = vane_shear_df.loc[group]
            reference_datetime = group_data.iloc[0]['created_datetime']

            # Find closest VNA measurement
            vna_match = self._find_closest_vna(vna_df, reference_datetime)

            if vna_match is None:
                continue

            # Create training record
            record = {
                'measurement_datetime': reference_datetime,
                'vna_datetime': vna_match['measurement_datetime'],
                'time_diff_seconds': abs((reference_datetime - vna_match['measurement_datetime']).total_seconds()),
                'latitude': self._extract_gps_component(group_data.iloc[0].get('Default GPS'), 0),
                'longitude': self._extract_gps_component(group_data.iloc[0].get('Default GPS'), 1),
                'vna_latitude': vna_match['latitude'],
                'vna_longitude': vna_match['longitude'],
            }

            # Add frequency features
            for i, (freq, re_val, im_val) in enumerate(zip(vna_match['frequencies'], 
                                                            vna_match['real_s11'], 
                                                            vna_match['imag_s11'])):
                freq_mhz = int(freq / 1e6)
                record[f'freq_{freq_mhz}_real'] = re_val
                record[f'freq_{freq_mhz}_imag'] = im_val
                record[f'freq_{freq_mhz}_magnitude'] = np.sqrt(re_val**2 + im_val**2)
                record[f'freq_{freq_mhz}_phase'] = np.arctan2(im_val, re_val)

            # Add shear strength values for each depth
            record['shear_20cm'] = None
            record['shear_50cm'] = None
            record['shear_80cm'] = None

            for _, depth_row in group_data.iterrows():
                depth_cm = depth_row['depth_cm']
                shear_val = depth_row['shear_strength']

                if depth_cm == 20:
                    record['shear_20cm'] = shear_val
                elif depth_cm == 50:
                    record['shear_50cm'] = shear_val
                elif depth_cm == 80:
                    record['shear_80cm'] = shear_val

            training_records.append(record)
            matched_count += 1

        print(f"Successfully matched {matched_count} locations")

        df = pd.DataFrame(training_records)

        # Report completeness
        print(f"\nShear strength data completeness:")
        for depth in [20, 50, 80]:
            col = f'shear_{depth}cm'
            count = df[col].notna().sum()
            pct = 100 * count / len(df)
            print(f"  {depth}cm: {count} / {len(df)} ({pct:.1f}%)")

        return df


    def _group_by_location(self, vane_shear_df: pd.DataFrame) -> List[List[int]]:
        """Group Vane Shear measurements by location (20cm, 50cm, 80cm sequence)"""
        groups = []
        current_group = []
        prev_datetime = None
        prev_depth = None

        for idx, row in vane_shear_df.iterrows():
            current_datetime = row['datetime']
            current_depth = row['depth_cm']

            # Start new group if:
            # 1. First measurement
            # 2. Back to 20cm after deeper depths (new location)
            # 3. Time gap > 10 minutes
            start_new_group = False

            if prev_datetime is None:
                start_new_group = True
            elif prev_depth is not None and current_depth == 20 and prev_depth in [50, 80]:
                start_new_group = True
            else:
                time_diff = abs((current_datetime - prev_datetime).total_seconds())
                if time_diff > 600:
                    start_new_group = True

            if start_new_group:
                if current_group:
                    groups.append(current_group)
                current_group = [idx]
            else:
                current_group.append(idx)

            prev_datetime = current_datetime
            prev_depth = current_depth

        if current_group:
            groups.append(current_group)

        return groups


    def _find_closest_vna(self, vna_df: pd.DataFrame, target_datetime: datetime, 
                         max_minutes: int = 5) -> Optional[Dict]:
        """Find VNA measurement closest to target datetime"""
        time_diffs = abs(vna_df['measurement_datetime'] - target_datetime)
        min_diff_idx = time_diffs.idxmin()
        min_diff = time_diffs.loc[min_diff_idx]

        if min_diff <= timedelta(minutes=max_minutes):
            return vna_df.loc[min_diff_idx].to_dict()

        return None


    def _extract_gps_component(self, gps_string, index: int):
        """Extract latitude (0) or longitude (1) from GPS string"""
        if pd.isna(gps_string):
            return None
        try:
            parts = str(gps_string).split(',')
            if len(parts) > index:
                return float(parts[index])
        except (ValueError, AttributeError):
            pass
        return None


    def _save_datasets(self, training_df: pd.DataFrame):
        """Save complete and depth-specific datasets"""

        # Save complete dataset
        complete_file = self.output_folder / "training_data_complete.csv"
        training_df.to_csv(complete_file, index=False)
        print(f"✓ Saved: {complete_file} ({len(training_df)} rows)")

        # Save depth-specific datasets
        for depth in [20, 50, 80]:
            depth_col = f'shear_{depth}cm'

            # Filter rows where this depth has data
            depth_df = training_df[training_df[depth_col].notna()].copy()

            if len(depth_df) > 0:
                depth_file = self.output_folder / f"training_data_{depth}cm.csv"
                depth_df.to_csv(depth_file, index=False)
                print(f"Saved: {depth_file} ({len(depth_df)} rows)")
            else:
                print(f"No data for {depth}cm depth")


def main():
    """Main entry point when running as a script"""
    processor = DataProcessor()
    processor.run()


if __name__ == "__main__":
    main()
