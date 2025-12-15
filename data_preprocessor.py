"""
Data Preprocessing Script for Density Sensor ML Training
=========================================================

This script matches VNA (Vector Network Analyzer) frequency measurements 
with Vane Shear test results to create training data for ML models.

The script:
1. Loads Vane Shear measurements (ground truth at 20, 50, 80 cm depths)
2. Loads VNA transfer files (frequency response measurements)
3. Matches VNA readings to Vane Shear tests by timestamp
4. Creates consolidated datasets for ML training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import re
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class VNADataProcessor:
    """Processes VNA transfer files and extracts frequency measurements"""
    
    def __init__(self, input_folder: str):
        self.input_folder = Path(input_folder)
        self.vna_data = []
        
    def parse_raw_sensor_data(self, raw_data: str) -> Optional[Dict]:
        """
        Parse the Raw Sensor Data field containing frequency measurements
        
        Args:
            raw_data: String containing frequency, ReS11, ImS11 data
            
        Returns:
            Dictionary with frequencies, real parts, and imaginary parts
        """
        if pd.isna(raw_data) or not isinstance(raw_data, str):
            return None
            
        frequencies = []
        real_s11 = []
        imag_s11 = []
        
        lines = raw_data.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Skip header lines and comments
            if not line or line.startswith('!') or line.startswith('#'):
                continue
                
            # Parse frequency and S11 parameters
            parts = line.split()
            if len(parts) == 3:
                try:
                    freq = float(parts[0])
                    re_s11 = float(parts[1])
                    im_s11 = float(parts[2])
                    
                    frequencies.append(freq)
                    real_s11.append(re_s11)
                    imag_s11.append(im_s11)
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
    
    def load_transfer_files(self) -> pd.DataFrame:
        """
        Load all transfer*.csv files from the input folder
        
        Returns:
            DataFrame with VNA measurements
        """
        transfer_files = sorted(self.input_folder.glob('transfer_*.csv'))
        
        if not transfer_files:
            raise FileNotFoundError(f"No transfer_*.csv files found in {self.input_folder}")
        
        print(f"Found {len(transfer_files)} transfer files")
        
        all_vna_data = []
        
        for file_path in transfer_files:
            print(f"Loading {file_path.name}...")
            
            df = pd.read_csv(file_path)
            
            # Parse measurement_date (already in ISO format with seconds)
            df['measurement_datetime'] = pd.to_datetime(df['measurement_date'])
            
            # Remove seconds for matching (as Vane Shear only has minute precision)
            df['measurement_datetime_rounded'] = df['measurement_datetime'].dt.floor('min')
            
            # Parse Raw Sensor Data for each row
            for idx, row in df.iterrows():
                sensor_data = self.parse_raw_sensor_data(row['Raw Sensor Data'])
                
                if sensor_data is not None:
                    vna_record = {
                        'measurement_datetime': row['measurement_datetime'],
                        'measurement_datetime_rounded': row['measurement_datetime_rounded'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'distance_to_ground': row['DistanceToGround'],
                        **sensor_data
                    }
                    all_vna_data.append(vna_record)
        
        print(f"Loaded {len(all_vna_data)} VNA measurements")
        
        self.vna_data = all_vna_data
        return pd.DataFrame(all_vna_data)


class VaneShearProcessor:
    """Processes Vane Shear test results"""
    
    def __init__(self, input_folder: str, site_filter: str = "zz Brisbane"):
        self.input_folder = Path(input_folder)
        self.site_filter = site_filter
        
    def load_vane_shear_data(self) -> pd.DataFrame:
        """
        Load Vane Shear Report CSV and filter by site
        
        Returns:
            DataFrame with Vane Shear measurements
        """
        vane_shear_file = self.input_folder / "Vane Shear Report.csv"
        
        if not vane_shear_file.exists():
            raise FileNotFoundError(f"Vane Shear Report.csv not found in {self.input_folder}")
        
        print(f"Loading {vane_shear_file.name}...")
        
        df = pd.read_csv(vane_shear_file)
        
        # Filter by site
        if self.site_filter:
            df = df[df['Site'] == self.site_filter].copy()
            print(f"Filtered to site '{self.site_filter}': {len(df)} measurements")
        
        # Parse DateTime (format: 2025-05-27T01:26:00Z)
        df['datetime'] = pd.to_datetime(df['DateTime'])
        
        # Remove seconds for matching
        df['datetime_rounded'] = df['datetime'].dt.floor('min')
        
        # Parse Created timestamp for better matching
        df['created_datetime'] = pd.to_datetime(df['Created'])
        
        # Extract depth and shear strength
        df['depth_cm'] = df['Depth'].astype(int) / 10  # Convert mm to cm
        df['shear_strength'] = df['ShearStrenght'].astype(float)
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        print(f"Loaded {len(df)} Vane Shear measurements")
        print(f"Depth distribution: {df['depth_cm'].value_counts().sort_index().to_dict()}")
        
        return df


class DataMatcher:
    """Matches VNA measurements with Vane Shear tests"""
    
    def __init__(self, vna_df: pd.DataFrame, vane_shear_df: pd.DataFrame):
        self.vna_df = vna_df
        self.vane_shear_df = vane_shear_df
        
    def find_closest_vna_measurement(
        self, 
        target_datetime: datetime, 
        max_time_diff_minutes: int = 5
    ) -> Optional[Dict]:
        """
        Find the VNA measurement closest to the target datetime
        
        Args:
            target_datetime: Target timestamp to match
            max_time_diff_minutes: Maximum acceptable time difference
            
        Returns:
            VNA measurement dictionary or None
        """
        # Calculate time differences
        time_diffs = abs(self.vna_df['measurement_datetime'] - target_datetime)
        
        # Find the closest measurement within the time window
        min_diff_idx = time_diffs.idxmin()
        min_diff = time_diffs.loc[min_diff_idx]
        
        if min_diff <= timedelta(minutes=max_time_diff_minutes):
            return self.vna_df.loc[min_diff_idx].to_dict()
        
        return None
    
    def group_vane_shear_by_location(self) -> List[List[int]]:
        """
        Group Vane Shear measurements by location
        
        When measurements at 20, 50, 80 cm are taken at the same location,
        they are typically taken within 3-5 minutes of each other, with
        20cm first, then 50cm, then 80cm.
        
        Logic: Start a new group when we see a 20cm measurement after having
        seen measurements at deeper depths, or when there's a time gap > 10 minutes.
        
        Returns:
            List of groups (each group is a list of row indices)
        """
        groups = []
        current_group = []
        prev_datetime = None
        prev_depth = None
        
        for idx, row in self.vane_shear_df.iterrows():
            current_datetime = row['datetime']
            current_depth = row['depth_cm']
            
            # Start a new group if:
            # 1. This is the first measurement
            # 2. We see a 20cm measurement after seeing 50cm or 80cm (new location)
            # 3. Time gap is more than 10 minutes
            start_new_group = False
            
            if prev_datetime is None:
                # First measurement
                start_new_group = True
            elif prev_depth is not None and current_depth == 20 and prev_depth in [50, 80]:
                # We're back to 20cm after measuring deeper - new location
                start_new_group = True
            else:
                time_diff = abs((current_datetime - prev_datetime).total_seconds())
                if time_diff > 600:  # 10 minutes
                    start_new_group = True
            
            if start_new_group:
                if current_group:
                    groups.append(current_group)
                current_group = [idx]
            else:
                current_group.append(idx)
            
            prev_datetime = current_datetime
            prev_depth = current_depth
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def create_training_dataset(self) -> pd.DataFrame:
        """
        Create the matched training dataset
        
        For each location (group of depth measurements), find the corresponding
        VNA measurement and create a single row with all three depth values.
        
        Returns:
            DataFrame suitable for ML training
        """
        location_groups = self.group_vane_shear_by_location()
        
        print(f"\nFound {len(location_groups)} measurement locations")
        
        training_records = []
        matched_count = 0
        
        for group_idx, group in enumerate(location_groups):
            # Get all measurements in this group
            group_data = self.vane_shear_df.loc[group]
            
            # Use the first measurement's timestamp to find VNA data
            reference_datetime = group_data.iloc[0]['created_datetime']
            
            # Find the closest VNA measurement
            vna_measurement = self.find_closest_vna_measurement(reference_datetime)
            
            if vna_measurement is None:
                print(f"Warning: No VNA match for group {group_idx} at {reference_datetime}")
                continue
            
            # Initialize record with VNA data
            record = {
                'measurement_datetime': reference_datetime,
                'vna_datetime': vna_measurement['measurement_datetime'],
                'time_diff_seconds': abs((reference_datetime - vna_measurement['measurement_datetime']).total_seconds()),
                'latitude': group_data.iloc[0].get('Default GPS', '').split(',')[0] if pd.notna(group_data.iloc[0].get('Default GPS')) else None,
                'longitude': group_data.iloc[0].get('Default GPS', '').split(',')[1] if pd.notna(group_data.iloc[0].get('Default GPS')) and ',' in str(group_data.iloc[0].get('Default GPS')) else None,
                'vna_latitude': vna_measurement['latitude'],
                'vna_longitude': vna_measurement['longitude'],
            }
            
            # Add frequency data
            frequencies = vna_measurement['frequencies']
            real_s11 = vna_measurement['real_s11']
            imag_s11 = vna_measurement['imag_s11']
            
            # Store frequency values as separate columns
            for i, (freq, re_val, im_val) in enumerate(zip(frequencies, real_s11, imag_s11)):
                freq_mhz = int(freq / 1e6)  # Convert to MHz
                record[f'freq_{freq_mhz}_real'] = re_val
                record[f'freq_{freq_mhz}_imag'] = im_val
                record[f'freq_{freq_mhz}_magnitude'] = np.sqrt(re_val**2 + im_val**2)
                record[f'freq_{freq_mhz}_phase'] = np.arctan2(im_val, re_val)
            
            # Initialize depth columns as None
            record['shear_20cm'] = None
            record['shear_50cm'] = None
            record['shear_80cm'] = None
            
            # Fill in the depth values we have
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
        
        print(f"\nSuccessfully matched {matched_count} measurement locations")
        
        df = pd.DataFrame(training_records)
        
        # Report on data completeness
        print(f"\nShear strength data completeness:")
        print(f"  20cm: {df['shear_20cm'].notna().sum()} / {len(df)} ({100*df['shear_20cm'].notna().sum()/len(df):.1f}%)")
        print(f"  50cm: {df['shear_50cm'].notna().sum()} / {len(df)} ({100*df['shear_50cm'].notna().sum()/len(df):.1f}%)")
        print(f"  80cm: {df['shear_80cm'].notna().sum()} / {len(df)} ({100*df['shear_80cm'].notna().sum()/len(df):.1f}%)")
        
        return df


def main():
    """Main execution function"""
    
    # Configuration
    INPUT_FOLDER = "Input"
    OUTPUT_FOLDER = "Output"
    SITE_FILTER = "zz Brisbane"
    
    # Create output folder
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(exist_ok=True)
    
    print("="*70)
    print("Density Sensor Data Preprocessing")
    print("="*70)
    
    # Step 1: Load VNA data
    print("\n[Step 1] Loading VNA transfer files...")
    vna_processor = VNADataProcessor(INPUT_FOLDER)
    vna_df = vna_processor.load_transfer_files()
    
    # Step 2: Load Vane Shear data
    print("\n[Step 2] Loading Vane Shear data...")
    vane_processor = VaneShearProcessor(INPUT_FOLDER, SITE_FILTER)
    vane_df = vane_processor.load_vane_shear_data()
    
    # Step 3: Match and create training dataset
    print("\n[Step 3] Matching VNA and Vane Shear measurements...")
    matcher = DataMatcher(vna_df, vane_df)
    training_df = matcher.create_training_dataset()
    
    # Step 4: Save the complete dataset
    print("\n[Step 4] Saving datasets...")
    
    # Save complete dataset
    complete_file = output_path / "training_data_complete.csv"
    training_df.to_csv(complete_file, index=False)
    print(f"Saved complete dataset: {complete_file} ({len(training_df)} rows)")
    
    # Create separate datasets for each depth (rows with non-null values)
    for depth in [20, 50, 80]:
        depth_df = training_df[training_df[f'shear_{depth}cm'].notna()].copy()
        depth_file = output_path / f"training_data_{depth}cm.csv"
        depth_df.to_csv(depth_file, index=False)
        print(f"Saved {depth}cm dataset: {depth_file} ({len(depth_df)} rows)")
    
    print("\n" + "="*70)
    print("Preprocessing complete!")
    print("="*70)
    
    # Display sample statistics
    print("\nDataset Statistics:")
    print(f"Total measurements: {len(training_df)}")
    print(f"Date range: {training_df['measurement_datetime'].min()} to {training_df['measurement_datetime'].max()}")
    print(f"Average time difference VNA-VaneShear: {training_df['time_diff_seconds'].mean():.1f} seconds")
    
    print("\nShear Strength Statistics:")
    for depth in [20, 50, 80]:
        col = f'shear_{depth}cm'
        valid_data = training_df[col].dropna()
        if len(valid_data) > 0:
            print(f"  {depth}cm: mean={valid_data.mean():.1f}, std={valid_data.std():.1f}, "
                  f"min={valid_data.min():.1f}, max={valid_data.max():.1f}")


if __name__ == "__main__":
    main()
