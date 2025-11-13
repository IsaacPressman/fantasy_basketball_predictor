"""
Prepare Training Data
Prepares data specifically for model training - train/val splits, handling missing values, scaling
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

class TrainingDataPreparator:
    """Prepare data for model training"""
    
    def __init__(self, data_dir='data/processed', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Define target columns
        self.target_cols = [
            'NEXT_PTS', 'NEXT_AST', 'NEXT_REB', 'NEXT_BLK', 
            'NEXT_STL', 'NEXT_TOV', 'NEXT_MIN', 'NEXT_FG3M',
            'NEXT_FGA', 'NEXT_FTA', 'NEXT_FG_PCT', 'NEXT_FT_PCT'
        ]
        
        # Metadata columns (not features)
        self.metadata_cols = [
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_NAME', 
            'TEAM_ABBREVIATION', 'SEASON', 'SEASON_ORDER',
            'BIRTHDATE', 'SCHOOL', 'COUNTRY', 'DRAFT_YEAR',
            'HEIGHT', 'TEAM_CITY'
        ]
    
    def load_data(self):
        """Load training and prediction datasets"""
        print("="*60)
        print("Loading Datasets")
        print("="*60)
        
        # Load training data
        train_path = os.path.join(self.data_dir, 'final_training_data.csv')
        if not os.path.exists(train_path):
            print(f"✗ Training data not found at {train_path}")
            return None, None
        
        train_df = pd.read_csv(train_path)
        print(f"✓ Loaded training data: {train_df.shape}")
        
        # Load prediction data
        pred_path = os.path.join(self.data_dir, 'final_prediction_data.csv')
        if not os.path.exists(pred_path):
            print(f"✗ Prediction data not found at {pred_path}")
            return train_df, None
        
        pred_df = pd.read_csv(pred_path)
        print(f"✓ Loaded prediction data: {pred_df.shape}")
        
        return train_df, pred_df
    
    def identify_feature_columns(self, df):
        """
        Identify which columns are features (vs targets/metadata)
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            tuple: (feature_cols, df_converted) - feature names and converted dataframe
        """
        print("\n" + "="*60)
        print("Identifying Feature Columns")
        print("="*60)
        
        # Work on a copy to avoid modifying original
        df = df.copy()
        
        # All columns that aren't targets or metadata
        feature_cols = [col for col in df.columns 
                       if col not in self.target_cols and col not in self.metadata_cols]
        
        # Check for non-numeric features
        non_numeric = []
        for col in feature_cols:
            if df[col].dtype == 'object':
                # Try to convert to numeric first
                if col in ['DRAFT_ROUND', 'DRAFT_NUMBER']:
                    # These should be numeric, try converting
                    print(f"\nConverting {col} to numeric...")
                    
                    if col == 'DRAFT_ROUND':
                        # Undrafted = 0, otherwise convert to int
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    elif col == 'DRAFT_NUMBER':
                        # Undrafted = 0
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                    print(f"✓ Converted {col} to numeric")
                else:
                    non_numeric.append(col)
        
        if non_numeric:
            print(f"\n⚠ Found {len(non_numeric)} non-numeric feature columns:")
            for col in non_numeric[:10]:
                print(f"  {col}")
            if len(non_numeric) > 10:
                print(f"  ... and {len(non_numeric) - 10} more")
            
            # Remove non-numeric columns (like POSITION string)
            feature_cols = [col for col in feature_cols if col not in non_numeric]
            print(f"\n✓ Removed non-numeric columns")
        
        print(f"\nTotal columns: {len(df.columns)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Targets: {len([c for c in self.target_cols if c in df.columns])}")
        print(f"  Metadata: {len([c for c in self.metadata_cols if c in df.columns])}")
        
        return feature_cols, df
    
    def create_train_val_split(self, train_df, feature_cols):
        """
        Split training data into train and validation sets BY SEASON
        Training: 2020-21, 2021-22, 2022-23
        Validation: 2023-24 (most recent with targets)
        
        Args:
            train_df (DataFrame): Training dataset
            feature_cols (list): Feature column names
            
        Returns:
            tuple: (X_train, X_val, y_train, y_val, metadata_train, metadata_val)
        """
        print("\n" + "="*60)
        print("Creating Train/Validation Split (Time-Based)")
        print("="*60)
        
        if 'SEASON' not in train_df.columns:
            print("⚠ Warning: SEASON column not found.")
            return None, None, None, None, None, None
        
        # Time-based split by season
        print("\nSplit strategy:")
        print("  Training: 2020-21, 2021-22, 2022-23")
        print("  Validation: 2023-24")
        
        # Training set: Earlier seasons
        train_seasons = ['2020-21', '2021-22', '2022-23']
        val_season = '2023-24'
        
        train_mask = train_df['SEASON'].isin(train_seasons)
        val_mask = train_df['SEASON'] == val_season
        
        # Separate features, targets, and metadata
        X_train = train_df.loc[train_mask, feature_cols].copy()
        X_val = train_df.loc[val_mask, feature_cols].copy()
        
        # Get targets
        available_targets = [col for col in self.target_cols if col in train_df.columns]
        y_train = train_df.loc[train_mask, available_targets].copy()
        y_val = train_df.loc[val_mask, available_targets].copy()
        
        # Get metadata
        available_metadata = [col for col in self.metadata_cols if col in train_df.columns]
        if available_metadata:
            metadata_train = train_df.loc[train_mask, available_metadata].copy()
            metadata_val = train_df.loc[val_mask, available_metadata].copy()
        else:
            metadata_train = pd.DataFrame(index=X_train.index)
            metadata_val = pd.DataFrame(index=X_val.index)
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"  Seasons: {sorted(metadata_train['SEASON'].unique()) if 'SEASON' in metadata_train.columns else 'N/A'}")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"  Seasons: {sorted(metadata_val['SEASON'].unique()) if 'SEASON' in metadata_val.columns else 'N/A'}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Targets: {y_train.shape[1]}")
        
        # Report target distributions
        print(f"\nTarget distributions (training set):")
        for target in available_targets[:6]:
            print(f"  {target:15s}: mean={y_train[target].mean():.2f}, std={y_train[target].std():.2f}")
        
        return X_train, X_val, y_train, y_val, metadata_train, metadata_val
    
    def prepare_prediction_data(self, pred_df, feature_cols):
        """
        Prepare prediction dataset
        
        Args:
            pred_df (DataFrame): Prediction dataset
            feature_cols (list): Feature column names
            
        Returns:
            tuple: (X_pred, metadata_pred)
        """
        print("\n" + "="*60)
        print("Preparing Prediction Data")
        print("="*60)
        
        # Get features
        X_pred = pred_df[feature_cols].copy()
        
        # Get metadata
        available_metadata = [col for col in self.metadata_cols if col in pred_df.columns]
        metadata_pred = pred_df[available_metadata].copy() if available_metadata else pd.DataFrame(index=pred_df.index)
        
        print(f"\nPrediction set: {X_pred.shape[0]} samples")
        print(f"Features: {X_pred.shape[1]}")
        
        return X_pred, metadata_pred
    
    def handle_missing_values(self, X_train, X_val, X_pred=None):
        """
        Handle missing values in features
        
        Args:
            X_train (DataFrame): Training features
            X_val (DataFrame): Validation features
            X_pred (DataFrame): Prediction features (optional)
            
        Returns:
            tuple: (X_train_filled, X_val_filled, X_pred_filled, fill_values)
        """
        print("\n" + "="*60)
        print("Handling Missing Values")
        print("="*60)
        
        # Calculate fill values from training data only
        fill_values = {}
        
        for col in X_train.columns:
            if X_train[col].isnull().any():
                # For numeric columns, use median
                if X_train[col].dtype in ['float64', 'int64']:
                    fill_values[col] = X_train[col].median()
                else:
                    fill_values[col] = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 0
        
        print(f"\nColumns with missing values: {len(fill_values)}")
        
        # Fill missing values
        X_train_filled = X_train.fillna(fill_values)
        X_val_filled = X_val.fillna(fill_values)
        
        X_pred_filled = None
        if X_pred is not None:
            X_pred_filled = X_pred.fillna(fill_values)
        
        # Report remaining missing (should be very few or zero)
        train_missing = X_train_filled.isnull().sum().sum()
        val_missing = X_val_filled.isnull().sum().sum()
        
        print(f"\nRemaining missing values:")
        print(f"  Training: {train_missing}")
        print(f"  Validation: {val_missing}")
        
        if train_missing > 0 or val_missing > 0:
            print("\n⚠ Warning: Some missing values remain, filling with 0")
            X_train_filled = X_train_filled.fillna(0)
            X_val_filled = X_val_filled.fillna(0)
            if X_pred_filled is not None:
                X_pred_filled = X_pred_filled.fillna(0)
        else:
            print("\n✓ All missing values handled")
        
        return X_train_filled, X_val_filled, X_pred_filled, fill_values
    
    def scale_features(self, X_train, X_val, X_pred=None):
        """
        Scale features using StandardScaler
        
        Args:
            X_train (DataFrame): Training features
            X_val (DataFrame): Validation features
            X_pred (DataFrame): Prediction features (optional)
            
        Returns:
            tuple: (X_train_scaled, X_val_scaled, X_pred_scaled, scaler)
        """
        print("\n" + "="*60)
        print("Scaling Features")
        print("="*60)
        
        # Note: Tree-based models (XGBoost/LightGBM) don't require scaling
        # But we'll do it anyway for flexibility
        
        print("\nUsing StandardScaler (mean=0, std=1)")
        print("Note: Optional for tree-based models, but good for consistency")
        
        scaler = StandardScaler()
        
        # Fit on training data only
        scaler.fit(X_train)
        
        # Transform all sets
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        X_pred_scaled = None
        if X_pred is not None:
            X_pred_scaled = pd.DataFrame(
                scaler.transform(X_pred),
                columns=X_pred.columns,
                index=X_pred.index
            )
        
        print(f"\n✓ Features scaled")
        print(f"  Training mean: {X_train_scaled.mean().mean():.4f}")
        print(f"  Training std: {X_train_scaled.std().mean():.4f}")
        
        return X_train_scaled, X_val_scaled, X_pred_scaled, scaler
    
    def save_prepared_data(self, X_train, X_val, X_pred, y_train, y_val, 
                          metadata_train, metadata_val, metadata_pred, fill_values):
        """Save all prepared datasets"""
        
        print("\n" + "="*60)
        print("Saving Prepared Data")
        print("="*60)
        
        # Create directory for prepared data
        prepared_dir = os.path.join(self.data_dir, 'prepared')
        os.makedirs(prepared_dir, exist_ok=True)
        
        # Save features
        X_train.to_csv(os.path.join(prepared_dir, 'X_train.csv'), index=False)
        X_val.to_csv(os.path.join(prepared_dir, 'X_val.csv'), index=False)
        if X_pred is not None:
            X_pred.to_csv(os.path.join(prepared_dir, 'X_pred.csv'), index=False)
        
        # Save targets
        y_train.to_csv(os.path.join(prepared_dir, 'y_train.csv'), index=False)
        y_val.to_csv(os.path.join(prepared_dir, 'y_val.csv'), index=False)
        
        # Save metadata
        metadata_train.to_csv(os.path.join(prepared_dir, 'metadata_train.csv'), index=False)
        metadata_val.to_csv(os.path.join(prepared_dir, 'metadata_val.csv'), index=False)
        if metadata_pred is not None:
            metadata_pred.to_csv(os.path.join(prepared_dir, 'metadata_pred.csv'), index=False)
        
        # Save fill values for missing data
        fill_values_df = pd.DataFrame({
            'feature': list(fill_values.keys()),
            'fill_value': list(fill_values.values())
        })
        fill_values_df.to_csv(os.path.join(prepared_dir, 'fill_values.csv'), index=False)
        
        # Save feature names
        feature_names = pd.DataFrame({'feature': X_train.columns})
        feature_names.to_csv(os.path.join(prepared_dir, 'feature_names.csv'), index=False)
        
        print(f"\n✓ Saved all prepared data to: {prepared_dir}")
        print(f"  - X_train.csv, X_val.csv, X_pred.csv")
        print(f"  - y_train.csv, y_val.csv")
        print(f"  - metadata files")
        print(f"  - fill_values.csv")
        print(f"  - feature_names.csv")
        
        return prepared_dir
    
    def generate_summary(self, X_train, X_val, X_pred, y_train):
        """
        Generate summary of prepared data
        
        Args:
            X_train, X_val, X_pred: Feature DataFrames
            y_train: Target DataFrame
        """
        print("\n" + "="*60)
        print("DATA PREPARATION SUMMARY")
        print("="*60)
        
        print(f"\nDataset Sizes:")
        print(f"  Training: {X_train.shape[0]} samples × {X_train.shape[1]} features")
        print(f"  Validation: {X_val.shape[0]} samples × {X_val.shape[1]} features")
        if X_pred is not None:
            print(f"  Prediction: {X_pred.shape[0]} samples × {X_pred.shape[1]} features")
        
        print(f"\nTarget Variables: {y_train.shape[1]}")
        print(f"  {', '.join(y_train.columns[:6])}...")
        
        # Only calculate statistics on numeric columns
        numeric_train = X_train.select_dtypes(include=[np.number])
        numeric_val = X_val.select_dtypes(include=[np.number])
        
        print(f"\nFeature Statistics:")
        print(f"  Numeric features: {numeric_train.shape[1]}/{X_train.shape[1]}")
        print(f"  Mean absolute value: {numeric_train.abs().mean().mean():.2f}")
        print(f"  Std dev: {numeric_train.std().mean():.2f}")
        
        print(f"\nMissing Values:")
        print(f"  Training features: {X_train.isnull().sum().sum()}")
        print(f"  Validation features: {X_val.isnull().sum().sum()}")
        print(f"  Training targets: {y_train.isnull().sum().sum()}")
        
        # Check for any infinite values
        inf_train = np.isinf(numeric_train).sum().sum()
        inf_val = np.isinf(numeric_val).sum().sum()
        
        print(f"\nInfinite Values:")
        print(f"  Training: {inf_train}")
        print(f"  Validation: {inf_val}")
        
        if inf_train > 0 or inf_val > 0:
            print("\n⚠ Warning: Infinite values detected")
        
        print("\n" + "="*60)
        print("✓ Data preparation complete!")
        print("="*60)
        print("\nReady for model training!")


def main():
    """Main execution function"""
    
    print("="*60)
    print("NBA Projections - Training Data Preparation")
    print("="*60)
    
    # Initialize preparator
    preparator = TrainingDataPreparator()
    
    # Load data
    train_df, pred_df = preparator.load_data()
    
    if train_df is None:
        print("\n✗ Failed to load data")
        return None
    
    # Identify feature columns and convert data types
    feature_cols, train_df = preparator.identify_feature_columns(train_df)
    
    # Also convert prediction data if it exists
    if pred_df is not None:
        _, pred_df = preparator.identify_feature_columns(pred_df)
    
    # Create train/validation split (time-based)
    X_train, X_val, y_train, y_val, metadata_train, metadata_val = \
        preparator.create_train_val_split(train_df, feature_cols)
    
    if X_train is None:
        print("\n✗ Failed to create train/val split")
        return None
    
    # Prepare prediction data
    X_pred, metadata_pred = None, None
    if pred_df is not None:
        X_pred, metadata_pred = preparator.prepare_prediction_data(pred_df, feature_cols)
    
    # Handle missing values
    X_train, X_val, X_pred, fill_values = \
        preparator.handle_missing_values(X_train, X_val, X_pred)
    
    # Replace any infinite values with 0
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_val = X_val.replace([np.inf, -np.inf], 0)
    if X_pred is not None:
        X_pred = X_pred.replace([np.inf, -np.inf], 0)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_pred_scaled, scaler = \
        preparator.scale_features(X_train, X_val, X_pred)
    
    # Generate summary
    preparator.generate_summary(X_train_scaled, X_val_scaled, X_pred_scaled, y_train)
    
    # Save prepared data (scaled versions)
    prepared_dir = preparator.save_prepared_data(
        X_train_scaled, X_val_scaled, X_pred_scaled, y_train, y_val,
        metadata_train, metadata_val, metadata_pred, fill_values
    )
    
    # Save the scaler
    scaler_path = os.path.join(preparator.models_dir, 'feature_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"\n✓ Saved feature scaler to: {scaler_path}")
    
    print("\n" + "="*60)
    print("✓ STEP 15 COMPLETE!")
    print("="*60)
    print("\nNext: Step 16 - Baseline Models")
    
    return X_train_scaled, X_val_scaled, X_pred_scaled, y_train, y_val


if __name__ == "__main__":
    # Run the main function
    result = main()