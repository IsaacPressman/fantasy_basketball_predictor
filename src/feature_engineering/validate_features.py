"""
Feature Validation and Final Preparation
Validates features, removes problematic ones, and prepares final modeling dataset
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureValidator:
    """Validate and prepare features for modeling"""
    
    def __init__(self, data_dir='data/processed'):
        self.data_dir = data_dir
        
        # Columns to exclude from modeling (identifiers, not features)
        self.exclude_cols = [
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_NAME', 
            'TEAM_ABBREVIATION', 'SEASON', 'SEASON_ORDER',
            'BIRTHDATE', 'SCHOOL', 'COUNTRY', 'DRAFT_YEAR',
            'HEIGHT', 'TEAM_CITY'  # Keep HEIGHT_INCHES instead
        ]
        
        # Target columns (what we're predicting)
        self.target_cols = [
            'NEXT_PTS', 'NEXT_AST', 'NEXT_REB', 'NEXT_BLK', 
            'NEXT_STL', 'NEXT_TOV', 'NEXT_MIN', 'NEXT_FG3M',
            'NEXT_FGA', 'NEXT_FTA', 'NEXT_FG_PCT', 'NEXT_FT_PCT'
        ]
    
    def load_dataset(self, filename='modeling_dataset.csv'):
        """Load the modeling dataset"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"✗ Dataset not found at {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        print(f"✓ Loaded dataset: {df.shape}")
        return df
    
    def check_missing_values(self, df):
        """
        Analyze missing values in features
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Missing value summary
        """
        print("\n" + "="*60)
        print("Checking Missing Values")
        print("="*60)
        
        # Calculate missing percentages
        missing_summary = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum(),
            'missing_pct': (df.isnull().sum() / len(df) * 100).round(2)
        })
        
        missing_summary = missing_summary[missing_summary['missing_count'] > 0].sort_values('missing_pct', ascending=False)
        
        print(f"\nColumns with missing data: {len(missing_summary)}/{len(df.columns)}")
        
        if len(missing_summary) > 0:
            print(f"\nTop 20 columns with most missing data:")
            print(missing_summary.head(20).to_string(index=False))
            
            # Categorize by severity
            critical = missing_summary[missing_summary['missing_pct'] > 80]
            high = missing_summary[(missing_summary['missing_pct'] > 50) & (missing_summary['missing_pct'] <= 80)]
            medium = missing_summary[(missing_summary['missing_pct'] > 20) & (missing_summary['missing_pct'] <= 50)]
            
            print(f"\nMissing data severity:")
            print(f"  Critical (>80% missing): {len(critical)} columns")
            print(f"  High (50-80% missing): {len(high)} columns")
            print(f"  Medium (20-50% missing): {len(medium)} columns")
        else:
            print("\n✓ No missing values found!")
        
        return missing_summary
    
    def remove_high_missing_features(self, df, threshold=90):
        """
        Remove features with too much missing data
        
        Args:
            df (DataFrame): Dataset
            threshold (int): Maximum % missing allowed
            
        Returns:
            DataFrame: Dataset with high-missing features removed
        """
        print(f"\n" + "="*60)
        print(f"Removing Features with >{threshold}% Missing Data")
        print("="*60)
        
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > threshold].sort_values(ascending=False)
        
        if len(high_missing) > 0:
            print(f"\nRemoving {len(high_missing)} features:")
            for col, pct in high_missing.items():
                # Don't remove target columns
                if col not in self.target_cols:
                    print(f"  {col}: {pct:.1f}% missing")
            
            cols_to_drop = [col for col in high_missing.index if col not in self.target_cols]
            df = df.drop(columns=cols_to_drop)
            print(f"\n✓ Removed {len(cols_to_drop)} features")
        else:
            print(f"\n✓ No features with >{threshold}% missing data")
        
        return df
    
    def check_constant_features(self, df):
        """
        Find and remove features with zero or very low variance
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset with constant features removed
        """
        print("\n" + "="*60)
        print("Checking for Constant/Low-Variance Features")
        print("="*60)
        
        # Only check numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        constant_features = []
        low_variance_features = []
        
        for col in numeric_cols:
            if col in self.target_cols or col in self.exclude_cols:
                continue
            
            # Check if constant (only one unique value)
            n_unique = df[col].nunique()
            if n_unique == 1:
                constant_features.append(col)
            
            # Check if very low variance
            elif n_unique <= 2 and df[col].notna().sum() > 0:
                # Binary feature, check if too imbalanced
                value_counts = df[col].value_counts(normalize=True)
                if len(value_counts) > 0 and value_counts.iloc[0] > 0.99:
                    low_variance_features.append(col)
        
        print(f"\nConstant features (1 unique value): {len(constant_features)}")
        if constant_features:
            for col in constant_features[:10]:
                print(f"  {col}")
            if len(constant_features) > 10:
                print(f"  ... and {len(constant_features) - 10} more")
        
        print(f"\nLow-variance features (>99% same value): {len(low_variance_features)}")
        if low_variance_features:
            for col in low_variance_features[:10]:
                print(f"  {col}")
        
        # Remove constant and low-variance features
        features_to_remove = constant_features + low_variance_features
        if features_to_remove:
            df = df.drop(columns=features_to_remove)
            print(f"\n✓ Removed {len(features_to_remove)} constant/low-variance features")
        else:
            print("\n✓ No constant or low-variance features found")
        
        return df
    
    def check_highly_correlated_features(self, df):
        """
        Find highly correlated feature pairs
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Correlation summary
        """
        print("\n" + "="*60)
        print("Checking for Highly Correlated Features")
        print("="*60)
        
        # Get numeric features only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols 
                       if col not in self.target_cols and col not in self.exclude_cols]
        
        print(f"\nAnalyzing correlations among {len(feature_cols)} features...")
        
        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr().abs()
        
        # Find highly correlated pairs (correlation > 0.95)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
            print(f"\n⚠ Found {len(high_corr_pairs)} highly correlated pairs (>0.95):")
            print(high_corr_df.head(10).to_string(index=False))
            
            if len(high_corr_pairs) > 10:
                print(f"  ... and {len(high_corr_pairs) - 10} more pairs")
            
            print("\nNote: High correlation is OK for tree-based models (XGBoost/LightGBM)")
            print("Consider removing one from each pair if using linear models")
        else:
            print("\n✓ No highly correlated feature pairs found (>0.95)")
        
        return high_corr_df if high_corr_pairs else pd.DataFrame()
    
    def separate_features_and_targets(self, df):
        """
        Separate features from targets and identifiers
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            tuple: (features_df, targets_df, metadata_df)
        """
        print("\n" + "="*60)
        print("Separating Features, Targets, and Metadata")
        print("="*60)
        
        # Identify column types
        feature_cols = []
        target_cols = []
        metadata_cols = []
        
        for col in df.columns:
            if col in self.target_cols:
                target_cols.append(col)
            elif col in self.exclude_cols:
                metadata_cols.append(col)
            else:
                feature_cols.append(col)
        
        print(f"\nColumn breakdown:")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Targets: {len(target_cols)}")
        print(f"  Metadata: {len(metadata_cols)}")
        
        # Create separate dataframes
        features_df = df[feature_cols].copy()
        targets_df = df[target_cols].copy() if target_cols else pd.DataFrame()
        metadata_df = df[metadata_cols].copy() if metadata_cols else pd.DataFrame()
        
        return features_df, targets_df, metadata_df
    
    def validate_feature_types(self, df):
        """
        Check data types and convert if necessary
        
        Args:
            df (DataFrame): Features dataframe
            
        Returns:
            DataFrame: Dataset with corrected types
        """
        print("\n" + "="*60)
        print("Validating Feature Data Types")
        print("="*60)
        
        print(f"\nData type distribution:")
        print(df.dtypes.value_counts())
        
        # Check for object columns that should be numeric
        object_cols = df.select_dtypes(include=['object']).columns
        
        if len(object_cols) > 0:
            print(f"\n⚠ Found {len(object_cols)} object columns:")
            for col in object_cols[:10]:
                unique_vals = df[col].nunique()
                print(f"  {col}: {unique_vals} unique values")
            
            if len(object_cols) > 10:
                print(f"  ... and {len(object_cols) - 10} more")
        else:
            print("\n✓ All features are numeric")
        
        return df
    
    def create_final_datasets(self, df):
        """
        Create final train/test datasets
        
        Args:
            df (DataFrame): Complete dataset
            
        Returns:
            tuple: (train_df, test_df)
        """
        print("\n" + "="*60)
        print("Creating Final Train/Test Datasets")
        print("="*60)
        
        # Split into training (has targets) and prediction (no targets)
        has_targets = df[self.target_cols[0]].notna()
        
        train_df = df[has_targets].copy()
        test_df = df[~has_targets].copy()
        
        print(f"\nTraining dataset:")
        print(f"  Shape: {train_df.shape}")
        print(f"  Seasons: {sorted(train_df['SEASON'].unique()) if 'SEASON' in train_df.columns else 'N/A'}")
        
        print(f"\nTest/Prediction dataset:")
        print(f"  Shape: {test_df.shape}")
        print(f"  Seasons: {sorted(test_df['SEASON'].unique()) if 'SEASON' in test_df.columns else 'N/A'}")
        
        return train_df, test_df
    
    def save_final_datasets(self, train_df, test_df):
        """Save final cleaned datasets"""
        
        # Save training dataset
        train_path = os.path.join(self.data_dir, 'final_training_data.csv')
        train_df.to_csv(train_path, index=False)
        print(f"\n✓ Saved training dataset to: {train_path}")
        
        # Save test/prediction dataset
        test_path = os.path.join(self.data_dir, 'final_prediction_data.csv')
        test_df.to_csv(test_path, index=False)
        print(f"✓ Saved prediction dataset to: {test_path}")
        
        return train_path, test_path
    
    def generate_final_summary(self, train_df, test_df):
        """
        Generate final summary report
        
        Args:
            train_df (DataFrame): Training dataset
            test_df (DataFrame): Test dataset
        """
        print("\n" + "="*60)
        print("FINAL FEATURE ENGINEERING SUMMARY")
        print("="*60)
        
        # Feature counts
        feature_cols = [col for col in train_df.columns 
                       if col not in self.target_cols and col not in self.exclude_cols]
        
        print(f"\nFinal feature count: {len(feature_cols)}")
        print(f"Target variables: {len(self.target_cols)}")
        
        # Training data summary
        print(f"\n{'='*60}")
        print("TRAINING DATA")
        print(f"{'='*60}")
        print(f"Total records: {len(train_df)}")
        print(f"Unique players: {train_df['PLAYER_ID'].nunique() if 'PLAYER_ID' in train_df.columns else 'N/A'}")
        
        if 'SEASON' in train_df.columns:
            print(f"\nRecords by season:")
            print(train_df['SEASON'].value_counts().sort_index())
        
        # Check target distributions
        print(f"\nTarget variable distributions:")
        for target in self.target_cols[:6]:  # Show first 6
            if target in train_df.columns:
                print(f"  {target:15s}: mean={train_df[target].mean():.2f}, "
                      f"std={train_df[target].std():.2f}, "
                      f"min={train_df[target].min():.2f}, "
                      f"max={train_df[target].max():.2f}")
        
        # Missing data in features
        feature_missing = train_df[feature_cols].isnull().sum()
        features_with_missing = (feature_missing > 0).sum()
        print(f"\nFeatures with missing values: {features_with_missing}/{len(feature_cols)}")
        
        # Prediction data summary
        print(f"\n{'='*60}")
        print("PREDICTION DATA (2025-26)")
        print(f"{'='*60}")
        print(f"Total records: {len(test_df)}")
        print(f"Unique players: {test_df['PLAYER_ID'].nunique() if 'PLAYER_ID' in test_df.columns else 'N/A'}")
        



def main():
    """Main execution function"""
    
    print("="*60)
    print("Feature Validation and Final Preparation")
    print("="*60)
    
    # Initialize validator
    validator = FeatureValidator()
    
    # Load dataset
    df = validator.load_dataset()
    
    if df is None:
        print("\n✗ Failed to load dataset")
        return None
    
    original_shape = df.shape
    
    # Check missing values
    missing_summary = validator.check_missing_values(df)
    
    # Remove features with >90% missing
    df = validator.remove_high_missing_features(df, threshold=90)
    
    # Remove constant features
    df = validator.check_constant_features(df)
    
    # Check correlations (informational only)
    corr_summary = validator.check_highly_correlated_features(df)
    
    # Validate data types
    df = validator.validate_feature_types(df)
    
    # Create final train/test split
    train_df, test_df = validator.create_final_datasets(df)
    
    # Generate summary
    validator.generate_final_summary(train_df, test_df)
    
    # Save final datasets
    validator.save_final_datasets(train_df, test_df)
    
    print(f"\nDataset size: {original_shape} → {df.shape}")
    
    return train_df, test_df


if __name__ == "__main__":
    # Run the main function
    train_df, test_df = main()