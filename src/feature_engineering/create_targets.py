"""
Create Target Variables
Defines what we're predicting (next season's stats) for each player
"""

import pandas as pd
import os

class TargetCreator:
    """Create target variables for ML prediction"""
    
    def __init__(self, data_dir='data/processed'):
        self.data_dir = data_dir
        
        # Define which stats we want to predict
        self.target_stats = [
            'PTS',      # Points per game
            'AST',      # Assists per game
            'REB',      # Rebounds per game
            'BLK',      # Blocks per game
            'STL',      # Steals per game
            'TOV',      # Turnovers per game
            'MIN',      # Minutes per game
            'FG3M',     # 3-pointers made per game
            'FGA',      # Field goal attempts per game
            'FTA',      # Free throw attempts per game
            'FG_PCT',   # Field goal percentage
            'FT_PCT'    # Free throw percentage
        ]
    
    def load_master_dataset(self, filename='master_dataset.csv'):
        """Load the master dataset"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"✗ Master dataset not found at {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        print(f"✓ Loaded master dataset: {df.shape}")
        return df
    
    def create_target_variables(self, df):
        """
        Create target variables (next season's stats) for each player
        
        Args:
            df (DataFrame): Master dataset
            
        Returns:
            DataFrame: Dataset with target variables added
        """
        print("\n" + "="*60)
        print("Creating Target Variables")
        print("="*60)
        
        # Sort by player and season to ensure correct ordering
        df = df.sort_values(['PLAYER_ID', 'SEASON']).copy()
        
        # Create season order (for shifting)
        season_order = {
            '2020-21': 1,
            '2021-22': 2,
            '2022-23': 3,
            '2023-24': 4,
            '2024-25': 5
        }
        df['SEASON_ORDER'] = df['SEASON'].map(season_order)
        
        print(f"\nCreating targets for {len(self.target_stats)} statistics:")
        for stat in self.target_stats:
            print(f"  - {stat}")
        
        # For each target stat, create NEXT_SEASON_{STAT} column
        for stat in self.target_stats:
            if stat in df.columns:
                # Shift stats up by one season for each player (next season = target)
                df[f'NEXT_{stat}'] = df.groupby('PLAYER_ID')[stat].shift(-1)
                
        print(f"\n✓ Created {len(self.target_stats)} target variables")
        
        # Show how many records have targets (vs. last season which has no targets)
        target_cols = [f'NEXT_{stat}' for stat in self.target_stats]
        records_with_targets = df[target_cols[0]].notna().sum()
        records_without_targets = df[target_cols[0]].isna().sum()
        
        print(f"\nRecords with targets (can be used for training): {records_with_targets}")
        print(f"Records without targets (most recent season): {records_without_targets}")
        
        return df
    
    def validate_targets(self, df):
        """
        Validate that targets make sense
        
        Args:
            df (DataFrame): Dataset with targets
        """
        print("\n" + "="*60)
        print("Validating Target Variables")
        print("="*60)
        
        # Check for any obvious issues
        target_cols = [f'NEXT_{stat}' for stat in self.target_stats]
        
        print("\nTarget variable statistics:")
        print(df[target_cols].describe().round(2))
        
        # Check for unrealistic values
        print("\nChecking for unrealistic target values:")
        
        issues_found = False
        
        # Points should be 0-50 per game
        if 'NEXT_PTS' in df.columns:
            outliers = df[(df['NEXT_PTS'] > 50) | (df['NEXT_PTS'] < 0)]
            if len(outliers) > 0:
                print(f"  ⚠ Found {len(outliers)} records with unusual NEXT_PTS")
                issues_found = True
        
        # Minutes should be 0-48 per game
        if 'NEXT_MIN' in df.columns:
            outliers = df[(df['NEXT_MIN'] > 48) | (df['NEXT_MIN'] < 0)]
            if len(outliers) > 0:
                print(f"  ⚠ Found {len(outliers)} records with unusual NEXT_MIN")
                issues_found = True
        
        # Percentages should be 0-1 (or 0-100)
        if 'NEXT_FG_PCT' in df.columns:
            outliers = df[(df['NEXT_FG_PCT'] > 1) | (df['NEXT_FG_PCT'] < 0)]
            if len(outliers) > 0:
                print(f"  ⚠ Found {len(outliers)} records with unusual NEXT_FG_PCT")
                issues_found = True
        
        if not issues_found:
            print("  ✓ All target values look reasonable")
        
        # Show example: player's current season stats vs next season targets
        print("\nExample - Player's stats vs their targets (next season):")
        
        sample_player = df[df['NEXT_PTS'].notna()].iloc[0]
        
        print(f"\nPlayer: {sample_player.get('PLAYER_NAME', 'Unknown')}")
        print(f"Season: {sample_player['SEASON']}")
        print(f"\nCurrent Season → Target (Next Season):")
        
        for stat in self.target_stats:
            if stat in df.columns and f'NEXT_{stat}' in df.columns:
                current = sample_player[stat]
                target = sample_player[f'NEXT_{stat}']
                print(f"  {stat:8s}: {current:6.2f} → {target:6.2f}")
    
    def prepare_training_data(self, df):
        """
        Prepare data for model training by removing records without targets
        
        Args:
            df (DataFrame): Dataset with targets
            
        Returns:
            tuple: (training_df, prediction_df)
                - training_df: Records with targets (can train on these)
                - prediction_df: Records without targets (most recent season, predict these)
        """
        print("\n" + "="*60)
        print("Preparing Training and Prediction Sets")
        print("="*60)
        
        target_cols = [f'NEXT_{stat}' for stat in self.target_stats]
        
        # Training data: has target variables (not most recent season)
        training_df = df[df[target_cols[0]].notna()].copy()
        
        # Prediction data: doesn't have targets (most recent season - what we want to predict)
        prediction_df = df[df[target_cols[0]].isna()].copy()
        
        print(f"\nTraining set: {len(training_df)} records")
        print(f"  Seasons: {sorted(training_df['SEASON'].unique())}")
        print(f"  Unique players: {training_df['PLAYER_ID'].nunique()}")
        
        print(f"\nPrediction set (2024-25): {len(prediction_df)} records")
        print(f"  Seasons: {sorted(prediction_df['SEASON'].unique())}")
        print(f"  Unique players: {prediction_df['PLAYER_ID'].nunique()}")
        
        return training_df, prediction_df
    
    def save_datasets(self, df_with_targets, training_df, prediction_df):
        """Save all datasets"""
        
        # Save full dataset with targets
        full_path = os.path.join(self.data_dir, 'dataset_with_targets.csv')
        df_with_targets.to_csv(full_path, index=False)
        print(f"\n✓ Saved full dataset to: {full_path}")
        
        # Save training dataset
        train_path = os.path.join(self.data_dir, 'training_dataset.csv')
        training_df.to_csv(train_path, index=False)
        print(f"✓ Saved training dataset to: {train_path}")
        
        # Save prediction dataset
        pred_path = os.path.join(self.data_dir, 'prediction_dataset.csv')
        prediction_df.to_csv(pred_path, index=False)
        print(f"✓ Saved prediction dataset to: {pred_path}")
        
        return full_path, train_path, pred_path
    
    def generate_target_summary(self, training_df):
        """
        Generate summary statistics for targets
        
        Args:
            training_df (DataFrame): Training dataset
        """
        print("\n" + "="*60)
        print("TARGET VARIABLE SUMMARY")
        print("="*60)
        
        target_cols = [f'NEXT_{stat}' for stat in self.target_stats]
        
        print("\nTarget distributions (training data):")
        summary = training_df[target_cols].describe().round(2)
        print(summary)
        
        print("\nTarget correlations (top 5 most correlated):")
        corr_matrix = training_df[target_cols].corr()
        
        # Get top correlations (excluding diagonal)
        correlations = []
        for i in range(len(target_cols)):
            for j in range(i+1, len(target_cols)):
                correlations.append({
                    'stat1': target_cols[i],
                    'stat2': target_cols[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
        
        correlations_df = pd.DataFrame(correlations)
        correlations_df = correlations_df.sort_values('correlation', ascending=False)
        print(correlations_df.head(5).to_string(index=False))


def main():
    """Main execution function"""
    
    print("="*60)
    print("NBA Target Variable Creation")
    print("="*60)
    
    # Initialize creator
    creator = TargetCreator()
    
    # Load master dataset
    df = creator.load_master_dataset()
    
    if df is None:
        print("\n✗ Failed to load master dataset")
        return None
    
    # Create target variables
    df_with_targets = creator.create_target_variables(df)
    
    # Validate targets
    creator.validate_targets(df_with_targets)
    
    # Prepare training and prediction sets
    training_df, prediction_df = creator.prepare_training_data(df_with_targets)
    
    # Generate summary
    creator.generate_target_summary(training_df)
    
    # Save all datasets
    creator.save_datasets(df_with_targets, training_df, prediction_df)
    
    print("\n" + "="*60)
    print("✓ Target variable creation complete!")
    print("="*60)
    
    return df_with_targets, training_df, prediction_df


if __name__ == "__main__":
    # Run the main function
    df_targets, df_train, df_pred = main()