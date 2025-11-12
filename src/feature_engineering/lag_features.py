"""
Create Lag Features
Creates features based on player's historical performance (last 1-3 seasons)
"""

import pandas as pd
import numpy as np
import os

class LagFeatureCreator:
    """Create lag features from historical player performance"""
    
    def __init__(self, data_dir='data/processed'):
        self.data_dir = data_dir
        
        # Stats to create lag features for
        self.lag_stats = [
            'PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV',
            'MIN', 'FG3M', 'FGA', 'FTA', 'FG_PCT', 'FT_PCT',
            'FGM', 'FTM', 'FG3A',  # Additional shooting stats
            'GP'  # Games played (availability indicator)
        ]
        
        # Advanced stats to create lags for
        self.advanced_lag_stats = [
            'ADV_USG_PCT', 'ADV_TS_PCT', 'ADV_AST_PCT', 
            'ADV_REB_PCT', 'ADV_PIE'
        ]
        
        # Number of seasons to look back
        self.lag_periods = [1, 2, 3]  # Last season, 2 seasons ago, 3 seasons ago
    
    def load_dataset(self, filename='dataset_with_targets.csv'):
        """Load the dataset with targets"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"✗ Dataset not found at {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        print(f"✓ Loaded dataset: {df.shape}")
        return df
    
    def create_lag_features(self, df):
        """
        Create lag features for each player
        
        Args:
            df (DataFrame): Dataset with targets
            
        Returns:
            DataFrame: Dataset with lag features added
        """
        print("\n" + "="*60)
        print("Creating Lag Features")
        print("="*60)
        
        # Sort by player and season
        df = df.sort_values(['PLAYER_ID', 'SEASON']).copy()
        
        # Create season order for proper shifting
        season_order = {
            '2020-21': 1,
            '2021-22': 2,
            '2022-23': 3,
            '2023-24': 4,
            '2024-25': 5
        }
        df['SEASON_ORDER'] = df['SEASON'].map(season_order)
        
        # Combine all stats to lag
        all_lag_stats = self.lag_stats + self.advanced_lag_stats
        
        # Filter to only stats that exist in the dataframe
        existing_stats = [stat for stat in all_lag_stats if stat in df.columns]
        
        print(f"\nCreating lag features for {len(existing_stats)} statistics:")
        print(f"Looking back: {self.lag_periods} seasons")
        print(f"Total lag features to create: {len(existing_stats) * len(self.lag_periods)}")
        
        # Create lag features for each stat
        for stat in existing_stats:
            for lag in self.lag_periods:
                # Create lagged column name
                lag_col = f'{stat}_LAG{lag}'
                
                # Shift stat by lag periods for each player
                df[lag_col] = df.groupby('PLAYER_ID')[stat].shift(lag)
        
        print(f"\n✓ Created {len(existing_stats) * len(self.lag_periods)} lag features")
        
        return df
    
    def create_rolling_averages(self, df):
        """
        Create rolling average features (e.g., 2-year average, 3-year average)
        
        Args:
            df (DataFrame): Dataset with lag features
            
        Returns:
            DataFrame: Dataset with rolling averages added
        """
        print("\n" + "="*60)
        print("Creating Rolling Average Features")
        print("="*60)
        
        # Stats to create rolling averages for (main counting stats)
        rolling_stats = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'MIN']
        
        # Filter to existing stats
        rolling_stats = [stat for stat in rolling_stats if stat in df.columns]
        
        print(f"\nCreating rolling averages for {len(rolling_stats)} statistics:")
        
        for stat in rolling_stats:
            # 2-year average (last 2 seasons)
            if f'{stat}_LAG1' in df.columns and f'{stat}_LAG2' in df.columns:
                df[f'{stat}_AVG2'] = (df[f'{stat}_LAG1'] + df[f'{stat}_LAG2']) / 2
            
            # 3-year average (last 3 seasons)
            if f'{stat}_LAG1' in df.columns and f'{stat}_LAG2' in df.columns and f'{stat}_LAG3' in df.columns:
                df[f'{stat}_AVG3'] = (df[f'{stat}_LAG1'] + df[f'{stat}_LAG2'] + df[f'{stat}_LAG3']) / 3
            
            # Weighted average (more weight on recent seasons)
            # Weight: 50% last season, 30% 2 seasons ago, 20% 3 seasons ago
            if f'{stat}_LAG1' in df.columns and f'{stat}_LAG2' in df.columns and f'{stat}_LAG3' in df.columns:
                df[f'{stat}_WAVG'] = (
                    0.5 * df[f'{stat}_LAG1'] + 
                    0.3 * df[f'{stat}_LAG2'] + 
                    0.2 * df[f'{stat}_LAG3']
                )
        
        print(f"✓ Created rolling averages (2-year, 3-year, weighted)")
        
        return df
    
    def create_trend_features(self, df):
        """
        Create trend features (is player improving or declining?)
        
        Args:
            df (DataFrame): Dataset with lag features
            
        Returns:
            DataFrame: Dataset with trend features added
        """
        print("\n" + "="*60)
        print("Creating Trend Features")
        print("="*60)
        
        # Stats to create trends for
        trend_stats = ['PTS', 'AST', 'REB', 'MIN', 'FG_PCT', 'ADV_TS_PCT']
        
        # Filter to existing stats
        trend_stats = [stat for stat in trend_stats if stat in df.columns]
        
        print(f"\nCreating trend features for {len(trend_stats)} statistics:")
        
        for stat in trend_stats:
            # Year-over-year change (this year vs last year)
            if f'{stat}_LAG1' in df.columns:
                df[f'{stat}_YOY_CHANGE'] = df[f'{stat}_LAG1'] - df[f'{stat}_LAG2']
            
            # 2-year trend (slope over last 3 seasons)
            if f'{stat}_LAG1' in df.columns and f'{stat}_LAG2' in df.columns and f'{stat}_LAG3' in df.columns:
                # Simple trend: (recent - old) / 2
                df[f'{stat}_TREND'] = (df[f'{stat}_LAG1'] - df[f'{stat}_LAG3']) / 2
        
        print(f"✓ Created trend features (YoY change, multi-year trend)")
        
        return df
    
    def create_consistency_features(self, df):
        """
        Create consistency features (how consistent is the player?)
        
        Args:
            df (DataFrame): Dataset with lag features
            
        Returns:
            DataFrame: Dataset with consistency features added
        """
        print("\n" + "="*60)
        print("Creating Consistency Features")
        print("="*60)
        
        # Stats to measure consistency for
        consistency_stats = ['PTS', 'MIN', 'GP']
        
        # Filter to existing stats
        consistency_stats = [stat for stat in consistency_stats if stat in df.columns]
        
        print(f"\nCreating consistency features for {len(consistency_stats)} statistics:")
        
        for stat in consistency_stats:
            # Standard deviation over last 3 seasons (lower = more consistent)
            lag_cols = [f'{stat}_LAG{i}' for i in [1, 2, 3]]
            
            if all(col in df.columns for col in lag_cols):
                df[f'{stat}_STD3'] = df[lag_cols].std(axis=1)
                
                # Coefficient of variation (std / mean) - normalized consistency measure
                mean_val = df[lag_cols].mean(axis=1)
                df[f'{stat}_CV3'] = df[f'{stat}_STD3'] / mean_val.replace(0, np.nan)
        
        print(f"✓ Created consistency features (std dev, coefficient of variation)")
        
        return df
    
    def handle_rookies_and_missing(self, df):
        """
        Handle rookies and players with missing historical data
        
        Args:
            df (DataFrame): Dataset with lag features
            
        Returns:
            DataFrame: Dataset with missing values handled
        """
        print("\n" + "="*60)
        print("Handling Rookies and Missing Data")
        print("="*60)
        
        # Create rookie indicator (no LAG1 data available)
        lag1_cols = [col for col in df.columns if '_LAG1' in col]
        if lag1_cols:
            df['IS_ROOKIE'] = df[lag1_cols[0]].isna().astype(int)
            rookie_count = df['IS_ROOKIE'].sum()
            print(f"\n✓ Identified {rookie_count} rookie records")
        
        # Create experience-based indicator
        if 'YEARS_EXPERIENCE' in df.columns:
            df['IS_YOUNG_PLAYER'] = (df['YEARS_EXPERIENCE'] <= 2).astype(int)
        
        # Count how many historical seasons available
        lag_availability = []
        for i in [1, 2, 3]:
            lag_col = f'PTS_LAG{i}'
            if lag_col in df.columns:
                lag_availability.append(~df[lag_col].isna())
        
        if lag_availability:
            df['HISTORICAL_SEASONS_AVAILABLE'] = sum(lag_availability)
        
        print(f"\n✓ Created indicators for rookies and data availability")
        
        # Report missing data
        lag_cols = [col for col in df.columns if '_LAG' in col or '_AVG' in col or '_TREND' in col]
        missing_pct = (df[lag_cols].isnull().sum() / len(df) * 100).round(1)
        
        print(f"\nMissing data summary (lag features):")
        high_missing = missing_pct[missing_pct > 30].sort_values(ascending=False)
        if len(high_missing) > 0:
            print(f"  Features with >30% missing:")
            for col, pct in high_missing.head(10).items():
                print(f"    {col}: {pct}%")
        
        return df
    
    def save_dataset(self, df, filename='dataset_with_lags.csv'):
        """Save dataset with lag features"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved dataset with lag features to: {filepath}")
        return filepath
    
    def generate_summary(self, df):
        """
        Generate summary of lag features
        
        Args:
            df (DataFrame): Dataset with lag features
        """
        print("\n" + "="*60)
        print("LAG FEATURES SUMMARY")
        print("="*60)
        
        print(f"\nDataset shape: {df.shape}")
        
        # Count different types of features
        lag_cols = [col for col in df.columns if '_LAG' in col]
        avg_cols = [col for col in df.columns if '_AVG' in col or '_WAVG' in col]
        trend_cols = [col for col in df.columns if '_TREND' in col or '_YOY' in col]
        consistency_cols = [col for col in df.columns if '_STD' in col or '_CV' in col]
        
        print(f"\nFeature counts:")
        print(f"  Lag features (LAG1, LAG2, LAG3): {len(lag_cols)}")
        print(f"  Rolling averages (AVG2, AVG3, WAVG): {len(avg_cols)}")
        print(f"  Trend features (YOY, TREND): {len(trend_cols)}")
        print(f"  Consistency features (STD, CV): {len(consistency_cols)}")
        print(f"  Total new features: {len(lag_cols) + len(avg_cols) + len(trend_cols) + len(consistency_cols)}")
        
        # Show example for one player
        print("\nExample - LeBron James lag features:")
        lebron = df[df['PLAYER_NAME'].str.contains('LeBron', case=False, na=False)]
        if len(lebron) > 0:
            lebron_recent = lebron.iloc[-1]
            
            example_cols = ['SEASON', 'PTS', 'PTS_LAG1', 'PTS_LAG2', 'PTS_AVG2', 
                          'PTS_TREND', 'IS_ROOKIE']
            existing_cols = [col for col in example_cols if col in df.columns]
            
            print(f"\nSeason: {lebron_recent.get('SEASON', 'N/A')}")
            for col in existing_cols:
                if col != 'SEASON':
                    print(f"  {col:20s}: {lebron_recent.get(col, 'N/A')}")


def main():
    """Main execution function"""
    
    print("="*60)
    print("NBA Lag Feature Creation")
    print("="*60)
    
    # Initialize creator
    creator = LagFeatureCreator()
    
    # Load dataset
    df = creator.load_dataset()
    
    if df is None:
        print("\n✗ Failed to load dataset")
        return None
    
    # Create lag features
    df_with_lags = creator.create_lag_features(df)
    
    # Create rolling averages
    df_with_lags = creator.create_rolling_averages(df_with_lags)
    
    # Create trend features
    df_with_lags = creator.create_trend_features(df_with_lags)
    
    # Create consistency features
    df_with_lags = creator.create_consistency_features(df_with_lags)
    
    # Handle rookies and missing data
    df_with_lags = creator.handle_rookies_and_missing(df_with_lags)
    
    # Generate summary
    creator.generate_summary(df_with_lags)
    
    # Save dataset
    creator.save_dataset(df_with_lags)
    
    print("\n" + "="*60)
    print("✓ Lag feature creation complete!")
    print("="*60)
    
    return df_with_lags


if __name__ == "__main__":
    # Run the main function
    df = main()