"""
Create Availability Features
Creates features related to injury history, games played, and availability
"""

import pandas as pd
import numpy as np
import os

class AvailabilityFeatureCreator:
    """Create features related to player availability and injury patterns"""
    
    def __init__(self, data_dir='data/processed'):
        self.data_dir = data_dir
    
    def load_dataset(self, filename='modeling_dataset.csv'):
        """Load the modeling dataset"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"✗ Dataset not found at {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        print(f"✓ Loaded dataset: {df.shape}")
        return df
    
    def create_games_played_features(self, df):
        """
        Create features based on games played history
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset with GP features added
        """
        print("\n" + "="*60)
        print("Creating Games Played Features")
        print("="*60)
        
        features_created = 0
        
        if 'GP' not in df.columns:
            print("⚠ GP column not found, skipping games played features")
            return df
        
        # Games played rate (% of 82 game season)
        df['GP_RATE_CURRENT'] = df['GP'] / 82
        features_created += 1
        
        # Missed games
        df['GAMES_MISSED'] = 82 - df['GP']
        features_created += 1
        
        # Availability categories
        df['HIGH_AVAILABILITY'] = (df['GP'] >= 70).astype(int)  # 70+ games = reliable
        df['MEDIUM_AVAILABILITY'] = ((df['GP'] >= 50) & (df['GP'] < 70)).astype(int)
        df['LOW_AVAILABILITY'] = (df['GP'] < 50).astype(int)
        features_created += 3
        
        # Historical games played patterns
        if 'GP_LAG1' in df.columns:
            df['GP_LAG1_RATE'] = df['GP_LAG1'] / 82
            df['MISSED_GAMES_LAG1'] = 82 - df['GP_LAG1']
            features_created += 2
        
        if 'GP_LAG2' in df.columns:
            df['GP_LAG2_RATE'] = df['GP_LAG2'] / 82
            df['MISSED_GAMES_LAG2'] = 82 - df['GP_LAG2']
            features_created += 2
        
        if 'GP_LAG3' in df.columns:
            df['GP_LAG3_RATE'] = df['GP_LAG3'] / 82
            df['MISSED_GAMES_LAG3'] = 82 - df['GP_LAG3']
            features_created += 2
        
        print(f"✓ Created {features_created} games played features")
        
        return df
    
    def create_availability_trend_features(self, df):
        """
        Create features showing trends in availability over time
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset with availability trend features added
        """
        print("\n" + "="*60)
        print("Creating Availability Trend Features")
        print("="*60)
        
        features_created = 0
        
        # Availability trend (are they playing more or fewer games over time?)
        if 'GP_LAG1' in df.columns and 'GP_LAG2' in df.columns:
            df['GP_TREND'] = df['GP_LAG1'] - df['GP_LAG2']
            df['GP_IMPROVING'] = (df['GP_TREND'] > 5).astype(int)  # Playing 5+ more games
            df['GP_DECLINING'] = (df['GP_TREND'] < -5).astype(int)  # Playing 5+ fewer games
            features_created += 3
        
        # Average games played over last 2-3 seasons
        if 'GP_LAG1' in df.columns and 'GP_LAG2' in df.columns:
            df['GP_AVG_2YR'] = (df['GP_LAG1'] + df['GP_LAG2']) / 2
            features_created += 1
        
        if all(f'GP_LAG{i}' in df.columns for i in [1, 2, 3]):
            df['GP_AVG_3YR'] = (df['GP_LAG1'] + df['GP_LAG2'] + df['GP_LAG3']) / 3
            features_created += 1
        
        # Consistency in games played (std dev over last 3 seasons)
        if 'GP_STD3' in df.columns:
            df['GP_CONSISTENCY'] = df['GP_STD3']
            df['HIGHLY_CONSISTENT_AVAILABILITY'] = (df['GP_STD3'] < 10).astype(int)
            features_created += 2
        
        print(f"✓ Created {features_created} availability trend features")
        
        return df
    
    def create_injury_proxy_features(self, df):
        """
        Create proxy features for injury likelihood (without direct injury data)
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset with injury proxy features added
        """
        print("\n" + "="*60)
        print("Creating Injury Proxy Features")
        print("="*60)
        
        features_created = 0
        
        # Age-based injury risk (older players miss more games)
        if 'AGE' in df.columns:
            df['AGE_INJURY_RISK'] = ((df['AGE'] - 25) / 10).clip(lower=0)  # Risk increases after 25
            features_created += 1
        
        # Veteran injury risk (30+ with declining availability)
        if 'AGE' in df.columns and 'GP_LAG1' in df.columns:
            df['VETERAN_INJURY_RISK'] = ((df['AGE'] > 30) & (df['GP_LAG1'] < 65)).astype(int)
            features_created += 1
        
        # Minutes load (high minutes might increase injury risk)
        if 'MIN' in df.columns:
            df['HEAVY_MINUTES_LOAD'] = (df['MIN'] > 34).astype(int)
            features_created += 1
            
            # Minutes load over time (playing more minutes as you age = risk)
            if 'AGE' in df.columns:
                df['AGE_MINUTES_RISK'] = (df['AGE'] / 10) * (df['MIN'] / 48)
                features_created += 1
        
        # Injury-prone indicator (missed 15+ games in multiple seasons)
        if all(f'GP_LAG{i}' in df.columns for i in [1, 2, 3]):
            missed_significant = (
                (df['GP_LAG1'] < 67) & 
                (df['GP_LAG2'] < 67)
            ).astype(int)
            df['INJURY_PRONE_PATTERN'] = missed_significant
            features_created += 1
        
        # Sudden availability drop (might indicate major injury)
        if 'GP_LAG1' in df.columns and 'GP_LAG2' in df.columns:
            gp_drop = df['GP_LAG2'] - df['GP_LAG1']
            df['MAJOR_GP_DROP'] = (gp_drop > 20).astype(int)  # Missed 20+ more games
            features_created += 1
        
        # Recovery year indicator (bouncing back from low GP)
        if 'GP_LAG1' in df.columns and 'GP_LAG2' in df.columns:
            df['RECOVERY_YEAR'] = ((df['GP_LAG2'] < 50) & (df['GP_LAG1'] > 60)).astype(int)
            features_created += 1
        
        print(f"✓ Created {features_created} injury proxy features")
        
        return df
    
    def create_workload_features(self, df):
        """
        Create features related to player workload and fatigue
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset with workload features added
        """
        print("\n" + "="*60)
        print("Creating Workload Features")
        print("="*60)
        
        features_created = 0
        
        # Total minutes played (GP * MIN)
        if 'GP' in df.columns and 'MIN' in df.columns:
            df['TOTAL_MINUTES'] = df['GP'] * df['MIN']
            features_created += 1
            
            # Workload categories
            df['HIGH_WORKLOAD'] = (df['TOTAL_MINUTES'] > 2500).astype(int)  # Heavy season
            df['MEDIUM_WORKLOAD'] = ((df['TOTAL_MINUTES'] >= 1500) & (df['TOTAL_MINUTES'] <= 2500)).astype(int)
            df['LOW_WORKLOAD'] = (df['TOTAL_MINUTES'] < 1500).astype(int)
            features_created += 3
        
        # Workload trend (increasing workload over time)
        if all(col in df.columns for col in ['GP_LAG1', 'MIN_LAG1', 'GP_LAG2', 'MIN_LAG2']):
            df['WORKLOAD_LAG1'] = df['GP_LAG1'] * df['MIN_LAG1']
            df['WORKLOAD_LAG2'] = df['GP_LAG2'] * df['MIN_LAG2']
            df['WORKLOAD_TREND'] = df['WORKLOAD_LAG1'] - df['WORKLOAD_LAG2']
            df['INCREASING_WORKLOAD'] = (df['WORKLOAD_TREND'] > 200).astype(int)
            features_created += 4
        
        # Usage rate with high minutes (star player fatigue risk)
        if 'ADV_USG_PCT' in df.columns and 'MIN' in df.columns:
            df['USAGE_MINUTES_PRODUCT'] = df['ADV_USG_PCT'] * df['MIN']
            df['HIGH_USAGE_HIGH_MINUTES'] = (
                (df['ADV_USG_PCT'] > 28) & (df['MIN'] > 33)
            ).astype(int)
            features_created += 2
        
        print(f"✓ Created {features_created} workload features")
        
        return df
    
    def create_rest_and_recovery_features(self, df):
        """
        Create features related to rest patterns and recovery
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset with rest/recovery features added
        """
        print("\n" + "="*60)
        print("Creating Rest and Recovery Features")
        print("="*60)
        
        features_created = 0
        
        # Minutes change (indication of load management or recovery)
        if 'MIN_LAG1' in df.columns and 'MIN_LAG2' in df.columns:
            df['MIN_CHANGE'] = df['MIN_LAG1'] - df['MIN_LAG2']
            df['REDUCED_MINUTES'] = (df['MIN_CHANGE'] < -3).astype(int)  # 3+ min reduction
            df['INCREASED_MINUTES'] = (df['MIN_CHANGE'] > 3).astype(int)  # 3+ min increase
            features_created += 3
        
        # Load management indicator (veteran playing fewer games on good team)
        if all(col in df.columns for col in ['AGE', 'GP_LAG1', 'TEAM_W_PCT']):
            df['LOAD_MANAGEMENT'] = (
                (df['AGE'] > 32) & 
                (df['GP_LAG1'] < 70) & 
                (df['TEAM_W_PCT'] > 0.5)
            ).astype(int)
            features_created += 1
        
        # Recovery from low minutes/games
        if 'GP_LAG1' in df.columns and 'MIN_LAG1' in df.columns:
            df['RECOVERING_FROM_LOW_PLAY'] = (
                (df['GP_LAG1'] < 60) | (df['MIN_LAG1'] < 25)
            ).astype(int)
            features_created += 1
        
        print(f"✓ Created {features_created} rest and recovery features")
        
        return df
    
    def create_availability_score(self, df):
        """
        Create composite availability score
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset with availability score added
        """
        print("\n" + "="*60)
        print("Creating Composite Availability Score")
        print("="*60)
        
        # Simple availability score based on recent history
        score_components = []
        
        # Recent games played (weighted)
        if 'GP_LAG1' in df.columns:
            score_components.append(df['GP_LAG1'] / 82 * 0.5)  # 50% weight on last season
        
        if 'GP_LAG2' in df.columns:
            score_components.append(df['GP_LAG2'] / 82 * 0.3)  # 30% weight on 2 seasons ago
        
        if 'GP_LAG3' in df.columns:
            score_components.append(df['GP_LAG3'] / 82 * 0.2)  # 20% weight on 3 seasons ago
        
        if score_components:
            df['AVAILABILITY_SCORE'] = sum(score_components)
            
            # Adjust for age (older players slightly lower score)
            if 'AGE' in df.columns:
                age_adjustment = 1 - ((df['AGE'] - 25) / 100).clip(lower=0, upper=0.15)
                df['AVAILABILITY_SCORE'] = df['AVAILABILITY_SCORE'] * age_adjustment
            
            # Categorize availability score
            df['AVAILABILITY_TIER'] = pd.cut(df['AVAILABILITY_SCORE'],
                                             bins=[0, 0.6, 0.8, 1.0],
                                             labels=[1, 2, 3])  # 1=Low, 2=Medium, 3=High
            df['AVAILABILITY_TIER'] = df['AVAILABILITY_TIER'].astype(float)
            
            print("✓ Created composite availability score")
            return df
        else:
            print("⚠ Not enough data to create availability score")
            return df
    
    def save_dataset(self, df, filename='modeling_dataset.csv'):
        """Save final modeling dataset (overwrite existing)"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved updated dataset to: {filepath}")
        return filepath
    
    def generate_summary(self, df, original_shape):
        """
        Generate summary of availability features
        
        Args:
            df (DataFrame): Final dataset
            original_shape (tuple): Original dataset shape
        """
        print("\n" + "="*60)
        print("AVAILABILITY FEATURES SUMMARY")
        print("="*60)
        
        print(f"\nOriginal dataset: {original_shape}")
        print(f"Final dataset: {df.shape}")
        print(f"Features added: {df.shape[1] - original_shape[1]}")
        
        # Availability statistics
        if 'HIGH_AVAILABILITY' in df.columns:
            high_avail = df['HIGH_AVAILABILITY'].sum()
            medium_avail = df['MEDIUM_AVAILABILITY'].sum() if 'MEDIUM_AVAILABILITY' in df.columns else 0
            low_avail = df['LOW_AVAILABILITY'].sum() if 'LOW_AVAILABILITY' in df.columns else 0
            
            print(f"\nAvailability breakdown:")
            print(f"  High availability (70+ games): {high_avail} ({high_avail/len(df)*100:.1f}%)")
            print(f"  Medium availability (50-69 games): {medium_avail} ({medium_avail/len(df)*100:.1f}%)")
            print(f"  Low availability (<50 games): {low_avail} ({low_avail/len(df)*100:.1f}%)")
        
        # Injury proxy indicators
        if 'INJURY_PRONE_PATTERN' in df.columns:
            injury_prone = df['INJURY_PRONE_PATTERN'].sum()
            print(f"\nInjury-prone pattern detected: {injury_prone} players ({injury_prone/len(df)*100:.1f}%)")
        
        if 'LOAD_MANAGEMENT' in df.columns:
            load_mgmt = df['LOAD_MANAGEMENT'].sum()
            print(f"Load management candidates: {load_mgmt} players ({load_mgmt/len(df)*100:.1f}%)")
        
        # Show example
        print("\nExample - High availability player:")
        if 'HIGH_AVAILABILITY' in df.columns:
            high_avail_player = df[df['HIGH_AVAILABILITY'] == 1].head(1)
            if len(high_avail_player) > 0:
                player = high_avail_player.iloc[0]
                print(f"  Player: {player.get('PLAYER_NAME', 'N/A')}")
                print(f"  Season: {player.get('SEASON', 'N/A')}")
                print(f"  GP: {player.get('GP', 'N/A')}")
                print(f"  Availability Score: {player.get('AVAILABILITY_SCORE', 'N/A'):.3f}")


def main():
    """Main execution function"""
    
    print("="*60)
    print("NBA Availability Feature Creation")
    print("="*60)
    
    # Initialize creator
    creator = AvailabilityFeatureCreator()
    
    # Load dataset
    df = creator.load_dataset()
    
    if df is None:
        print("\n✗ Failed to load dataset")
        return None
    
    original_shape = df.shape
    
    # Create all availability features
    df = creator.create_games_played_features(df)
    df = creator.create_availability_trend_features(df)
    df = creator.create_injury_proxy_features(df)
    df = creator.create_workload_features(df)
    df = creator.create_rest_and_recovery_features(df)
    df = creator.create_availability_score(df)
    
    # Generate summary
    creator.generate_summary(df, original_shape)
    
    # Save updated modeling dataset
    creator.save_dataset(df)
    
    print("\n" + "="*60)
    print("✓ Availability feature creation complete!")
    print("="*60)

    
    return df


if __name__ == "__main__":
    # Run the main function
    df = main()