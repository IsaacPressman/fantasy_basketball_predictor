"""
Create Derived Features
Engineers new features from existing data using basketball knowledge
"""

import pandas as pd
import numpy as np
import os

class DerivedFeatureCreator:
    """Create derived/engineered features using domain knowledge"""
    
    def __init__(self, data_dir='data/processed'):
        self.data_dir = data_dir
    
    def load_dataset(self, filename='dataset_with_lags.csv'):
        """Load the dataset with lag features"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"✗ Dataset not found at {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        print(f"✓ Loaded dataset: {df.shape}")
        return df
    
    def create_age_features(self, df):
        """
        Create age-related features (age curves, prime indicators)
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset with age features added
        """
        print("\n" + "="*60)
        print("Creating Age-Based Features")
        print("="*60)
        
        if 'AGE' not in df.columns:
            print("⚠ AGE column not found, skipping age features")
            return df
        
        # Age squared (captures non-linear age effects - peak and decline)
        df['AGE_SQUARED'] = df['AGE'] ** 2
        
        # Prime age indicator (27-30 is typically prime for NBA)
        df['IN_PRIME'] = ((df['AGE'] >= 27) & (df['AGE'] <= 30)).astype(int)
        
        # Young player indicator (under 25)
        df['IS_YOUNG'] = (df['AGE'] < 25).astype(int)
        
        # Veteran indicator (over 32)
        df['IS_VETERAN'] = (df['AGE'] > 32).astype(int)
        
        # Age bins (categorical encoding of age ranges)
        df['AGE_BIN'] = pd.cut(df['AGE'], 
                                bins=[0, 23, 27, 30, 33, 100],
                                labels=[1, 2, 3, 4, 5])
        df['AGE_BIN'] = df['AGE_BIN'].astype(float)
        
        print("✓ Created 6 age-based features")
        
        return df
    
    def create_efficiency_features(self, df):
        """
        Create efficiency metrics
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset with efficiency features added
        """
        print("\n" + "="*60)
        print("Creating Efficiency Features")
        print("="*60)
        
        features_created = 0
        
        # Points per minute
        if 'PTS' in df.columns and 'MIN' in df.columns:
            df['PTS_PER_MIN'] = df['PTS'] / df['MIN'].replace(0, np.nan)
            features_created += 1
        
        # Assists per turnover ratio
        if 'AST' in df.columns and 'TOV' in df.columns:
            df['AST_TO_RATIO'] = df['AST'] / df['TOV'].replace(0, np.nan)
            features_created += 1
        
        # Points per shot attempt (includes FTs)
        if 'PTS' in df.columns and 'FGA' in df.columns and 'FTA' in df.columns:
            total_attempts = df['FGA'] + (0.44 * df['FTA'])
            df['PTS_PER_ATTEMPT'] = df['PTS'] / total_attempts.replace(0, np.nan)
            features_created += 1
        
        # Shooting efficiency score (TS% if available, otherwise calculated)
        if 'ADV_TS_PCT' in df.columns:
            df['SHOOTING_EFFICIENCY'] = df['ADV_TS_PCT']
            features_created += 1
        elif all(col in df.columns for col in ['PTS', 'FGA', 'FTA']):
            df['SHOOTING_EFFICIENCY'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA'])).replace(0, np.nan)
            features_created += 1
        
        # Minutes per game (games played availability)
        if 'MIN' in df.columns and 'GP' in df.columns:
            df['MPG'] = df['MIN']  # Already per game, but good to have separate column
            df['GP_RATE'] = df['GP'] / 82  # What % of season did they play?
            features_created += 2
        
        print(f"✓ Created {features_created} efficiency features")
        
        return df
    
    def create_role_features(self, df):
        """
        Create features that indicate player's role on team
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset with role features added
        """
        print("\n" + "="*60)
        print("Creating Role/Usage Features")
        print("="*60)
        
        features_created = 0
        
        # Usage rate (if not already present from advanced stats)
        if 'ADV_USG_PCT' in df.columns:
            df['USAGE_RATE'] = df['ADV_USG_PCT']
            features_created += 1
        
        # Shot volume (attempts per minute)
        if 'FGA' in df.columns and 'MIN' in df.columns:
            df['FGA_PER_MIN'] = df['FGA'] / df['MIN'].replace(0, np.nan)
            features_created += 1
        
        # Three-point volume (3PA per game)
        if 'FG3A' in df.columns:
            df['THREE_PT_VOLUME'] = df['FG3A']
            
            # Three-point rate (% of FGA that are 3s)
            if 'FGA' in df.columns:
                df['THREE_PT_RATE'] = df['FG3A'] / df['FGA'].replace(0, np.nan)
                features_created += 1
            
            features_created += 1
        
        # Scoring role (high volume scorer vs efficient role player)
        if 'FGA' in df.columns and 'PTS' in df.columns:
            df['SCORER_ROLE'] = (df['FGA'] > 12).astype(int)  # 12+ FGA = primary scorer
            features_created += 1
        
        # Playmaker role (high assist rate)
        if 'AST' in df.columns:
            df['PLAYMAKER_ROLE'] = (df['AST'] > 4).astype(int)  # 4+ APG = playmaker
            features_created += 1
        
        # Rebounder role
        if 'REB' in df.columns:
            df['REBOUNDER_ROLE'] = (df['REB'] > 7).astype(int)  # 7+ RPG = rebounder
            features_created += 1
        
        # Defender role (steals + blocks)
        if 'STL' in df.columns and 'BLK' in df.columns:
            df['DEFENSIVE_STATS'] = df['STL'] + df['BLK']
            df['DEFENDER_ROLE'] = (df['DEFENSIVE_STATS'] > 2).astype(int)
            features_created += 2
        
        print(f"✓ Created {features_created} role/usage features")
        
        return df
    
    def create_team_context_features(self, df):
        """
        Create features that adjust for team context
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset with team context features added
        """
        print("\n" + "="*60)
        print("Creating Team Context Features")
        print("="*60)
        
        features_created = 0
        
        # Pace-adjusted stats (adjust player stats for team pace)
        if 'TEAM_PACE' in df.columns:
            league_avg_pace = 100  # Approximate NBA average
            df['PACE_ADJUSTMENT'] = df['TEAM_PACE'] / league_avg_pace
            
            # Adjust counting stats for pace
            if 'PTS' in df.columns:
                df['PTS_PACE_ADJ'] = df['PTS'] / df['PACE_ADJUSTMENT']
                features_created += 1
            
            if 'AST' in df.columns:
                df['AST_PACE_ADJ'] = df['AST'] / df['PACE_ADJUSTMENT']
                features_created += 1
            
            if 'REB' in df.columns:
                df['REB_PACE_ADJ'] = df['REB'] / df['PACE_ADJUSTMENT']
                features_created += 1
            
            features_created += 1  # For PACE_ADJUSTMENT itself
        
        # Team quality indicators
        if 'TEAM_W_PCT' in df.columns:
            df['ON_WINNING_TEAM'] = (df['TEAM_W_PCT'] > 0.5).astype(int)
            df['ON_PLAYOFF_TEAM'] = (df['TEAM_W_PCT'] > 0.45).astype(int)
            features_created += 2
        
        # Offensive team strength
        if 'TEAM_OFF_RATING' in df.columns:
            league_avg_ortg = 110  # Approximate
            df['TEAM_OFF_STRENGTH'] = df['TEAM_OFF_RATING'] - league_avg_ortg
            features_created += 1
        
        # Defensive team strength
        if 'TEAM_DEF_RATING' in df.columns:
            league_avg_drtg = 110  # Approximate
            df['TEAM_DEF_STRENGTH'] = league_avg_drtg - df['TEAM_DEF_RATING']  # Reversed (lower is better)
            features_created += 1
        
        # Net rating (team quality)
        if 'TEAM_NET_RATING' in df.columns:
            df['TEAM_NET_RATING_ABS'] = df['TEAM_NET_RATING'].abs()  # Overall team strength
            features_created += 1
        
        print(f"✓ Created {features_created} team context features")
        
        return df
    
    def create_interaction_features(self, df):
        """
        Create interaction features (combinations of existing features)
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset with interaction features added
        """
        print("\n" + "="*60)
        print("Creating Interaction Features")
        print("="*60)
        
        features_created = 0
        
        # Age * Usage (young high-usage players might improve more)
        if 'AGE' in df.columns and 'ADV_USG_PCT' in df.columns:
            df['AGE_USAGE_INTERACTION'] = df['AGE'] * df['ADV_USG_PCT']
            features_created += 1
        
        # Minutes * Efficiency (players with high minutes and efficiency are stars)
        if 'MIN' in df.columns and 'ADV_TS_PCT' in df.columns:
            df['MIN_EFFICIENCY_INTERACTION'] = df['MIN'] * df['ADV_TS_PCT']
            features_created += 1
        
        # Team pace * Usage (high usage on fast team = more opportunities)
        if 'TEAM_PACE' in df.columns and 'ADV_USG_PCT' in df.columns:
            df['PACE_USAGE_INTERACTION'] = df['TEAM_PACE'] * df['ADV_USG_PCT']
            features_created += 1
        
        # Age * Team quality (older players on good teams rest more)
        if 'AGE' in df.columns and 'TEAM_W_PCT' in df.columns:
            df['AGE_TEAM_QUALITY'] = df['AGE'] * df['TEAM_W_PCT']
            features_created += 1
        
        # Years experience * Minutes (experienced players with high minutes are reliable)
        if 'YEARS_EXPERIENCE' in df.columns and 'MIN' in df.columns:
            df['EXP_MIN_INTERACTION'] = df['YEARS_EXPERIENCE'] * df['MIN']
            features_created += 1
        
        print(f"✓ Created {features_created} interaction features")
        
        return df
    
    def create_position_features(self, df):
        """
        Create position-based features
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset with position features added
        """
        print("\n" + "="*60)
        print("Creating Position Features")
        print("="*60)
        
        if 'POSITION' not in df.columns:
            print("⚠ POSITION column not found, skipping position features")
            return df
        
        # One-hot encode positions
        position_dummies = pd.get_dummies(df['POSITION'], prefix='POS')
        
        # Add position dummies to dataframe
        for col in position_dummies.columns:
            df[col] = position_dummies[col]
        
        # Create position groups
        df['IS_GUARD'] = df['POSITION'].str.contains('G', na=False).astype(int)
        df['IS_FORWARD'] = df['POSITION'].str.contains('F', na=False).astype(int)
        df['IS_CENTER'] = df['POSITION'].str.contains('C', na=False).astype(int)
        
        features_created = len(position_dummies.columns) + 3
        print(f"✓ Created {features_created} position features")
        
        return df
    
    def create_physical_features(self, df):
        """
        Create features from physical attributes
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            DataFrame: Dataset with physical features added
        """
        print("\n" + "="*60)
        print("Creating Physical Attribute Features")
        print("="*60)
        
        features_created = 0
        
        # BMI (Body Mass Index)
        if 'HEIGHT_INCHES' in df.columns and 'WEIGHT' in df.columns:
            df['BMI'] = (df['WEIGHT'] / (df['HEIGHT_INCHES'] ** 2)) * 703
            features_created += 1
        
        # Height categories
        if 'HEIGHT_INCHES' in df.columns:
            df['HEIGHT_CAT'] = pd.cut(df['HEIGHT_INCHES'],
                                       bins=[0, 75, 79, 82, 100],
                                       labels=[1, 2, 3, 4])  # Short, Average, Tall, Very Tall
            df['HEIGHT_CAT'] = df['HEIGHT_CAT'].astype(float)
            features_created += 1
        
        # Weight categories
        if 'WEIGHT' in df.columns:
            df['WEIGHT_CAT'] = pd.cut(df['WEIGHT'],
                                       bins=[0, 190, 220, 250, 400],
                                       labels=[1, 2, 3, 4])  # Light, Average, Heavy, Very Heavy
            df['WEIGHT_CAT'] = df['WEIGHT_CAT'].astype(float)
            features_created += 1
        
        print(f"✓ Created {features_created} physical attribute features")
        
        return df
    
    def save_dataset(self, df, filename='modeling_dataset.csv'):
        """Save final modeling dataset"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved modeling dataset to: {filepath}")
        return filepath
    
    def generate_summary(self, df, original_shape):
        """
        Generate summary of feature engineering
        
        Args:
            df (DataFrame): Final dataset
            original_shape (tuple): Original dataset shape
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        
        print(f"\nOriginal dataset: {original_shape}")
        print(f"Final dataset: {df.shape}")
        print(f"Features added: {df.shape[1] - original_shape[1]}")
        
        # Count different feature types
        age_features = [col for col in df.columns if 'AGE' in col.upper() or 'PRIME' in col or 'YOUNG' in col or 'VETERAN' in col]
        efficiency_features = [col for col in df.columns if 'EFFICIENCY' in col or 'PER_MIN' in col or 'RATIO' in col]
        role_features = [col for col in df.columns if 'ROLE' in col or 'VOLUME' in col]
        team_features = [col for col in df.columns if 'TEAM' in col or 'PACE_ADJ' in col]
        interaction_features = [col for col in df.columns if 'INTERACTION' in col]
        position_features = [col for col in df.columns if 'POS_' in col or 'IS_GUARD' in col or 'IS_FORWARD' in col]
        physical_features = [col for col in df.columns if 'BMI' in col or 'HEIGHT_CAT' in col or 'WEIGHT_CAT' in col]
        
        print(f"\nFeature breakdown:")
        print(f"  Age-based features: {len(age_features)}")
        print(f"  Efficiency features: {len(efficiency_features)}")
        print(f"  Role/Usage features: {len(role_features)}")
        print(f"  Team context features: {len(team_features)}")
        print(f"  Interaction features: {len(interaction_features)}")
        print(f"  Position features: {len(position_features)}")
        print(f"  Physical features: {len(physical_features)}")
        
        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
        high_missing = missing_pct[missing_pct > 50].sort_values(ascending=False)
        
        print(f"\nFeatures with >50% missing data: {len(high_missing)}")
        if len(high_missing) > 0:
            print("Top 5:")
            for col, pct in high_missing.head(5).items():
                print(f"  {col}: {pct}%")


def main():
    """Main execution function"""
    
    print("="*60)
    print("NBA Derived Feature Creation")
    print("="*60)
    
    # Initialize creator
    creator = DerivedFeatureCreator()
    
    # Load dataset
    df = creator.load_dataset()
    
    if df is None:
        print("\n✗ Failed to load dataset")
        return None
    
    original_shape = df.shape
    
    # Create all derived features
    df = creator.create_age_features(df)
    df = creator.create_efficiency_features(df)
    df = creator.create_role_features(df)
    df = creator.create_team_context_features(df)
    df = creator.create_interaction_features(df)
    df = creator.create_position_features(df)
    df = creator.create_physical_features(df)
    
    # Generate summary
    creator.generate_summary(df, original_shape)
    
    # Save final modeling dataset
    creator.save_dataset(df)
    
    print("\n" + "="*60)
    print("✓ Feature engineering complete!")
    print("="*60)
    print("\nYour data is now ready for model training!")
    
    return df


if __name__ == "__main__":
    # Run the main function
    df = main()