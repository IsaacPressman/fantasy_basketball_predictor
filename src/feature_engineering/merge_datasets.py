"""
Merge Datasets
Combines player stats, player info, team stats, and advanced stats into one master dataset
"""

import pandas as pd
import os

class DatasetMerger:
    """Merge all collected NBA data sources"""
    
    def __init__(self, data_dir='data/raw', output_dir='data/processed'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_all_data(self):
        """
        Load all CSV files from raw data directory
        
        Returns:
            dict: Dictionary containing all DataFrames
        """
        print("="*60)
        print("Loading all datasets...")
        print("="*60)
        
        data = {}
        
        # Load player stats
        player_stats_path = os.path.join(self.data_dir, 'player_stats_2020_2025.csv')
        if os.path.exists(player_stats_path):
            data['player_stats'] = pd.read_csv(player_stats_path)
            print(f"✓ Player stats: {data['player_stats'].shape}")
        else:
            print(f"✗ Player stats not found at {player_stats_path}")
        
        # Load player info
        player_info_path = os.path.join(self.data_dir, 'player_info.csv')
        if os.path.exists(player_info_path):
            data['player_info'] = pd.read_csv(player_info_path)
            print(f"✓ Player info: {data['player_info'].shape}")
        else:
            print(f"✗ Player info not found at {player_info_path}")
        
        # Load team stats
        team_stats_path = os.path.join(self.data_dir, 'team_stats_2020_2025.csv')
        if os.path.exists(team_stats_path):
            data['team_stats'] = pd.read_csv(team_stats_path)
            print(f"✓ Team stats: {data['team_stats'].shape}")
        else:
            print(f"✗ Team stats not found at {team_stats_path}")
        
        # Load advanced stats
        advanced_stats_path = os.path.join(self.data_dir, 'advanced_stats_2020_2025.csv')
        if os.path.exists(advanced_stats_path):
            data['advanced_stats'] = pd.read_csv(advanced_stats_path)
            print(f"✓ Advanced stats: {data['advanced_stats'].shape}")
        else:
            print(f"✗ Advanced stats not found at {advanced_stats_path}")
        
        print()
        return data
    
    def merge_datasets(self, data):
        """
        Merge all datasets into one master dataset
        
        Args:
            data (dict): Dictionary containing all DataFrames
            
        Returns:
            DataFrame: Merged master dataset
        """
        print("Merging datasets...")
        
        # Start with player stats as the base
        if 'player_stats' not in data:
            print("✗ Cannot merge without player stats")
            return None
        
        df_master = data['player_stats'].copy()
        print(f"Starting with player stats: {df_master.shape}")
        
        # Merge player info (biographical data)
        if 'player_info' in data:
            df_info = data['player_info'].copy()
            
            # Select relevant columns (avoid duplicates)
            info_cols = ['PLAYER_ID', 'POSITION', 'HEIGHT', 'HEIGHT_INCHES', 
                        'WEIGHT', 'YEARS_EXPERIENCE', 'DRAFT_YEAR', 'DRAFT_ROUND', 
                        'DRAFT_NUMBER', 'SCHOOL', 'COUNTRY']
            
            # Keep only columns that exist
            info_cols = [col for col in info_cols if col in df_info.columns]
            df_info_subset = df_info[info_cols].copy()
            
            # Remove duplicates (player_info doesn't have season, so one row per player)
            df_info_subset = df_info_subset.drop_duplicates(subset=['PLAYER_ID'])
            
            # Merge
            df_master = df_master.merge(
                df_info_subset,
                on='PLAYER_ID',
                how='left'
            )
            print(f"After merging player info: {df_master.shape}")
        
        # Merge team stats (team context)
        if 'team_stats' in data:
            df_team = data['team_stats'].copy()
            
            # Select relevant team columns
            team_cols = ['TEAM_NAME', 'SEASON', 'PACE', 'OFF_RATING', 'DEF_RATING', 
                        'NET_RATING', 'W', 'L', 'W_PCT']
            
            # Add optional columns if they exist
            optional_team_cols = ['TS_PCT', 'EFG_PCT', 'FTA_RATE', 'TM_TOV_PCT', 
                                 'OREB_PCT', 'PIE']
            
            for col in optional_team_cols:
                if col in df_team.columns:
                    team_cols.append(col)
            
            # Keep only columns that exist
            team_cols = [col for col in team_cols if col in df_team.columns]
            df_team_subset = df_team[team_cols].copy()
            
            # Rename to add TEAM_ prefix (except TEAM_NAME and SEASON)
            rename_dict = {col: f'TEAM_{col}' for col in team_cols 
                          if col not in ['TEAM_NAME', 'SEASON']}
            df_team_subset = df_team_subset.rename(columns=rename_dict)
            
            # NBA team abbreviation to full name mapping
            abbrev_to_name = {
                'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
                'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
                'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
                'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
                'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
                'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
                'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
                'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
                'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
                'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
            }
            
            # Add TEAM_NAME to master using abbreviation mapping
            if 'TEAM_ABBREVIATION' in df_master.columns:
                df_master['TEAM_NAME'] = df_master['TEAM_ABBREVIATION'].map(abbrev_to_name)
                
                # Merge with team stats
                df_master = df_master.merge(
                    df_team_subset,
                    on=['TEAM_NAME', 'SEASON'],
                    how='left'
                )
                print(f"After merging team stats: {df_master.shape}")
            else:
                print("⚠ Warning: Could not merge team stats (missing TEAM_ABBREVIATION)")
        
        # Merge advanced stats
        if 'advanced_stats' in data:
            df_advanced = data['advanced_stats'].copy()
            
            # Select relevant advanced columns
            advanced_cols = ['PLAYER_ID', 'SEASON', 'USG_PCT', 'AST_PCT', 'AST_TO', 
                           'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 
                           'EFG_PCT', 'TS_PCT', 'PACE', 'PIE']
            
            # Add optional columns
            optional_adv_cols = ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 
                               'POSS', 'PCT_FGA_2PT', 'PCT_FGA_3PT',
                               'AST_RATIO', 'PCT_PTS_2PT', 'PCT_PTS_3PT']
            
            for col in optional_adv_cols:
                if col in df_advanced.columns:
                    advanced_cols.append(col)
            
            # Keep only columns that exist
            advanced_cols = [col for col in advanced_cols if col in df_advanced.columns]
            df_advanced_subset = df_advanced[advanced_cols].copy()
            
            # Rename to add ADV_ prefix (except PLAYER_ID and SEASON)
            # Also avoid renaming columns that would conflict with team stats
            rename_dict = {}
            for col in advanced_cols:
                if col not in ['PLAYER_ID', 'SEASON']:
                    # Add ADV_ prefix to player advanced stats
                    rename_dict[col] = f'ADV_{col}'
            
            df_advanced_subset = df_advanced_subset.rename(columns=rename_dict)
            
            # Merge
            df_master = df_master.merge(
                df_advanced_subset,
                on=['PLAYER_ID', 'SEASON'],
                how='left'
            )
            print(f"After merging advanced stats: {df_master.shape}")
        
        print(f"\n✓ Final master dataset: {df_master.shape}")
        return df_master
    
    def clean_merged_data(self, df):
        """
        Clean the merged dataset
        
        Args:
            df (DataFrame): Merged dataset
            
        Returns:
            DataFrame: Cleaned dataset
        """
        print("\nCleaning merged data...")
        
        # Remove duplicates
        before_count = len(df)
        df = df.drop_duplicates(subset=['PLAYER_ID', 'SEASON'], keep='first')
        if before_count > len(df):
            print(f"  Removed {before_count - len(df)} duplicate records")
        
        # Filter to players with minimum games
        df = df[df['GP'] >= 10].copy()
        print(f"  Filtered to players with 10+ games: {len(df)} records")
        
        # Sort by player and season
        df = df.sort_values(['PLAYER_ID', 'SEASON'])
        
        # Check for missing values
        missing_summary = df.isnull().sum()
        missing_pct = (missing_summary / len(df) * 100).round(1)
        
        print(f"\n  Columns with missing values (>5%):")
        high_missing = missing_pct[missing_pct > 5].sort_values(ascending=False)
        if len(high_missing) > 0:
            for col, pct in high_missing.items():
                print(f"    {col}: {pct}%")
        else:
            print("    None (all columns <5% missing)")
        
        return df
    
    def save_master_dataset(self, df, filename='master_dataset.csv'):
        """Save master dataset to CSV"""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved master dataset to: {filepath}")
        return filepath
    
    def generate_summary_report(self, df):
        """
        Generate a summary report of the merged dataset
        
        Args:
            df (DataFrame): Master dataset
        """
        print("\n" + "="*60)
        print("MASTER DATASET SUMMARY")
        print("="*60)
        
        print(f"\nTotal records: {len(df)}")
        print(f"Unique players: {df['PLAYER_ID'].nunique()}")
        print(f"Seasons: {sorted(df['SEASON'].unique())}")
        print(f"Total features: {len(df.columns)}")
        
        print("\nRecords per season:")
        print(df['SEASON'].value_counts().sort_index())
        
        print("\nSample of data:")
        display_cols = ['PLAYER_NAME', 'SEASON', 'AGE', 'POSITION', 
                       'GP', 'MIN', 'PTS', 'REB', 'AST']
        available_cols = [col for col in display_cols if col in df.columns]
        print(df[available_cols].head(10))
        
        print("\nAll columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:3d}. {col}")


def main():
    """Main execution function"""
    
    # Initialize merger
    merger = DatasetMerger()
    
    # Load all data
    data = merger.load_all_data()
    
    if len(data) == 0:
        print("\n✗ No data files found. Please run data collection scripts first.")
        return None
    
    # Merge datasets
    df_master = merger.merge_datasets(data)
    
    if df_master is not None:
        # Clean merged data
        df_clean = merger.clean_merged_data(df_master)
        
        # Generate summary report
        merger.generate_summary_report(df_clean)
        
        # Save master dataset
        merger.save_master_dataset(df_clean)
        
        print("\n" + "="*60)
        print("✓ Master dataset creation complete!")
        print("="*60)
        
        return df_clean
    else:
        print("\n✗ Failed to create master dataset")
        return None


if __name__ == "__main__":
    # Run the main function
    df = main()