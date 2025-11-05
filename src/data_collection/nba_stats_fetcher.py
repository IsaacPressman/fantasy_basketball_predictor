"""
NBA Stats Fetcher
Collects historical player statistics from NBA API for multiple seasons
"""

import pandas as pd
import time
from nba_api.stats.endpoints import leaguedashplayerstats, playercareerstats
from nba_api.stats.static import players
import os

class NBAStatsFetcher:
    """Fetch historical NBA player statistics"""
    
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def fetch_player_season_stats(self, season, per_mode='PerGame'):
        """
        Fetch player stats for a specific season
        
        Args:
            season (str): Season in format '2023-24'
            per_mode (str): 'PerGame', 'Totals', or 'Per36'
            
        Returns:
            DataFrame: Player statistics
        """
        print(f"Fetching {per_mode} stats for {season} season...")
        
        try:
            # Fetch stats using NBA API
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed=per_mode,
                season_type_all_star='Regular Season'
            )
            
            # Convert to DataFrame
            df = stats.get_data_frames()[0]
            
            # Add season column
            df['SEASON'] = season
            
            # Wait to respect API rate limits
            time.sleep(1)
            
            print(f"✓ Fetched {len(df)} players for {season}")
            return df
            
        except Exception as e:
            print(f"✗ Error fetching {season}: {str(e)}")
            return None
    
    def fetch_multiple_seasons(self, start_year=2020
                               , end_year=2024, per_mode='PerGame', active_only=True):
        """
        Fetch player stats for multiple seasons
        
        Args:
            start_year (int): Starting year (e.g., 2019 for 2019-20 season)
            end_year (int): Ending year (e.g., 2024 for 2023-24 season)
            per_mode (str): Stats mode
            active_only (bool): If True, only fetch active players
            
        Returns:
            DataFrame: Combined stats from all seasons
        """
        # Get active players first if filtering
        active_player_ids = None
        if active_only:
            print("Getting list of active players...")
            active_players = self.get_all_active_players()
            active_player_ids = set(active_players['id'].tolist())
            print(f"✓ Found {len(active_player_ids)} active players\n")
        
        all_seasons = []
        
        for year in range(start_year, end_year + 1):
            # Format season string (e.g., '2023-24')
            season = f"{year}-{str(year + 1)[-2:]}"
            
            # Fetch season data
            season_df = self.fetch_player_season_stats(season, per_mode)
            
            if season_df is not None:
                # Filter to active players if requested
                if active_only and active_player_ids:
                    before_count = len(season_df)
                    season_df = season_df[season_df['PLAYER_ID'].isin(active_player_ids)].copy()
                    print(f"  → Filtered to {len(season_df)} active players (from {before_count} total)")
                
                all_seasons.append(season_df)
        
        # Combine all seasons
        if all_seasons:
            combined_df = pd.concat(all_seasons, ignore_index=True)
            print(f"\n✓ Total records collected: {len(combined_df)}")
            return combined_df
        else:
            print("✗ No data collected")
            return None
    
    def clean_stats_data(self, df):
        """
        Clean and prepare stats data
        
        Args:
            df (DataFrame): Raw stats data
            
        Returns:
            DataFrame: Cleaned stats data
        """
        print("\nCleaning data...")
        
        # Select relevant columns
        columns_to_keep = [
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'AGE',
            'GP', 'MIN',  # Games played, Minutes
            'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',  # Basic stats
            'FGM', 'FGA', 'FG_PCT',  # Field goals made, attempted, percentage
            'FG3M', 'FG3A', 'FG3_PCT',  # Three-pointers made, attempted, percentage
            'FTM', 'FTA', 'FT_PCT',  # Free throws made, attempted, percentage
            'SEASON'
        ]
        
        # Check which columns exist
        available_columns = [col for col in columns_to_keep if col in df.columns]
        df_clean = df[available_columns].copy()
        
        # Filter out players with very few games (< 10)
        df_clean = df_clean[df_clean['GP'] >= 10].copy()
        
        # Handle missing values
        df_clean['FG_PCT'] = df_clean['FG_PCT'].fillna(0)
        df_clean['FG3_PCT'] = df_clean['FG3_PCT'].fillna(0)
        df_clean['FT_PCT'] = df_clean['FT_PCT'].fillna(0)
        
        # Sort by season and points
        df_clean = df_clean.sort_values(['SEASON', 'PTS'], ascending=[True, False])
        
        print(f"✓ Cleaned data: {len(df_clean)} player-seasons")
        print(f"✓ Players with 10+ games from {df_clean['SEASON'].min()} to {df_clean['SEASON'].max()}")
        
        return df_clean
    
    def save_stats(self, df, filename='player_stats_2020_2025.csv'):
        """Save stats to CSV file"""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved to: {filepath}")
        return filepath
    
    def get_all_active_players(self):
        """Get list of all active NBA players"""
        print("Fetching all active players...")
        
        all_players = players.get_players()
        df_players = pd.DataFrame(all_players)
        
        # Filter active players only
        df_active = df_players[df_players['is_active'] == True].copy()
        
        print(f"✓ Found {len(df_active)} active players")
        return df_active


def main():
    """Main execution function"""
    
    print("="*60)
    print("NBA Historical Stats Collection")
    print("="*60)
    
    # Initialize fetcher
    fetcher = NBAStatsFetcher()
    
    # Fetch data for last 5 seasons (2019-20 through 2023-24)
    # active_only=True means we only fetch currently active players
    df_stats = fetcher.fetch_multiple_seasons(
        start_year=2020,
        end_year=2024,
        per_mode='PerGame',
        active_only=True  # Set to False to include all players (retired too)
    )
    
    if df_stats is not None:
        # Clean the data
        df_clean = fetcher.clean_stats_data(df_stats)
        
        # Display sample
        print("\nSample data:")
        print(df_clean.head(10))
        
        # Display summary statistics
        print("\nSummary by season:")
        print(df_clean.groupby('SEASON').agg({
            'PLAYER_ID': 'count',
            'PTS': 'mean',
            'REB': 'mean',
            'AST': 'mean'
        }).round(2))
        
        # Save to CSV
        fetcher.save_stats(df_clean)
        
        print("\n" + "="*60)
        print("✓ Data collection complete!")
        print("="*60)
        
        return df_clean
    else:
        print("✗ Data collection failed")
        return None


if __name__ == "__main__":
    # Run the main function
    df = main()