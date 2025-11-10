"""
Advanced Stats Fetcher
Collects advanced player metrics like PER, Win Shares, Usage Rate, etc.
"""

import pandas as pd
import time
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import players
import os

class AdvancedStatsFetcher:
    """Fetch advanced NBA player statistics"""
    
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def get_all_active_players(self):
        """Get list of all active NBA players"""
        print("Getting list of active players...")
        
        all_players = players.get_players()
        df_players = pd.DataFrame(all_players)
        
        # Filter active players only
        df_active = df_players[df_players['is_active'] == True].copy()
        
        print(f"✓ Found {len(df_active)} active players\n")
        return df_active
    
    def fetch_advanced_stats(self, season, measure_type='Advanced'):
        """
        Fetch advanced player stats for a specific season
        
        Args:
            season (str): Season in format '2023-24'
            measure_type (str): 'Advanced', 'Misc', 'Usage', 'Scoring'
            
        Returns:
            DataFrame: Advanced player statistics
        """
        print(f"Fetching {measure_type} stats for {season} season...")
        
        try:
            # Fetch stats using NBA API
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='PerGame',
                measure_type_detailed_defense=measure_type,
                season_type_all_star='Regular Season'
            )
            
            # Convert to DataFrame
            df = stats.get_data_frames()[0]
            
            # Add season column
            df['SEASON'] = season
            df['MEASURE_TYPE'] = measure_type
            
            # Wait to respect API rate limits
            time.sleep(1)
            
            print(f"  ✓ Fetched {len(df)} players")
            return df
            
        except Exception as e:
            print(f"  ✗ Error fetching {season} {measure_type}: {str(e)}")
            return None
    
    def fetch_multiple_seasons(self, start_year=2020, end_year=2024, active_only=True):
        """
        Fetch advanced stats for multiple seasons and measure types
        
        Args:
            start_year (int): Starting year (e.g., 2020 for 2020-21 season)
            end_year (int): Ending year (e.g., 2023 for 2023-24 season)
            active_only (bool): If True, only fetch currently active players
            
        Returns:
            dict: DataFrames for each measure type
        """
        # Get active players first if filtering
        active_player_ids = None
        if active_only:
            active_players = self.get_all_active_players()
            active_player_ids = set(active_players['id'].tolist())
        
        # Measure types to fetch
        measure_types = ['Advanced', 'Misc', 'Usage', 'Scoring']
        
        all_data = {mt: [] for mt in measure_types}
        
        for year in range(start_year, end_year + 1):
            # Format season string (e.g., '2023-24')
            season = f"{year}-{str(year + 1)[-2:]}"
            
            for measure_type in measure_types:
                season_df = self.fetch_advanced_stats(season, measure_type)
                
                if season_df is not None:
                    # Filter to active players if requested
                    if active_only and active_player_ids:
                        before_count = len(season_df)
                        season_df = season_df[season_df['PLAYER_ID'].isin(active_player_ids)].copy()
                        print(f"  → Filtered to {len(season_df)} active players (from {before_count} total)")
                    
                    all_data[measure_type].append(season_df)
        
        # Combine each measure type
        combined_data = {}
        for measure_type, dfs in all_data.items():
            if dfs:
                combined_data[measure_type] = pd.concat(dfs, ignore_index=True)
                print(f"\n✓ {measure_type}: {len(combined_data[measure_type])} player-seasons")
        
        return combined_data
    
    def merge_advanced_stats(self, stats_dict):
        """
        Merge different measure types into one comprehensive dataset
        
        Args:
            stats_dict (dict): Dictionary of DataFrames from different measure types
            
        Returns:
            DataFrame: Merged advanced statistics
        """
        print("\nMerging advanced statistics...")
        
        # Start with Advanced stats as base
        df_merged = stats_dict.get('Advanced', pd.DataFrame())
        
        if df_merged.empty:
            print("✗ No advanced stats available")
            return None
        
        # Keep key columns from Advanced
        base_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'AGE', 
                     'GP', 'MIN', 'SEASON']
        
        # Advanced metrics we want
        advanced_cols = ['PLAYER_ID', 'SEASON', 'OFF_RATING', 'DEF_RATING', 
                        'NET_RATING', 'AST_PCT', 'AST_TO', 'AST_RATIO',
                        'OREB_PCT', 'DREB_PCT', 'REB_PCT', 
                        'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'PACE', 'PIE']
        
        # Usage metrics
        usage_cols = ['PLAYER_ID', 'SEASON', 'USG_PCT', 'POSS']
        
        # Misc metrics
        misc_cols = ['PLAYER_ID', 'SEASON', 'PTS_OFF_TOV', 'PTS_2ND_CHANCE', 
                    'PTS_FB', 'PTS_PAINT', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE',
                    'OPP_PTS_FB', 'OPP_PTS_PAINT']
        
        # Scoring metrics  
        scoring_cols = ['PLAYER_ID', 'SEASON', 'PCT_FGA_2PT', 'PCT_FGA_3PT',
                       'PCT_PTS_2PT', 'PCT_PTS_2PT_MR', 'PCT_PTS_3PT',
                       'PCT_PTS_FB', 'PCT_PTS_FT', 'PCT_PTS_OFF_TOV', 'PCT_PTS_PAINT']
        
        # Keep relevant columns from Advanced
        keep_cols = [col for col in base_cols + advanced_cols if col in df_merged.columns]
        keep_cols = list(set(keep_cols))  # Remove duplicates
        df_merged = df_merged[keep_cols].copy()
        
        # Merge Usage stats
        if 'Usage' in stats_dict:
            df_usage = stats_dict['Usage']
            merge_cols = [col for col in usage_cols if col in df_usage.columns]
            df_merged = df_merged.merge(
                df_usage[merge_cols],
                on=['PLAYER_ID', 'SEASON'],
                how='left',
                suffixes=('', '_usage')
            )
        
        # Merge Misc stats
        if 'Misc' in stats_dict:
            df_misc = stats_dict['Misc']
            merge_cols = [col for col in misc_cols if col in df_misc.columns]
            df_merged = df_merged.merge(
                df_misc[merge_cols],
                on=['PLAYER_ID', 'SEASON'],
                how='left',
                suffixes=('', '_misc')
            )
        
        # Merge Scoring stats
        if 'Scoring' in stats_dict:
            df_scoring = stats_dict['Scoring']
            merge_cols = [col for col in scoring_cols if col in df_scoring.columns]
            df_merged = df_merged.merge(
                df_scoring[merge_cols],
                on=['PLAYER_ID', 'SEASON'],
                how='left',
                suffixes=('', '_scoring')
            )
        
        print(f"✓ Merged data: {len(df_merged)} player-seasons with {len(df_merged.columns)} features")
        return df_merged
    
    def clean_advanced_stats(self, df):
        """
        Clean and prepare advanced statistics
        
        Args:
            df (DataFrame): Raw advanced stats data
            
        Returns:
            DataFrame: Cleaned advanced stats
        """
        print("\nCleaning advanced statistics...")
        
        # Filter out players with very few games (< 10)
        df = df[df['GP'] >= 10].copy()
        
        # Remove any duplicate player-season combinations (keep first)
        df = df.drop_duplicates(subset=['PLAYER_ID', 'SEASON'], keep='first')
        
        # Handle missing values for percentage stats (fill with 0)
        pct_cols = [col for col in df.columns if 'PCT' in col.upper()]
        for col in pct_cols:
            df[col] = df[col].fillna(0)
        
        # Handle missing values for rate stats (fill with median)
        rate_cols = ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE']
        for col in rate_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Sort by season and player name
        df = df.sort_values(['SEASON', 'PLAYER_NAME'])
        
        print(f"✓ Cleaned data: {len(df)} player-seasons")
        print(f"✓ Players with 10+ games from {df['SEASON'].min()} to {df['SEASON'].max()}")
        
        return df
    
    def save_advanced_stats(self, df, filename='advanced_stats_2020_2025.csv'):
        """Save advanced stats to CSV file"""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved to: {filepath}")
        return filepath


def main():
    """Main execution function"""
    
    print("="*60)
    print("NBA Advanced Statistics Collection")
    print("="*60)
    
    # Initialize fetcher
    fetcher = AdvancedStatsFetcher()
    
    # Fetch data for seasons 2020-21 through 2023-24
    # active_only=True means we only fetch currently active players
    stats_dict = fetcher.fetch_multiple_seasons(
        start_year=2020,
        end_year=2024,
        active_only=True
    )
    
    if stats_dict:
        # Merge all measure types
        df_merged = fetcher.merge_advanced_stats(stats_dict)
        
        if df_merged is not None:
            # Clean the data
            df_clean = fetcher.clean_advanced_stats(df_merged)
            
            # Display sample
            print("\nSample advanced stats:")
            display_cols = ['PLAYER_NAME', 'SEASON', 'USG_PCT', 'TS_PCT', 
                          'AST_PCT', 'REB_PCT', 'PIE']
            available_display = [col for col in display_cols if col in df_clean.columns]
            print(df_clean[available_display].head(10))
            
            # Display summary
            print("\nSummary statistics:")
            summary_cols = ['USG_PCT', 'TS_PCT', 'OFF_RATING', 'DEF_RATING']
            available_summary = [col for col in summary_cols if col in df_clean.columns]
            if available_summary:
                print(df_clean[available_summary].describe().round(2))
            
            print(f"\nTotal columns collected: {len(df_clean.columns)}")
            
            # Save to CSV
            fetcher.save_advanced_stats(df_clean)
            
            print("\n" + "="*60)
            print("✓ Advanced stats collection complete!")
            print("="*60)
            
            return df_clean
        else:
            print("✗ Failed to merge advanced stats")
            return None
    else:
        print("✗ Advanced stats collection failed")
        return None


if __name__ == "__main__":
    # Run the main function
    df = main()