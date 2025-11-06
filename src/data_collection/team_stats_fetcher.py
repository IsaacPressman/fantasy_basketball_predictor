"""
Team Stats Fetcher
Collects team-level statistics that provide context for player projections
"""

import pandas as pd
import time
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams
import os

class TeamStatsFetcher:
    """Fetch NBA team statistics for multiple seasons"""
    
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def get_all_teams(self):
        """Get list of all NBA teams"""
        print("Fetching all NBA teams...")
        
        all_teams = teams.get_teams()
        df_teams = pd.DataFrame(all_teams)
        
        print(f"✓ Found {len(df_teams)} teams")
        return df_teams
    
    def fetch_team_season_stats(self, season, measure_type='Base'):
        """
        Fetch team stats for a specific season
        
        Args:
            season (str): Season in format '2023-24'
            measure_type (str): 'Base', 'Advanced', 'Four Factors', 'Misc'
            
        Returns:
            DataFrame: Team statistics
        """
        print(f"Fetching {measure_type} stats for {season} season...")
        
        try:
            # Fetch team stats - specify NBA league ID
            stats = leaguedashteamstats.LeagueDashTeamStats(
                league_id_nullable='00',
                season=season,
                measure_type_detailed_defense=measure_type,
                per_mode_detailed='PerGame',
                season_type_all_star='Regular Season',
            )
            
            df = stats.get_data_frames()[0]
            
            # Additional filter: Only keep NBA teams (30 teams)
            # Filter out any non-NBA teams by checking team ID format
            if 'TEAM_ID' in df.columns:
                # NBA team IDs are 10 digits starting with '161'
                df = df[df['TEAM_ID'].astype(str).str.startswith('161')].copy()
            
            df['SEASON'] = season
            df['MEASURE_TYPE'] = measure_type
            
            # Wait to respect API rate limits
            time.sleep(1)
            
            print(f"  ✓ Fetched {len(df)} NBA teams")
            return df
            
        except Exception as e:
            print(f"  ✗ Error fetching {season} {measure_type}: {str(e)}")
            return None
    
    
    def fetch_multiple_seasons(self, start_year=2020, end_year=2024):
        """
        Fetch team stats for multiple seasons
        
        Args:
            start_year (int): Starting year (e.g., 2020 for 2020-21 season)
            end_year (int): Ending year (e.g., 2023 for 2023-24 season)
            
        Returns:
            dict: DataFrames for each measure type
        """
        # We need to fetch multiple measure types to get all stats we want
        measure_types = ['Base', 'Advanced', 'Four Factors', 'Misc']
        
        all_data = {mt: [] for mt in measure_types}
        
        for year in range(start_year, end_year + 1):
            season = f"{year}-{str(year + 1)[-2:]}"
            
            for measure_type in measure_types:
                season_df = self.fetch_team_season_stats(season, measure_type)
                
                if season_df is not None:
                    all_data[measure_type].append(season_df)
        
        # Combine each measure type
        combined_data = {}
        for measure_type, dfs in all_data.items():
            if dfs:
                combined_data[measure_type] = pd.concat(dfs, ignore_index=True)
                print(f"\n✓ {measure_type}: {len(combined_data[measure_type])} team-seasons")
        
        return combined_data
    
    def merge_team_stats(self, stats_dict):
        """
        Merge different measure types into one comprehensive dataset
        
        Args:
            stats_dict (dict): Dictionary of DataFrames from different measure types
            
        Returns:
            DataFrame: Merged team statistics
        """
        print("\nMerging team statistics...")
        
        # Start with Base stats
        df_merged = stats_dict.get('Base', pd.DataFrame())
        
        if df_merged.empty:
            print("✗ No base stats available")
            return None
        
        # Define which columns to keep from each measure type
        # Base columns (already have most)
        base_cols = ['TEAM_ID', 'TEAM_NAME', 'SEASON', 'GP', 'W', 'L', 'W_PCT']
        
        # Advanced columns
        advanced_cols = ['TEAM_ID', 'SEASON', 'OFF_RATING', 'DEF_RATING', 
                        'NET_RATING', 'PACE', 'PIE']
        
        # Four Factors columns
        four_factors_cols = ['TEAM_ID', 'SEASON', 'EFG_PCT', 'FTA_RATE', 
                            'TM_TOV_PCT', 'OREB_PCT']
        
        # Misc columns
        misc_cols = ['TEAM_ID', 'SEASON', 'PTS_2ND_CHANCE', 'PTS_FB', 
                    'PTS_PAINT', 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_FB']
        
        # Keep relevant columns from base
        keep_from_base = [col for col in base_cols if col in df_merged.columns]
        keep_from_base += ['MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT', 
                          'FG3M', 'FG3A', 'FG3_PCT', 
                          'FTM', 'FTA', 'FT_PCT',
                          'REB', 'AST', 'TOV', 'STL', 'BLK']
        
        df_merged = df_merged[[col for col in keep_from_base if col in df_merged.columns]].copy()
        
        # Merge Advanced stats
        if 'Advanced' in stats_dict:
            df_advanced = stats_dict['Advanced']
            merge_cols = [col for col in advanced_cols if col in df_advanced.columns]
            df_merged = df_merged.merge(
                df_advanced[merge_cols],
                on=['TEAM_ID', 'SEASON'],
                how='left'
            )
        
        # Merge Four Factors
        if 'Four Factors' in stats_dict:
            df_four = stats_dict['Four Factors']
            merge_cols = [col for col in four_factors_cols if col in df_four.columns]
            df_merged = df_merged.merge(
                df_four[merge_cols],
                on=['TEAM_ID', 'SEASON'],
                how='left'
            )
        
        # Merge Misc
        if 'Misc' in stats_dict:
            df_misc = stats_dict['Misc']
            merge_cols = [col for col in misc_cols if col in df_misc.columns]
            df_merged = df_merged.merge(
                df_misc[merge_cols],
                on=['TEAM_ID', 'SEASON'],
                how='left'
            )
        
        print(f"✓ Merged data: {len(df_merged)} team-seasons with {len(df_merged.columns)} features")
        return df_merged
    
    def calculate_derived_stats(self, df):
        """
        Calculate additional useful statistics
        
        Args:
            df (DataFrame): Team stats
            
        Returns:
            DataFrame: Team stats with derived features
        """
        print("\nCalculating derived statistics...")
        
        df = df.copy()
        
        # True Shooting % (if not already present)
        if 'TS_PCT' not in df.columns and all(col in df.columns for col in ['PTS', 'FGA', 'FTA']):
            df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
        
        # 3-Point Attempt Rate
        if 'FG3A_RATE' not in df.columns and all(col in df.columns for col in ['FG3A', 'FGA']):
            df['FG3A_RATE'] = df['FG3A'] / df['FGA']
        
        # Assist Ratio (assists per 100 possessions)
        if 'AST_RATIO' not in df.columns and all(col in df.columns for col in ['AST', 'FGA', 'TOV', 'FTA']):
            possessions = df['FGA'] + 0.44 * df['FTA'] - df['OREB_PCT'] * (df['FGA'] - df['FGM']) + df['TOV']
            df['AST_RATIO'] = 100 * df['AST'] / possessions
        
        # Turnover Rate (turnovers per 100 possessions)
        if 'TOV_RATE' not in df.columns and 'TM_TOV_PCT' in df.columns:
            df['TOV_RATE'] = df['TM_TOV_PCT']
        elif 'TOV_RATE' not in df.columns and all(col in df.columns for col in ['TOV', 'FGA', 'FTA']):
            possessions = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
            df['TOV_RATE'] = 100 * df['TOV'] / possessions
        
        # Rebound percentage (if components available)
        if 'REB_PCT' not in df.columns and 'OREB_PCT' in df.columns:
            df['REB_PCT'] = df['OREB_PCT']  # Using offensive rebound % as proxy
        
        print(f"✓ Added derived statistics")
        return df
    
    def clean_team_stats(self, df):
        """
        Clean and organize team statistics
        
        Args:
            df (DataFrame): Raw team stats
            
        Returns:
            DataFrame: Cleaned team stats
        """
        print("\nCleaning team statistics...")
        
        # Remove any duplicate team-season combinations
        df = df.drop_duplicates(subset=['TEAM_ID', 'SEASON'], keep='first')
        
        # Sort by season and team name
        df = df.sort_values(['SEASON', 'TEAM_NAME'])
        
        # Fill any missing values with league averages (by season)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if col not in ['TEAM_ID', 'GP', 'W', 'L']:
                df[col] = df.groupby('SEASON')[col].transform(
                    lambda x: x.fillna(x.mean())
                )
        
        print(f"✓ Cleaned data: {len(df)} team-seasons")
        return df
    
    def save_team_stats(self, df, filename='team_stats_2020_2025.csv'):
        """Save team stats to CSV file"""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved to: {filepath}")
        return filepath


def main():
    """Main execution function"""
    
    print("="*60)
    print("NBA Team Statistics Collection")
    print("="*60)
    
    # Initialize fetcher
    fetcher = TeamStatsFetcher()
    
    # Fetch data for seasons 2020-21 through 2024-25
    stats_dict = fetcher.fetch_multiple_seasons(
        start_year=2020,
        end_year=2024
    )
    
    if stats_dict:
        # Merge all measure types
        df_merged = fetcher.merge_team_stats(stats_dict)
        
        if df_merged is not None:
            # Calculate derived stats
            df_derived = fetcher.calculate_derived_stats(df_merged)
            
            # Clean the data
            df_clean = fetcher.clean_team_stats(df_derived)
            
            # Display sample
            print("\nSample team stats:")
            display_cols = ['TEAM_NAME', 'SEASON', 'W', 'L', 'PACE', 
                          'OFF_RATING', 'DEF_RATING', 'TS_PCT']
            available_display = [col for col in display_cols if col in df_clean.columns]
            print(df_clean[available_display].head(10))
            
            # Display summary
            print("\nSummary statistics by season:")
            summary_cols = ['PACE', 'OFF_RATING', 'DEF_RATING']
            available_summary = [col for col in summary_cols if col in df_clean.columns]
            if available_summary:
                print(df_clean.groupby('SEASON')[available_summary].mean().round(2))
            
            print(f"\nTotal columns collected: {len(df_clean.columns)}")
            print(f"Columns: {df_clean.columns.tolist()}")
            
            # Save to CSV
            fetcher.save_team_stats(df_clean)
            
            print("\n" + "="*60)
            print("✓ Team stats collection complete!")
            print("="*60)
            
            return df_clean
        else:
            print("✗ Failed to merge team stats")
            return None
    else:
        print("✗ Team stats collection failed")
        return None


if __name__ == "__main__":
    # Run the main function
    df = main()