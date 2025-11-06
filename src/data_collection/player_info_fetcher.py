"""
Player Info Fetcher
Collects biographical and career information for NBA players
"""

import pandas as pd
import time
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonplayerinfo, playercareerstats
import os

class PlayerInfoFetcher:
    """Fetch NBA player biographical information"""
    
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def get_all_active_players(self):
        """Get list of all active NBA players"""
        print("Fetching all active players...")
        
        all_players = players.get_players()
        df_players = pd.DataFrame(all_players)
        
        # Filter active players only
        df_active = df_players[df_players['is_active'] == True].copy()
        
        print(f"✓ Found {len(df_active)} active players")
        return df_active
    
    def fetch_player_info(self, player_id):
        """
        Fetch detailed info for a single player
        
        Args:
            player_id (int): NBA player ID
            
        Returns:
            dict: Player information
        """
        try:
            # Fetch player info
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            df_info = player_info.get_data_frames()[0]
            
            if len(df_info) > 0:
                info = df_info.iloc[0].to_dict()
                
                # Wait to respect API rate limits
                time.sleep(0.6)  # NBA API rate limit is ~1 request per 0.6 seconds
                
                return info
            else:
                return None
                
        except Exception as e:
            print(f"  ✗ Error fetching player {player_id}: {str(e)}")
            return None
    
    def fetch_all_player_info(self, player_ids=None):
        """
        Fetch detailed info for multiple players
        
        Args:
            player_ids (list): List of player IDs. If None, fetches all active players
            
        Returns:
            DataFrame: Player information
        """
        # Get active players if no IDs provided
        if player_ids is None:
            active_players = self.get_all_active_players()
            player_ids = active_players['id'].tolist()
        
        print(f"\nFetching detailed info for {len(player_ids)} players...")
        print("This may take a few minutes due to API rate limits...\n")
        
        all_info = []
        
        for idx, player_id in enumerate(player_ids, 1):
            if idx % 50 == 0:
                print(f"Progress: {idx}/{len(player_ids)} players...")
            
            info = self.fetch_player_info(player_id)
            
            if info:
                all_info.append(info)
        
        # Convert to DataFrame
        df_info = pd.DataFrame(all_info)
        
        print(f"\n✓ Successfully fetched info for {len(df_info)} players")
        return df_info
    
    def clean_player_info(self, df):
        """
        Clean and select relevant player information
        
        Args:
            df (DataFrame): Raw player info data
            
        Returns:
            DataFrame: Cleaned player info
        """
        print("\nCleaning player info...")
        
        # Select relevant columns
        columns_to_keep = [
            'PERSON_ID',
            'DISPLAY_FIRST_LAST',  # Full name
            'BIRTHDATE',
            'SCHOOL',  # College
            'COUNTRY',
            'HEIGHT',
            'WEIGHT',
            'SEASON_EXP',  # Years of experience
            'POSITION',
            'DRAFT_YEAR',
            'DRAFT_ROUND',
            'DRAFT_NUMBER',
            'TEAM_ID',
            'TEAM_NAME',
            'TEAM_ABBREVIATION',
            'TEAM_CITY'
        ]
        
        # Check which columns exist
        available_columns = [col for col in columns_to_keep if col in df.columns]
        df_clean = df[available_columns].copy()
        
        # Rename columns for consistency
        rename_map = {
            'PERSON_ID': 'PLAYER_ID',
            'DISPLAY_FIRST_LAST': 'PLAYER_NAME',
            'SEASON_EXP': 'YEARS_EXPERIENCE'
        }
        df_clean = df_clean.rename(columns=rename_map)
        
        # Convert birthdate to age (if available)
        if 'BIRTHDATE' in df_clean.columns:
            df_clean['BIRTHDATE'] = pd.to_datetime(df_clean['BIRTHDATE'], errors='coerce')
            current_date = pd.Timestamp.now()
            df_clean['AGE'] = ((current_date - df_clean['BIRTHDATE']).dt.days / 365.25).round(1)
        
        # Parse height (convert from '6-7' format to inches)
        if 'HEIGHT' in df_clean.columns:
            df_clean['HEIGHT_INCHES'] = df_clean['HEIGHT'].apply(self._parse_height)
        
        # Convert weight to numeric
        if 'WEIGHT' in df_clean.columns:
            df_clean['WEIGHT'] = pd.to_numeric(df_clean['WEIGHT'], errors='coerce')
        
        # Fill missing values
        if 'YEARS_EXPERIENCE' in df_clean.columns:
            df_clean['YEARS_EXPERIENCE'] = df_clean['YEARS_EXPERIENCE'].fillna(0)
        
        if 'DRAFT_YEAR' in df_clean.columns:
            df_clean['DRAFT_YEAR'] = df_clean['DRAFT_YEAR'].fillna('Undrafted')
        
        # Sort by player name
        df_clean = df_clean.sort_values('PLAYER_NAME')
        
        print(f"✓ Cleaned info for {len(df_clean)} players")
        return df_clean
    
    def _parse_height(self, height_str):
        """
        Convert height from '6-7' format to total inches
        
        Args:
            height_str (str): Height in format 'feet-inches'
            
        Returns:
            int: Total height in inches
        """
        if pd.isna(height_str) or height_str == '':
            return None
        
        try:
            parts = str(height_str).split('-')
            if len(parts) == 2:
                feet = int(parts[0])
                inches = int(parts[1])
                return feet * 12 + inches
            else:
                return None
        except:
            return None
    
    def save_player_info(self, df, filename='player_info.csv'):
        """Save player info to CSV file"""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved to: {filepath}")
        return filepath
    
    def get_player_career_stats(self, player_id):
        """
        Get career totals for a player (useful for experience metrics)
        
        Args:
            player_id (int): NBA player ID
            
        Returns:
            dict: Career statistics
        """
        try:
            career = playercareerstats.PlayerCareerStats(player_id=player_id)
            df_career = career.get_data_frames()[0]
            
            if len(df_career) > 0:
                # Sum up totals across all seasons
                career_totals = {
                    'PLAYER_ID': player_id,
                    'CAREER_GP': df_career['GP'].sum(),
                    'CAREER_MIN': df_career['MIN'].sum(),
                    'CAREER_PTS': df_career['PTS'].sum(),
                    'SEASONS_PLAYED': len(df_career)
                }
                
                time.sleep(0.6)
                return career_totals
            else:
                return None
                
        except Exception as e:
            return None


def main():
    """Main execution function"""
    
    print("="*60)
    print("NBA Player Biographical Data Collection")
    print("="*60)
    
    # Initialize fetcher
    fetcher = PlayerInfoFetcher()
    
    # Fetch all active player info
    df_info = fetcher.fetch_all_player_info()
    
    if df_info is not None and len(df_info) > 0:
        # Clean the data
        df_clean = fetcher.clean_player_info(df_info)
        
        # Display sample
        print("\nSample player info:")
        print(df_clean[['PLAYER_NAME', 'POSITION', 'HEIGHT', 'WEIGHT', 
                        'YEARS_EXPERIENCE', 'TEAM_ABBREVIATION']].head(10))
        
        # Display summary statistics
        print("\nSummary:")
        print(f"Total players: {len(df_clean)}")
        print(f"\nPositions breakdown:")
        print(df_clean['POSITION'].value_counts())
        print(f"\nAverage experience: {df_clean['YEARS_EXPERIENCE'].mean():.1f} years")
        print(f"Average age: {df_clean['AGE'].mean():.1f} years")
        
        # Save to CSV
        fetcher.save_player_info(df_clean)
        
        print("\n" + "="*60)
        print("✓ Player info collection complete!")
        print("="*60)
        
        return df_clean
    else:
        print("✗ Player info collection failed")
        return None


if __name__ == "__main__":
    # Run the main function
    df = main()