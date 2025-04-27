# --- START OF FILE multi_page_app.py ---

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np # Import numpy
import os
import traceback # Import traceback for detailed error logging

# --- Helper Function: Load Main Data (from total_data.csv) ---
# [LOAD DATA FUNCTIONS - KEEP AS IS - OMITTED FOR BREVITY]
# --- Helper Function: Load Main Data (from total_data.csv) ---
def load_main_data():
    """Loads, cleans, standardizes main data from total_data.csv."""
    print("--- Running load_main_data() [from total_data.csv] ---")
    data_filename = 'total_data.csv'
    data_path = os.path.join('data', data_filename)
    print(f"Attempting to load main data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"ERROR: Main data file '{data_path}' not found.")
        return None
    try:
        # Consider loading without dtype_spec first if errors occur, then apply types
        df = pd.read_csv(data_path, low_memory=False) # Removed dtype_spec for initial load flexibility
        print(f"Initial main data shape: {df.shape}")

        # --- Data Cleaning (Main Data - Keep relevant parts) ---
        numeric_cols_main = [
            'runs_scored', 'balls_faced', 'wickets_taken', 'balls_bowled',
            'runs_conceded', 'bowled_done', 'lbw_done', 'player_out',
            'fours_scored', 'sixes_scored', 'runs_off_bat', 'extras',
            'wides', 'noballs', 'byes', 'legbyes', 'penalty', 'win_by_runs', 'win_by_wickets',
            'caught_done', 'stumped_done', 'run_out_direct', 'run_out_throw',
            'run_out_involved', 'dot_balls_as_bowler', 'maidens', 'catches_taken',
            'stumpings_done', 'innings', 'over', 'delivery', 'dot_balls_as_batsman'
        ]
        missing_numeric = []
        for col in numeric_cols_main:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col in ['win_by_runs', 'win_by_wickets', 'innings', 'over', 'delivery']:
                     # Coerce to float first to handle non-numeric strings if any, then Int64
                     df[col] = df[col].astype(float).astype('Int64')
                else:
                    df[col] = df[col].fillna(0)
                    # Decide int vs float after fillna
                    # Check using vectorized operations for efficiency
                    is_whole = pd.isna(df[col]) | (df[col] % 1 == 0)
                    if is_whole.all():
                        max_val = df[col].abs().max()
                        if pd.isna(max_val) or max_val < 2**31:
                            df[col] = df[col].astype(np.int32)
                        else:
                            df[col] = df[col].astype(np.int64)
                    # else: keep as float (already is)
            else:
                print(f"Warning (Main Data): Numeric column '{col}' not found. Creating with 0.")
                df[col] = 0
                missing_numeric.append(col)


        # String/Categorical Cleaning
        if 'out_kind' in df.columns:
            df['out_kind'] = df['out_kind'].fillna('not out').astype(str).str.lower().str.strip()
        else:
             df['out_kind'] = 'not out'

        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        else:
             df['start_date'] = pd.NaT

        if 'match_type' in df.columns:
            ALLOWED_FORMATS = ["ODI", "T20", "Test", "T20I"]
            df['match_type'] = df['match_type'].astype(str).str.upper().str.strip()
            original_rows = len(df)
            df = df[df["match_type"].isin(ALLOWED_FORMATS)].copy()
            print(f"Filtered main data by match_type. Kept {len(df)} of {original_rows} rows.")
        else:
            print("Warning (Main Data): 'match_type' column not found.")
            # Decide how to handle this - maybe return None or create a default?
            # For now, let it proceed, downstream pages must handle missing column.


        team_cols = ['player_team', 'opposition_team', 'batting_team', 'bowling_team', 'winner', 'toss_winner']
        team_replacements = {
            'U.A.E.': 'United Arab Emirates', 'UAE': 'United Arab Emirates',
            'P.N.G.': 'Papua New Guinea', 'PNG': 'Papua New Guinea',
            'USA': 'United States of America',
            'West Indies Cricket Board': 'West Indies',
            'Rising Pune Supergiant': 'Rising Pune Supergiants',
            'Rising Pune Supergiants': 'Rising Pune Supergiants',
            'Delhi Daredevils': 'Delhi Capitals',
        }
        missing_teams = []
        for team_col in team_cols:
             if team_col in df.columns:
                df[team_col] = df[team_col].astype(str).str.strip()
                df[team_col] = df[team_col].replace(team_replacements)
                df[team_col] = df[team_col].fillna('Unknown')
             else:
                 print(f"Warning (Main Data): Team column '{team_col}' not found. Creating with 'Unknown'.")
                 df[team_col] = 'Unknown'
                 missing_teams.append(team_col)

        str_cols_to_fill = ['venue', 'city', 'toss_decision', 'result', 'result_details',
                            'umpire1', 'umpire2', 'event_name', 'match_number',
                            'batsman', 'non_striker', 'bowler', 'fielder', 'bowler_involved_in_out',
                            'name', 'bowling_style', 'role']
        missing_str = []
        for col in str_cols_to_fill:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)
            else:
                print(f"Warning (Main Data): String column '{col}' not found. Creating with 'Unknown'.")
                df[col] = 'Unknown'
                missing_str.append(col)

        if df.empty:
            print("Warning: Main DataFrame became empty after cleaning/filtering.")
            return None

        print(f"--- load_main_data() successful, final shape: {df.shape} ---")
        if missing_numeric or missing_teams or missing_str:
             print(f"    -> Note: Created missing columns: {missing_numeric + missing_teams + missing_str}")
        return df

    except FileNotFoundError:
        print(f"ERROR: Main data file '{data_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"ERROR: Main data file '{data_path}' is empty.")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error during main data loading/processing: {e}")
        traceback.print_exc()
        return None

# --- Helper Function: Load Tournament Data (from mw_overall.csv) ---
def load_tournament_data():
    """Loads and preprocesses data from mw_overall.csv for Match Explorer."""
    print("--- Running load_tournament_data() [from mw_overall.csv] ---")
    tournament_filename = 'mw_overall.csv'
    data_path = os.path.join('data', tournament_filename)
    print(f"Attempting to load tournament data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"ERROR: Tournament data file '{data_path}' not found.")
        return None
    try:
        df = pd.read_csv(data_path, low_memory=False)
        print(f"Tournament data initial load shape: {df.shape}")

        required_cols_me = [
            'match_id', 'event_name', 'city', 'venue', 'winner', 'toss_winner', 'toss_decision',
            'batting_team', 'bowling_team', 'runs_off_bat', 'player_out', # Using player_out
            'wides', 'noballs', 'byes', 'legbyes', 'penalty', 'out_kind',
            'season', 'start_date', 'match_type', 'umpire1', 'umpire2'
        ]
        required_cols_pre = ['batting_team', 'match_id', 'city', 'venue']
        all_required_cols = list(set(required_cols_me + required_cols_pre))

        missing_cols = [col for col in all_required_cols if col not in df.columns]
        if missing_cols:
            print(f"WARNING (Tournament Data): File '{tournament_filename}' is missing columns: {missing_cols}.")
            for col in missing_cols:
                if any(k in col for k in ['run', 'wide', 'noball', 'bye', 'legbye', 'penalty', 'out']):
                    print(f"  -> Creating missing numeric column '{col}' with 0.")
                    df[col] = 0
                elif col == 'bowling_team':
                    print(f"  -> Column '{col}' will be derived.")
                    continue
                else:
                    print(f"  -> Creating missing string column '{col}' with 'Unknown'.")
                    df[col] = 'Unknown'

        numeric_cols_mw = ['runs_off_bat', 'wides', 'noballs', 'byes', 'legbyes', 'penalty', 'player_out']
        for col in numeric_cols_mw:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                is_whole = pd.isna(df[col]) | (df[col] % 1 == 0)
                if is_whole.all():
                    max_val = df[col].abs().max()
                    if pd.isna(max_val) or max_val < 2**31:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
             # Creation handled above

        str_cols_mw = ['event_name', 'winner', 'season', 'batting_team', 'match_id',
                       'city', 'venue', 'toss_winner', 'toss_decision', 'out_kind',
                       'umpire1', 'umpire2', 'match_type']
        team_replacements_mw = { # Use the same replacements as main data
            'U.A.E.': 'United Arab Emirates', 'UAE': 'United Arab Emirates',
            'P.N.G.': 'Papua New Guinea', 'PNG': 'Papua New Guinea',
            'USA': 'United States of America',
            'West Indies Cricket Board': 'West Indies',
            'Rising Pune Supergiant': 'Rising Pune Supergiants',
            'Rising Pune Supergiants': 'Rising Pune Supergiants',
            'Delhi Daredevils': 'Delhi Capitals',
        }
        for col in str_cols_mw:
             if col in df.columns:
                 df[col] = df[col].astype(str).str.strip()
                 if col in ['batting_team', 'winner', 'toss_winner']:
                     df[col] = df[col].replace(team_replacements_mw)
                 if col == 'out_kind':
                     df[col] = df[col].fillna('not out').str.lower().str.strip()
                 elif col == 'match_type':
                      ALLOWED_FORMATS = ["ODI", "T20", "Test", "T20I"]
                      df[col] = df[col].str.upper()
                      # Consider filtering here if needed, or let page handle it
                      # df = df[df["match_type"].isin(ALLOWED_FORMATS)].copy()
                 else:
                     df[col] = df[col].fillna('Unknown')
             # Creation handled above

        # --- Preprocessing ---
        print("Applying preprocessing for Match Explorer...")
        # 1. Map Cities/Venues to Countries
        city_to_country = {
             # Use the extensive dictionary from previous code
            'Dubai': 'United Arab Emirates', 'Sharjah': 'United Arab Emirates', 'Abu Dhabi': 'United Arab Emirates',
            'London': 'United Kingdom', 'Manchester': 'United Kingdom', 'Birmingham': 'United Kingdom', 'Cardiff': 'United Kingdom', 'Southampton': 'United Kingdom', 'Leeds': 'United Kingdom', 'Chester-le-Street': 'United Kingdom', 'Nottingham': 'United Kingdom', 'Bristol': 'United Kingdom', 'Taunton': 'United Kingdom', 'Hove': 'United Kingdom', "Lord's": 'United Kingdom', 'Kennington Oval': 'United Kingdom', 'Edgbaston': 'United Kingdom', 'The Rose Bowl': 'United Kingdom', 'Sophia Gardens': 'United Kingdom', 'Trent Bridge': 'United Kingdom', 'Headingley': 'United Kingdom', 'Old Trafford': 'United Kingdom', 'Belfast': 'United Kingdom', 'Bready': 'United Kingdom', 'Edinburgh': 'United Kingdom', 'Glasgow': 'United Kingdom', 'Aberdeen': 'United Kingdom',
            'Sydney': 'Australia', 'Melbourne': 'Australia', 'Adelaide': 'Australia', 'Perth': 'Australia', 'Brisbane': 'Australia', 'Hobart': 'Australia', 'Canberra': 'Australia', 'Geelong':'Australia', 'Launceston': 'Australia', 'MCG': 'Australia', 'SCG': 'Australia', 'Adelaide Oval': 'Australia', 'WACA': 'Australia', 'Gabba': 'Australia', 'Bellerive Oval': 'Australia', 'Manuka Oval': 'Australia',
            'Mumbai': 'India', 'Delhi': 'India', 'Kolkata': 'India', 'Chennai': 'India', 'Bengaluru': 'India', 'Bangalore': 'India', 'Hyderabad': 'India', 'Mohali': 'India', 'Nagpur': 'India', 'Pune': 'India', 'Ahmedabad': 'India', 'Dharamsala': 'India', 'Visakhapatnam': 'India', 'Indore': 'India', 'Rajkot': 'India', 'Ranchi': 'India', 'Cuttack': 'India', 'Guwahati': 'India', 'Lucknow': 'India', 'Kanpur': 'India', 'Jaipur': 'India', 'Chandigarh':'India', 'Eden Gardens': 'India', 'Wankhede Stadium': 'India', 'MA Chidambaram Stadium': 'India', 'M Chinnaswamy Stadium': 'India', 'Rajiv Gandhi International Stadium': 'India', 'Punjab Cricket Association Stadium, Mohali': 'India',
            'Cape Town': 'South Africa', 'Johannesburg': 'South Africa', 'Durban': 'South Africa', 'Centurion': 'South Africa', 'Port Elizabeth': 'South Africa', 'Gqeberha': 'South Africa', 'Paarl': 'South Africa', 'Bloemfontein': 'South Africa', 'East London': 'South Africa', 'Potchefstroom': 'South Africa', 'Kimberley': 'South Africa', 'Benoni': 'South Africa', 'Newlands': 'South Africa', 'Wanderers Stadium': 'South Africa', 'Kingsmead': 'South Africa', 'SuperSport Park': 'South Africa', "St George's Park": 'South Africa',
            'Auckland': 'New Zealand', 'Wellington': 'New Zealand', 'Christchurch': 'New Zealand', 'Hamilton': 'New Zealand', 'Napier': 'New Zealand', 'Dunedin': 'New Zealand', 'Mount Maunganui': 'New Zealand', 'Queenstown': 'New Zealand', 'Nelson': 'New Zealand', 'Eden Park': 'New Zealand', 'Basin Reserve': 'New Zealand', 'Hagley Oval': 'New Zealand', 'Seddon Park': 'New Zealand', 'McLean Park': 'New Zealand', 'University Oval': 'New Zealand', 'Bay Oval': 'New Zealand',
            'Karachi': 'Pakistan', 'Lahore': 'Pakistan', 'Rawalpindi': 'Pakistan', 'Multan': 'Pakistan', 'Faisalabad': 'Pakistan', 'National Stadium, Karachi': 'Pakistan', 'Gaddafi Stadium, Lahore': 'Pakistan',
            'Colombo': 'Sri Lanka', 'Kandy': 'Sri Lanka', 'Galle': 'Sri Lanka', 'Hambantota': 'Sri Lanka', 'Dambulla': 'Sri Lanka', 'Pallekele': 'Sri Lanka', 'R Premadasa Stadium': 'Sri Lanka', 'Pallekele International Cricket Stadium': 'Sri Lanka', 'Galle International Stadium': 'Sri Lanka',
            'Chattogram': 'Bangladesh', 'Chittagong': 'Bangladesh', 'Dhaka': 'Bangladesh', 'Sylhet': 'Bangladesh', 'Mirpur': 'Bangladesh', 'Khulna': 'Bangladesh', 'Fatullah': 'Bangladesh', 'Sher-e-Bangla National Cricket Stadium': 'Bangladesh', 'Zahur Ahmed Chowdhury Stadium': 'Bangladesh',
            'Harare': 'Zimbabwe', 'Bulawayo': 'Zimbabwe', 'Kwekwe': 'Zimbabwe', 'Mutare': 'Zimbabwe', 'Harare Sports Club': 'Zimbabwe', "Queen's Sports Club": 'Zimbabwe',
            'Bridgetown': 'Barbados', 'Kensington Oval, Bridgetown': 'Barbados',
            'Gros Islet': 'Saint Lucia', 'Beausejour Stadium, Gros Islet': 'Saint Lucia',
            'Port of Spain': 'Trinidad and Tobago', "Queen's Park Oval, Port of Spain": 'Trinidad and Tobago',
            'Kingston': 'Jamaica', 'Sabina Park, Kingston': 'Jamaica',
            'Providence': 'Guyana', 'Providence Stadium': 'Guyana',
            'North Sound': 'Antigua and Barbuda', 'Sir Vivian Richards Stadium, North Sound': 'Antigua and Barbuda',
            'Basseterre': 'Saint Kitts and Nevis', 'Warner Park, Basseterre': 'Saint Kitts and Nevis',
            'Kingstown': 'Saint Vincent and the Grenadines', 'Arnos Vale Ground, Kingstown': 'Saint Vincent and the Grenadines',
            'Roseau': 'Dominica', 'Windsor Park, Roseau': 'Dominica',
            'Lauderhill': 'United States', 'Lauderhill, Florida': 'United States', 'Central Broward Regional Park Stadium Turf Ground': 'United States',
            'Dublin': 'Ireland', 'Malahide': 'Ireland',
            'Amstelveen': 'Netherlands', 'Rotterdam': 'Netherlands', 'The Hague': 'Netherlands',
            'Windhoek': 'Namibia', 'Nairobi': 'Kenya', 'Kampala': 'Uganda',
            'Muscat': 'Oman', 'Al Amerat': 'Oman',
            'Kathmandu': 'Nepal', 'Kirtipur': 'Nepal',
            'Singapore': 'Singapore', 'Kuala Lumpur': 'Malaysia', 'Hong Kong': 'Hong Kong',
        }
        if 'city' in df.columns:
            df['country'] = df['city'].map(city_to_country)
            if 'venue' in df.columns:
                 df['country'] = df['country'].fillna(df['venue'].map(city_to_country))
            df['country'] = df['country'].fillna('Unknown')
            unmapped = df[df['country'] == 'Unknown']
            if not unmapped.empty:
                 print(f"  - Warning: {len(unmapped)} rows have unmapped countries.")
                 # print(unmapped[['city', 'venue']].drop_duplicates().head(10))
        else:
            print("  - Warning: 'city' column missing. Cannot map countries.")
            df['country'] = 'Unknown'

        # 2. Calculate unique teams per match
        print("  - Calculating unique teams per match...")
        if 'batting_team' in df.columns and 'match_id' in df.columns:
            df['batting_team_clean'] = df['batting_team'].astype(str).str.strip().replace('', 'Unknown Team').fillna('Unknown Team')
            match_teams_grouped = df[df['batting_team_clean'] != 'Unknown Team'].groupby('match_id')['batting_team_clean'].unique()
            match_teams_grouped = match_teams_grouped.apply(lambda x: sorted(list(set(x))) if isinstance(x, np.ndarray) else [])
            match_teams_df = match_teams_grouped.reset_index()
            match_teams_df.rename(columns={'batting_team_clean': 'match_teams_list'}, inplace=True)

            match_teams_df = match_teams_df[
                match_teams_df['match_teams_list'].apply(lambda x: isinstance(x, list) and len(x) == 2 and x[0] != x[1])
            ]
            print(f"  - Identified {len(match_teams_df)} matches with exactly two unique, different teams.")

            if not match_teams_df.empty:
                df = pd.merge(df, match_teams_df[['match_id', 'match_teams_list']], on='match_id', how='left')
                rows_without_teams = df['match_teams_list'].isnull().sum()
                if rows_without_teams > 0:
                    print(f"  - Note: {rows_without_teams} rows do not have a valid 2-team list.")
            else:
                 print("  - Warning: No matches found with exactly two distinct teams.")
                 df['match_teams_list'] = pd.NA
            df.drop(columns=['batting_team_clean'], inplace=True, errors='ignore')
        else:
            print("  - Warning: Cannot calculate unique match teams ('batting_team' or 'match_id' missing).")
            df['match_teams_list'] = pd.NA

        # 3. Derive Bowling Team
        print("  - Deriving 'bowling_team' column...")
        def get_bowling_team_robust(row):
            match_teams = row['match_teams_list']
            batter_team = row['batting_team'] # Use original column
            if isinstance(match_teams, list) and len(match_teams) == 2:
                if batter_team == match_teams[0]: return match_teams[1]
                if batter_team == match_teams[1]: return match_teams[0]
            return 'Unknown'

        # Only derive if the column doesn't exist or is missing substantially
        derive_bowling = False
        if 'bowling_team' not in df.columns:
            derive_bowling = True
            print("  - 'bowling_team' column missing, will derive.")
        elif df['bowling_team'].astype(str).fillna('Unknown').isin(['Unknown', 'nan', 'None', '']).mean() > 0.5: # Check if >50% unknown
            derive_bowling = True
            print("  - 'bowling_team' column seems mostly missing/unknown, attempting to derive.")

        if derive_bowling:
            if 'match_teams_list' in df.columns and 'batting_team' in df.columns:
                 df['derived_bowling_team'] = df.apply(get_bowling_team_robust, axis=1)
                 # Clean the derived team names
                 df['derived_bowling_team'] = df['derived_bowling_team'].astype(str).str.strip().replace(team_replacements_mw).fillna('Unknown')
                 # Overwrite original bowling_team only where derived is not Unknown
                 # Or completely replace if original was missing
                 if 'bowling_team' not in df.columns:
                     df['bowling_team'] = df['derived_bowling_team']
                 else:
                     df['bowling_team'] = np.where(
                         df['derived_bowling_team'] != 'Unknown',
                         df['derived_bowling_team'],
                         df['bowling_team'].astype(str).fillna('Unknown') # Keep original if derived is Unknown
                     )
                 df.drop(columns=['derived_bowling_team'], inplace=True)
                 unknown_bowling_teams = (df['bowling_team'] == 'Unknown').sum()
                 if unknown_bowling_teams > 0:
                      print(f"  - After derivation, 'bowling_team' is 'Unknown' for {unknown_bowling_teams} rows.")
            else:
                 print("  - Warning: Cannot derive 'bowling_team' ('match_teams_list' or 'batting_team' unavailable).")
                 if 'bowling_team' not in df.columns: df['bowling_team'] = 'Unknown' # Ensure column exists
        else:
            print("  - Skipping derivation of 'bowling_team' as existing column seems sufficient.")
            # Still ensure it exists and is cleaned
            if 'bowling_team' in df.columns:
                df['bowling_team'] = df['bowling_team'].astype(str).str.strip().replace(team_replacements_mw).fillna('Unknown')
            else:
                 df['bowling_team'] = 'Unknown'


        # --- Final Date Conversion ---
        if 'start_date' in df.columns:
             df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        else:
             df['start_date'] = pd.NaT


        if df.empty:
            print("Warning: Tournament DataFrame is empty after processing.")
            return None

        print(f"--- load_tournament_data() successful, final shape: {df.shape} ---")
        # print(df.info())
        return df

    except FileNotFoundError:
        print(f"ERROR: Tournament data file '{data_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
         print(f"ERROR: Tournament data file '{data_path}' is empty.")
         return None
    except Exception as e:
        print(f"ERROR: Unexpected error during tournament data loading/processing: {e}")
        traceback.print_exc()
        return None


# --- Helper Function: Load Player Analysis Data (from app6.csv) ---
def load_player_analysis_data(filepath="app6.csv"):
    """Loads and preprocesses data for the Player Analysis page."""
    print(f"--- Running load_player_analysis_data({filepath}) ---")
    actual_path = os.path.join('data', filepath)
    if not os.path.exists(actual_path):
        print(f"ERROR: Player analysis data file '{actual_path}' not found.")
        return None
    try:
        df = pd.read_csv(actual_path)
        print(f"Initial Player Analysis data shape: {df.shape}")

        # Basic Cleaning & Type Conversion
        if 'name' not in df.columns:
             print(f"ERROR (Player Analysis): Required column 'name' missing in {actual_path}.")
             return None
        df['name'] = df['name'].astype(str).str.strip()

        if 'start_date' not in df.columns:
            print(f"ERROR (Player Analysis): Required column 'start_date' missing in {actual_path}.")
            return None
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        # Drop rows with invalid dates or missing player name early
        df.dropna(subset=['start_date', 'name'], inplace=True)
        if df.empty:
             print(f"Warning: Player analysis DataFrame empty after dropping invalid dates/names from {actual_path}.")
             return None


        numeric_cols = [
            'runs_scored', 'balls_faced', 'fours_scored', 'sixes_scored',
            'catches_taken', 'run_out_direct', 'run_out_throw', 'stumpings_done',
            'player_out', 'balls_bowled', 'runs_conceded', 'wickets_taken', 'bowled_done',
            'lbw_done', 'maidens', 'dot_balls_as_batsman', 'dot_balls_as_bowler'
        ]
        missing_numeric = []
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                is_whole = pd.isna(df[col]) | (df[col] % 1 == 0)
                if is_whole.all():
                    max_val = df[col].abs().max()
                    if pd.isna(max_val) or max_val < 2**31:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                # else keep as float
            else:
                 print(f"Warning (Player Analysis): Numeric column '{col}' missing. Creating with 0.")
                 df[col] = 0
                 missing_numeric.append(col)

        cat_cols = ['out_kind', 'match_type', 'venue', 'city', 'match_id']
        missing_cat = []
        for col in cat_cols:
            if col in df.columns:
                 if col == 'out_kind':
                     df[col] = df[col].fillna('not out').astype(str).str.lower().str.strip()
                 else:
                     df[col] = df[col].fillna('Unknown').astype(str).str.strip()
            else:
                print(f"Warning (Player Analysis): Categorical column '{col}' missing. Creating with 'Unknown'.")
                df[col] = 'Unknown'
                missing_cat.append(col)

        df.sort_values(by='start_date', inplace=True)

        print(f"--- load_player_analysis_data() successful, final shape: {df.shape} ---")
        if missing_numeric or missing_cat:
            print(f"    -> Note: Created missing columns: {missing_numeric + missing_cat}")
        # print(df.info())
        return df

    except FileNotFoundError: # Already checked
        print(f"ERROR: Player analysis data file '{actual_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
         print(f"ERROR: Player analysis data file '{actual_path}' is empty.")
         return None
    except Exception as e:
        print(f"ERROR: Error during player analysis data loading ({actual_path}): {e}")
        traceback.print_exc()
        return None


# --- Helper Function: Load Betting Analyzer Data ---
def load_betting_data(file_path="style_data_with_start_date.csv"):
    """Loads and processes data for the Betting Analyzer page."""
    print(f"--- Running load_betting_data({file_path}) ---")
    actual_path = os.path.join('data', file_path)
    if not os.path.exists(actual_path):
         print(f"ERROR: Betting data file not found at '{actual_path}'.")
         return None

    try:
        df = pd.read_csv(actual_path)
        print(f"Betting Analyzer: Read CSV '{actual_path}'. Shape: {df.shape}")

        required_cols = ['name', 'match_type',
                         'balls_against_spin', 'runs_against_spin', 'outs_against_spin',
                         'balls_against_right_fast', 'runs_against_right_fast', 'outs_against_right_fast',
                         'balls_against_left_fast', 'runs_against_left_fast', 'outs_against_left_fast']
        stat_cols = [c for c in required_cols if c not in ['name', 'match_type']]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"ERROR (Betting Data): Missing required columns: {', '.join(missing_cols)} in {actual_path}")
            return None

        # Basic Cleaning
        df['match_type'] = df['match_type'].fillna('Unknown').astype(str)
        df['name'] = df['name'].fillna('Unknown Player').astype(str)

        for col in stat_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            # Ensure integer type - use int64 for safety with sums
            df[col] = df[col].astype(np.int64)


        grouping_cols = ['name', 'match_type']
        agg_rules = {col: 'sum' for col in stat_cols}
        player_agg = df.groupby(grouping_cols, as_index=False).agg(agg_rules)

        all_player_data = []
        bowling_types_config = {
            'Spin': ('balls_against_spin', 'runs_against_spin', 'outs_against_spin'),
            'Right Fast': ('balls_against_right_fast', 'runs_against_right_fast', 'outs_against_right_fast'),
            'Left Fast': ('balls_against_left_fast', 'runs_against_left_fast', 'outs_against_left_fast'),
        }

        for _, row in player_agg.iterrows():
            for bowling_label, (balls_col, runs_col, outs_col) in bowling_types_config.items():
                 if row[balls_col] > 0: # Only include if faced > 0 balls
                     all_player_data.append({
                        'name': row['name'], 'match_type': row['match_type'],
                        'Bowling Type': bowling_label,
                        'Total Runs': row[runs_col],
                        'Total Balls': row[balls_col],
                        'Total Outs': row[outs_col]
                     })

        if not all_player_data:
            print("WARNING (Betting Data): No valid player/bowling type combinations after aggregation.")
            # Return empty DF with correct columns for downstream consistency
            return pd.DataFrame(columns=['name', 'match_type', 'Bowling Type', 'Total Runs', 'Total Balls', 'Total Outs', 'run_rate', 'out_rate'])

        processed_df = pd.DataFrame(all_player_data)

        # Calculate rates safely (using float division)
        processed_df['run_rate'] = np.where(
            processed_df['Total Balls'] > 0,
            (processed_df['Total Runs'].astype(float) * 100.0) / processed_df['Total Balls'],
            0.0
        )
        processed_df['out_rate'] = np.where(
            processed_df['Total Balls'] > 0,
            (processed_df['Total Outs'].astype(float) * 100.0) / processed_df['Total Balls'],
            0.0
        )

        player_options = sorted(processed_df['name'].unique())
        print(f"--- load_betting_data() successful, {len(processed_df)} rows processed, {len(player_options)} unique players ---")
        return processed_df

    except FileNotFoundError: # Already checked
        print(f"ERROR: Betting data file '{actual_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
         print(f"ERROR: Betting data file '{actual_path}' is empty.")
         return None
    except Exception as e:
        print(f"ERROR during Betting data loading/processing: {e}")
        traceback.print_exc()
        return None


# --- Load Data on Startup ---
print("="*50)
print("Starting Data Loading Sequence...")
print("="*50)

# Ensure 'data' directory exists
if not os.path.isdir('data'):
    print("ERROR: 'data' directory not found. Please create it and place CSV files inside.")
    # exit(1) # Optional: Force exit if data dir is missing

df_main_data = load_main_data()
df_tournament_data = load_tournament_data()
df_player_analysis_data = load_player_analysis_data() # Will look for 'data/app6.csv'
df_betting_data = load_betting_data() # Will look for 'data/style_data_with_start_date.csv'

print("="*50)
print("Data Loading Sequence Complete.")
print("="*50)

# --- Prepare Data for Stores (Serialization) ---
# [SERIALIZE_DF FUNCTION - KEEP AS IS - OMITTED FOR BREVITY]
def serialize_df(df):
    """Safely serializes a DataFrame to JSON, handling dates and potential issues."""
    if df is None:
        print("Warning: Attempted to serialize a None DataFrame.")
        return None
    if not isinstance(df, pd.DataFrame):
         print(f"Warning: Attempted to serialize a non-DataFrame object: {type(df)}")
         return None
    try:
        df_copy = df.copy()
        datetime_cols = df_copy.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
        if not datetime_cols.empty:
            for col in datetime_cols:
                 df_copy[col] = df_copy[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)

        nullable_int_cols = df_copy.select_dtypes(include=['Int64', 'Int32', 'Int16', 'Int8']).columns
        if not nullable_int_cols.empty:
            for col in nullable_int_cols:
                 df_copy[col] = df_copy[col].astype(float) # NaN becomes null

        # Handle lists/arrays specifically before replacing inf
        for col in df_copy.columns:
             # Check if *any* element in the column is a list or ndarray
             # Make sure lists/arrays don't contain np.inf themselves if that's possible
             if df_copy[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                 print(f"Note: Column '{col}' contains list/array types. Converting ndarrays to lists.")
                 df_copy[col] = df_copy[col].apply(lambda x: list(x) if isinstance(x, np.ndarray) else x)
                 # You might need more complex handling here if lists can contain np.inf/NaN

        # Replace inf/-inf AFTER handling potential lists containing them
        df_copy.replace([np.inf, -np.inf], [None, None], inplace=True)

        # Convert remaining object columns that might contain complex types to string as a fallback
        object_cols = df_copy.select_dtypes(include=['object']).columns
        for col in object_cols:
             # Be careful with this - it might convert lists/dicts to their string representation
             # Only apply if you suspect problematic object types other than str/list/dict
             # if df_copy[col].apply(lambda x: not isinstance(x, (str, list, dict, type(None)))).any():
             #    print(f"Warning: Converting object column '{col}' to string due to potential non-serializable types.")
             #    df_copy[col] = df_copy[col].astype(str)
             pass # Avoid aggressive conversion for now


        data_for_store = df_copy.to_json(orient='split', date_format='iso', default_handler=str)
        return data_for_store
    except Exception as e:
        print(f"ERROR: Failed to serialize DataFrame to JSON: {e}")
        traceback.print_exc()
        # Debugging problematic columns
        for col in df.columns: # Iterate original df columns
            try:
                col_data = df[[col]]
                serialize_df(col_data) # Test serialization on single column df
            except Exception as col_e:
                print(f"----> Serialization potentially failed on column: {col} ({df[col].dtype}). Error: {col_e}")
                try:
                    unique_types = df[col].apply(type).unique()
                    print(f"----> Unique types in column '{col}': {unique_types}")
                    # print(f"----> Sample values: {df[col].unique()[:10]}")
                except Exception as debug_e:
                    print(f"----> Could not debug column '{col}'. Error: {debug_e}")
        return None


print("\nSerializing data for stores...")
main_data_for_store = serialize_df(df_main_data)
tournament_data_for_store = serialize_df(df_tournament_data) # Includes Match Explorer data
player_analysis_data_for_store = serialize_df(df_player_analysis_data)
betting_data_for_store = serialize_df(df_betting_data)

# --- Initialize Dash App ---
# Ensure 'pages' directory exists
if not os.path.isdir('pages'):
    print("ERROR: 'pages' directory not found. Please create it and place your page layout files (e.g., 01_home.py) inside.")
    # exit(1) # Optional: Force exit if pages dir is missing

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder='pages', # Explicitly state the pages folder
    external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True, # Keep True for multi-page apps
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server # Expose server for deployment

# --- Define Layout Constants ---
SIDEBAR_WIDTH = "16rem"
CONTENT_PADDING = "2rem 1.5rem" # Padding inside the content area

# --- Sidebar Style ---
sidebar_style = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": SIDEBAR_WIDTH,
    "padding": "2rem 1rem", # Internal padding for the sidebar itself
    "background-color": "#f8f9fa", # Standard light background
    "overflow-y": "auto", # Allow sidebar scroll if needed
}

# --- Content Style ---
# Apply margin-left to push content to the right of the sidebar
content_style = {
    "marginLeft": SIDEBAR_WIDTH,
    "padding": CONTENT_PADDING, # Internal padding for the content area
    "overflowX": "hidden", # Prevent horizontal scroll in content area
}

# --- Sidebar Component ---
sidebar = dbc.Nav(
    [
         dbc.NavLink(
            [html.I(className="fas fa-home me-2"), html.Span("Home")],
            href="/", active="exact"
        ),
        html.Hr(),
        html.P("Analysis Pages", className="text-muted small fw-bold ps-3"),
        *[
            dbc.NavLink(
                 html.Span(page["name"]),
                 href=page["relative_path"],
                 active="exact"
            )
            for page in sorted(dash.page_registry.values(), key=lambda p: p['module'])
            if page["path"] != "/"
        ],
    ],
    vertical=True,
    pills=True,
    # No className or style needed here, applied via sidebar_style below
)

# --- Main App Layout ---
app.layout = dbc.Container(
    [
        # --- Data Stores ---
        dcc.Store(id='main-data-store', storage_type='memory', data=main_data_for_store),
        dcc.Store(id='tournament-data-store', storage_type='memory', data=tournament_data_for_store),
        dcc.Store(id='player-analysis-data-store', storage_type='memory', data=player_analysis_data_for_store),
        dcc.Store(id='betting-analyzer-data-store', storage_type='memory', data=betting_data_for_store),

        # --- Sidebar ---
        # Placed directly in the container, styled with position:fixed
        html.Div(sidebar, style=sidebar_style),

        # --- Page Content ---
        # Wrapped in a Div that has margin-left applied to avoid the sidebar
        html.Div(
            dash.page_container, # Where page layouts will be displayed
            style=content_style
        ),
    ],
    fluid=True, # Use fluid container for full width
    style={"padding": "0"} # Remove default container padding if it causes issues
)


# --- Run the App ---
if __name__ == '__main__':
    print("\n--- Data Loading & Serialization Summary ---")
    print(f"Main Data (total_data.csv): {'OK' if main_data_for_store else 'FAILED'}")
    print(f"Tournament Data (mw_overall.csv): {'OK' if tournament_data_for_store else 'FAILED'}")
    print(f"Player Analysis Data (app6.csv): {'OK' if player_analysis_data_for_store else 'FAILED'}")
    print(f"Betting Analyzer Data (style_data_with_start_date.csv): {'OK' if betting_data_for_store else 'FAILED'}")
    print("------------------------------------------")

    # Critical failure checks [KEEP AS IS - OMITTED FOR BREVITY]
    critical_failures = []
    if not tournament_data_for_store:
         critical_failures.append("Tournament Data (mw_overall.csv required by Match Explorer)")
    if not betting_data_for_store:
         critical_failures.append("Betting Analyzer Data (style_data_with_start_date.csv)")
    if not player_analysis_data_for_store:
         critical_failures.append("Player Analysis Data (app6.csv)")
    # if not main_data_for_store: critical_failures.append("Main Data (total_data.csv)")

    if critical_failures:
        print("\n" + "="*30 + " CRITICAL DATA FAILURE " + "="*30)
        print(f"Essential data failed to load for: {', '.join(critical_failures)}")
        print("The application cannot function correctly without this data. Please fix the data loading issues.")
        print("Common issues: Check file paths, file existence, CSV formatting, required columns, and read permissions.")
        print("="*81)
        # exit(1) # Uncomment to force exit on failure
        print("\n*** Proceeding despite critical failures (affected pages WILL NOT work correctly) ***")


    print("\nRegistered Pages:")
    if not dash.page_registry:
         print("  WARNING: No pages seem to be registered. Ensure pages are in the 'pages' directory and use dash.register_page().")
    else:
        try:
            page_files = [f for f in os.listdir('pages') if os.path.isfile(os.path.join('pages', f)) and f.endswith('.py') and not f.startswith('_')]
            if not page_files:
                 print("  WARNING: The 'pages' directory exists but contains no Python files (or only files starting with '_').")
        except FileNotFoundError:
             print("  ERROR: The 'pages' directory does not exist.")

        for page in sorted(dash.page_registry.values(), key=lambda p: p['module']):
            print(f"  - Name: {page.get('name', 'N/A')}, Path: {page.get('path', 'N/A')}, Module: {page.get('module', 'N/A')}")

    print("\nLaunching Multi-Page Dash Server...")
    app.run(debug=True, use_reloader=True) # use_reloader=False if data loading is slow/problematic

# --- END OF FILE multi_page_app.py ---