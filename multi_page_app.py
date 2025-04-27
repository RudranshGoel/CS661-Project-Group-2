# --- START OF FILE multi_page_app2.py ---

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np # Import numpy
import os
import traceback # Import traceback for detailed error logging

# --- Helper Function: Load Main Data (from total_data.csv) ---
# This function seems complex but likely okay for its purpose (Performance Analysis?)
# Keep as is unless specific errors point here. Add minor robustness checks.
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
        # Increased dtype specification for robustness (keep if effective)
        dtype_spec = {
            'match_id': str, 'season': str, 'start_date': str, 'venue': str, 'city': str,
            'match_type': str, 'player_team': str, 'opposition_team': str, 'toss_winner': str,
            'toss_decision': str, 'winner': str, 'win_by_runs': 'Int64', 'win_by_wickets': 'Int64',
            'result': str, 'result_details': str, 'umpire1': str, 'umpire2': str,
            'event_name': str, 'match_number': str, 'batting_team': str, 'bowling_team': str,
            'innings': 'Int64', 'over': 'Int64', 'delivery': 'Int64', 'batsman': str, 'non_striker': str,
            'bowler': str, 'runs_scored': 'Int64', 'runs_off_bat': 'Int64', 'extras': 'Int64',
            'wides': 'Int64', 'noballs': 'Int64', 'byes': 'Int64', 'legbyes': 'Int64', 'penalty': 'Int64',
            'balls_faced': 'Int64', 'fours_scored': 'Int64', 'sixes_scored': 'Int64',
            'dot_balls_as_batsman': 'Int64', 'strike_rate_batting': float, 'player_out': 'Int64',
            'out_kind': str, 'fielder': str, 'bowler_involved_in_out': str, 'balls_bowled': 'Int64',
            'runs_conceded': 'Int64', 'wickets_taken': 'Int64', 'bowled_done': 'Int64',
            'lbw_done': 'Int64', 'caught_done': 'Int64', 'stumped_done': 'Int64',
            'run_out_direct': 'Int64', 'run_out_throw': 'Int64', 'run_out_involved': 'Int64',
            'dot_balls_as_bowler': 'Int64', 'maidens': 'Int64', 'economy_rate': float,
            'strike_rate_bowling': float, 'bowling_average': float, 'catches_taken': 'Int64',
            'stumpings_done': 'Int64', 'name': str, 'bowling_style': str, 'role': str
        }
        # Consider removing dtype_spec if it causes issues and handle types post-load
        df = pd.read_csv(data_path, low_memory=False, dtype=dtype_spec)

        # --- Data Cleaning (Main Data - Keep relevant parts) ---
        # List all numeric columns potentially used by ANY page that might use this main data
        numeric_cols_main = [
            'runs_scored', 'balls_faced', 'wickets_taken', 'balls_bowled',
            'runs_conceded', 'bowled_done', 'lbw_done', 'player_out',
            'fours_scored', 'sixes_scored', 'runs_off_bat', 'extras',
            'wides', 'noballs', 'byes', 'legbyes', 'penalty', 'win_by_runs', 'win_by_wickets',
            'caught_done', 'stumped_done', 'run_out_direct', 'run_out_throw',
            'run_out_involved', 'dot_balls_as_bowler', 'maidens', 'catches_taken',
            'stumpings_done', 'innings', 'over', 'delivery', 'dot_balls_as_batsman'
            # Add others if needed by other pages using df_main_data
        ]
        for col in numeric_cols_main:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Use Int64 for columns that might have missing values but should be integer
                if col in ['win_by_runs', 'win_by_wickets', 'innings', 'over', 'delivery']:
                    df[col] = df[col].astype('Int64')
                else:
                    # For other numeric columns, fill NA with 0 and decide int/float
                    df[col] = df[col].fillna(0)
                    # Decide int vs float based on potential decimal need
                    if df[col].apply(lambda x: x % 1 == 0).all(): # Check if all are whole numbers
                        # Use appropriate integer size
                        if df[col].abs().max() < 2**31:
                            df[col] = df[col].astype(np.int32)
                        else:
                            df[col] = df[col].astype(np.int64)
                    else:
                        df[col] = df[col].astype(float) # Keep as float if decimals exist
            else:
                print(f"Warning (Main Data): Numeric column '{col}' not found. Skipping conversion.")

        # String/Categorical Cleaning
        if 'out_kind' in df.columns:
            df['out_kind'] = df['out_kind'].fillna('not out').astype(str).str.lower().str.strip()
        else:
             df['out_kind'] = 'not out' # Ensure column exists

        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        else:
             df['start_date'] = pd.NaT # Ensure column exists

        if 'match_type' in df.columns:
            ALLOWED_FORMATS = ["ODI", "T20", "Test", "T20I"] # T20I needed?
            df['match_type'] = df['match_type'].astype(str).str.upper() # Standardize case
            df = df[df["match_type"].isin(ALLOWED_FORMATS)].copy()
        # else: df['match_type'] = 'Unknown' # Maybe filter later instead.

        # Consolidate team name cleaning
        team_cols = ['player_team', 'opposition_team', 'batting_team', 'bowling_team', 'winner', 'toss_winner']
        team_replacements = {
            'U.A.E.': 'United Arab Emirates', 'UAE': 'United Arab Emirates',
            'P.N.G.': 'Papua New Guinea', 'PNG': 'Papua New Guinea',
            'USA': 'United States of America',
            'West Indies Cricket Board': 'West Indies',
            'Rising Pune Supergiant': 'Rising Pune Supergiants',
            'Delhi Daredevils': 'Delhi Capitals',
             # Add more corrections as needed
        }
        for team_col in team_cols:
             if team_col in df.columns:
                df[team_col] = df[team_col].astype(str).str.strip() # Ensure string before replace
                df[team_col] = df[team_col].replace(team_replacements)
                df[team_col] = df[team_col].fillna('Unknown') # Fill NaN *after* replace

        # Fill potentially missing string columns expected by pages
        str_cols_to_fill = ['venue', 'city', 'toss_decision', 'result', 'batsman', 'non_striker', 'bowler', 'fielder', 'bowler_involved_in_out', 'name', 'bowling_style', 'role', 'event_name']
        for col in str_cols_to_fill:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)
            else:
                print(f"Warning (Main Data): String column '{col}' not found. Creating with 'Unknown'.")
                df[col] = 'Unknown'

        if df.empty:
            print("Warning: Main DataFrame became empty after cleaning/filtering.")
            return None

        print(f"--- load_main_data() successful, final shape: {df.shape} ---")
        # print(f"DEBUG Main Data Columns: {df.columns.tolist()}")
        # print(df.info())
        return df

    except FileNotFoundError: # Already checked, but keep for redundancy
        print(f"ERROR: Main data file '{data_path}' not found.")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error during main data loading/processing: {e}")
        traceback.print_exc()
        return None


# --- Helper Function: Load Tournament Data (from mw_overall.csv) ---
# --- ENHANCED for Match Explorer requirements ---
def load_tournament_data():
    """
    Loads data from mw_overall.csv, performs cleaning, and adds columns
    needed for the Tournament Analysis and Match Explorer pages.
    """
    print("--- Running load_tournament_data() [from mw_overall.csv] ---")
    tournament_filename = 'mw_overall.csv'
    data_path = os.path.join('data', tournament_filename)
    print(f"Attempting to load tournament data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"ERROR: Tournament data file '{data_path}' not found.")
        return None
    try:
        # Load without strict dtypes initially to avoid load errors, handle types later
        df = pd.read_csv(data_path, low_memory=False)
        print(f"Tournament data initial load shape: {df.shape}")
        # print(f"Initial Columns: {df.columns.tolist()}")

        # --- Define ALL columns needed by Match Explorer Stats callback ---
        # Ensure these cover every column used in update_stats() in 10_match_performance.py
        required_cols_me = [
            'match_id', 'event_name', 'city', 'venue', 'winner', 'toss_winner', 'toss_decision',
            'batting_team', # Needed for filtering, grouping runs/wickets/dismissals
            'bowling_team', # Needed for grouping extras (will be derived)
            'runs_off_bat', 'player_dismissed', # Core stats
            'wides', 'noballs', 'byes', 'legbyes', 'penalty', # Extras
            'out_kind', # For dismissal breakdown
            'season', 'start_date', 'match_type', 'umpire1', 'umpire2' # Other potential context cols
            # Add any others if the stats callback uses them
        ]
        # Add columns specifically needed for preprocessing steps
        required_cols_pre = ['batting_team', 'match_id', 'city', 'venue']
        all_required_cols = list(set(required_cols_me + required_cols_pre))

        # --- Check for and Create Missing Columns ---
        missing_cols = [col for col in all_required_cols if col not in df.columns]
        if missing_cols:
            print(f"WARNING (Tournament Data): File '{tournament_filename}' is missing columns needed by Match Explorer: {missing_cols}.")
            # Create missing columns with default values based on expected type
            for col in missing_cols:
                # Guess type based on name (adjust as needed)
                if any(k in col for k in ['run', 'wide', 'noball', 'bye', 'legbye', 'penalty', 'wicket', 'dismissed']):
                    print(f"  -> Creating missing numeric column '{col}' with 0.")
                    df[col] = 0
                elif col == 'bowling_team': # We derive this later
                    print(f"  -> Column '{col}' will be derived.")
                    continue # Skip creation for now
                else: # Assume string/object for others
                    print(f"  -> Creating missing string column '{col}' with 'Unknown'.")
                    df[col] = 'Unknown'

        # --- Type Conversion and Basic Cleaning ---
        # Numeric Conversion (ensure all expected numeric cols are handled)
        numeric_cols_mw = ['runs_off_bat', 'wides', 'noballs', 'byes', 'legbyes', 'penalty', 'player_dismissed'] # player_dismissed needs care - is it ID or flag? Assuming flag/count here.
        for col in numeric_cols_mw:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                # Use appropriate integer type after filling NaNs
                if df[col].abs().max() < 2**31:
                     df[col] = df[col].astype(np.int32)
                else:
                     df[col] = df[col].astype(np.int64)
             # else: Handled by missing column creation above

        # String/Categorical Conversion & Filling
        str_cols_mw = ['event_name', 'winner', 'season', 'batting_team', 'match_id',
                       'city', 'venue', 'toss_winner', 'toss_decision', 'out_kind']
        team_replacements_mw = { # Use the same replacements as main data
            'U.A.E.': 'United Arab Emirates', 'UAE': 'United Arab Emirates',
            'P.N.G.': 'Papua New Guinea', 'PNG': 'Papua New Guinea',
            'USA': 'United States of America',
            'West Indies Cricket Board': 'West Indies',
            'Rising Pune Supergiant': 'Rising Pune Supergiants',
            'Delhi Daredevils': 'Delhi Capitals',
        }
        for col in str_cols_mw:
             if col in df.columns:
                 df[col] = df[col].astype(str).str.strip() # Ensure string before operations
                 if col in ['batting_team', 'winner', 'toss_winner']: # Apply team name cleaning
                     df[col] = df[col].replace(team_replacements_mw)
                 if col == 'out_kind':
                     # Normalize dismissal types more thoroughly if needed
                     df[col] = df[col].fillna('not out').str.lower().str.strip()
                     # Add replacements here if necessary (e.g., 'caught and bowled' -> 'caught_bowled')
                 else:
                     # Generic fillna for other string columns
                     df[col] = df[col].fillna('Unknown')
             # else: Handled by missing column creation above

        # --- Match Explorer Preprocessing ---
        print("Applying preprocessing for Match Explorer...")

        # 1. Map Cities to Countries (Ensure this is comprehensive)
        city_to_country = {
            # Keep the extensive dictionary from the original multi_page_app2.py
            'Dubai': 'United Arab Emirates', 'Sharjah': 'United Arab Emirates', 'Abu Dhabi': 'United Arab Emirates',
            'London': 'United Kingdom', 'Manchester': 'United Kingdom', 'Birmingham': 'United Kingdom', 'Cardiff': 'United Kingdom', 'Southampton': 'United Kingdom', 'Leeds': 'United Kingdom', 'Chester-le-Street': 'United Kingdom', 'Nottingham': 'United Kingdom', 'Bristol': 'United Kingdom', 'Taunton': 'United Kingdom', 'Hove': 'United Kingdom', "Lord's": 'United Kingdom',
            'Sydney': 'Australia', 'Melbourne': 'Australia', 'Adelaide': 'Australia', 'Perth': 'Australia', 'Brisbane': 'Australia', 'Hobart': 'Australia', 'Canberra': 'Australia', 'Geelong':'Australia', 'Launceston': 'Australia',
            'Mumbai': 'India', 'Delhi': 'India', 'Kolkata': 'India', 'Chennai': 'India', 'Bengaluru': 'India', 'Bangalore': 'India', 'Hyderabad': 'India', 'Mohali': 'India', 'Nagpur': 'India', 'Pune': 'India', 'Ahmedabad': 'India', 'Dharamsala': 'India', 'Visakhapatnam': 'India', 'Indore': 'India', 'Rajkot': 'India', 'Ranchi': 'India', 'Cuttack': 'India', 'Guwahati': 'India', 'Lucknow': 'India', 'Kanpur': 'India', 'Jaipur': 'India', 'Chandigarh':'India',
            'Cape Town': 'South Africa', 'Johannesburg': 'South Africa', 'Durban': 'South Africa', 'Centurion': 'South Africa', 'Port Elizabeth': 'South Africa', 'Gqeberha': 'South Africa', 'Paarl': 'South Africa', 'Bloemfontein': 'South Africa', 'East London': 'South Africa', 'Potchefstroom': 'South Africa', 'Kimberley': 'South Africa', 'Benoni': 'South Africa',
            'Auckland': 'New Zealand', 'Wellington': 'New Zealand', 'Christchurch': 'New Zealand', 'Hamilton': 'New Zealand', 'Napier': 'New Zealand', 'Dunedin': 'New Zealand', 'Mount Maunganui': 'New Zealand', 'Queenstown': 'New Zealand', 'Nelson': 'New Zealand',
            'Karachi': 'Pakistan', 'Lahore': 'Pakistan', 'Rawalpindi': 'Pakistan', 'Multan': 'Pakistan', 'Faisalabad': 'Pakistan',
            'Colombo': 'Sri Lanka', 'Kandy': 'Sri Lanka', 'Galle': 'Sri Lanka', 'Hambantota': 'Sri Lanka', 'Dambulla': 'Sri Lanka', 'Pallekele': 'Sri Lanka',
            'Chattogram': 'Bangladesh', 'Chittagong': 'Bangladesh', 'Dhaka': 'Bangladesh', 'Sylhet': 'Bangladesh', 'Mirpur': 'Bangladesh', 'Khulna': 'Bangladesh', 'Fatullah': 'Bangladesh',
            'Harare': 'Zimbabwe', 'Bulawayo': 'Zimbabwe', 'Kwekwe': 'Zimbabwe', 'Mutare': 'Zimbabwe',
            'Bridgetown': 'Barbados', 'Gros Islet': 'Saint Lucia', 'Port of Spain': 'Trinidad and Tobago', 'Kingston': 'Jamaica', 'Providence': 'Guyana', 'North Sound': 'Antigua and Barbuda', 'Basseterre': 'Saint Kitts and Nevis', 'Kingstown': 'Saint Vincent and the Grenadines', 'Roseau': 'Dominica', 'Lauderhill': 'United States',
            'Dublin': 'Ireland', 'Belfast': 'United Kingdom', 'Malahide': 'Ireland', 'Bready': 'United Kingdom',
            'Edinburgh': 'United Kingdom', 'Glasgow': 'United Kingdom', 'Aberdeen': 'United Kingdom',
            'Amstelveen': 'Netherlands', 'Rotterdam': 'Netherlands', 'The Hague': 'Netherlands',
            'Windhoek': 'Namibia', 'Nairobi': 'Kenya', 'Kampala': 'Uganda',
            'Muscat': 'Oman', 'Al Amerat': 'Oman', 'Kathmandu': 'Nepal', 'Kirtipur': 'Nepal',
            'Singapore': 'Singapore', 'Kuala Lumpur': 'Malaysia', 'Hong Kong': 'Hong Kong',
             # Add venue mappings if city is often missing/unreliable
             'Old Trafford': 'United Kingdom', # Example
        }
        if 'city' in df.columns:
            df['country'] = df['city'].map(city_to_country)
             # Attempt mapping via venue if city didn't map
            if 'venue' in df.columns:
                 df['country'] = df['country'].fillna(df['venue'].map(city_to_country))
            df['country'] = df['country'].fillna('Unknown') # Final fill for unmapped
            unmapped_rows = df[df['country'] == 'Unknown'].shape[0]
            if unmapped_rows > 0:
                 print(f"  - Warning (Tournament Data): {unmapped_rows} rows have unmapped cities/venues to countries.")
                 # print(df[df['country'] == 'Unknown'][['city', 'venue']].drop_duplicates().head(10)) # Debug: show some unmapped
        else:
            print("  - Warning (Tournament Data): 'city' column not found. Cannot map to countries.")
            df['country'] = 'Unknown'

        # 2. Calculate unique teams per match (More Robustly)
        print("  - Calculating unique teams per match...")
        if 'batting_team' in df.columns and 'match_id' in df.columns:
            # Ensure batting_team is clean and not NaN before grouping
            df['batting_team'] = df['batting_team'].astype(str).str.strip().replace('', 'Unknown Team').fillna('Unknown Team')

            # Group by match_id, get unique non-unknown teams, sort them
            match_teams_grouped = df[df['batting_team'] != 'Unknown Team'].groupby('match_id')['batting_team'].unique()
            match_teams_grouped = match_teams_grouped.apply(lambda x: sorted(list(set(x)))) # Ensure unique and sorted

            # Create DataFrame from this series
            match_teams_df = match_teams_grouped.reset_index()
            match_teams_df.rename(columns={'batting_team': 'match_teams_list'}, inplace=True)

            # Filter for matches where exactly two *different* teams were identified
            match_teams_df = match_teams_df[
                match_teams_df['match_teams_list'].apply(lambda x: isinstance(x, list) and len(x) == 2 and x[0] != x[1])
            ]
            print(f"  - Identified {len(match_teams_df)} matches with exactly two unique, different teams.")

            if not match_teams_df.empty:
                # Merge this back using LEFT JOIN to keep all original rows but add team list where available
                original_rows = df.shape[0]
                df = pd.merge(df, match_teams_df, on='match_id', how='left')
                # Fill NaN in 'match_teams_list' for matches not meeting the criteria
                # df['match_teams_list'] = df['match_teams_list'].fillna(pd.Series([[]]*len(df))) # Fill with empty list? Or keep NaN? Let's keep NaN for now.
                print(f"  - Merged match teams list. Shape remains: {df.shape}")
                rows_without_teams = df['match_teams_list'].isnull().sum()
                if rows_without_teams > 0:
                    print(f"  - Note: {rows_without_teams} rows do not have a valid 2-team list (e.g., single team entries, >2 teams).")
            else:
                 print("  - Warning (Tournament Data): No matches found with exactly two distinct teams. Team filters/bowling team derivation might fail.")
                 df['match_teams_list'] = pd.NA # Or np.nan, or empty list

        else:
            print("  - Warning (Tournament Data): Cannot calculate unique match teams ('batting_team' or 'match_id' missing).")
            df['match_teams_list'] = pd.NA # Add column with missing indicator

        # 3. Derive Bowling Team (using the 'match_teams_list')
        print("  - Deriving 'bowling_team' column...")
        def get_bowling_team_robust(row):
            match_teams = row['match_teams_list']
            batter = row['batting_team']
            # Check if match_teams is a valid list of 2
            if isinstance(match_teams, list) and len(match_teams) == 2:
                if batter == match_teams[0]: return match_teams[1]
                if batter == match_teams[1]: return match_teams[0]
            # Fallback if batting team isn't in the list or list is invalid
            return 'Unknown' # Assign 'Unknown'

        if 'match_teams_list' in df.columns and 'batting_team' in df.columns:
             df['bowling_team'] = df.apply(get_bowling_team_robust, axis=1)
             # Clean the derived bowling team names too
             df['bowling_team'] = df['bowling_team'].astype(str).str.strip().replace(team_replacements_mw).fillna('Unknown')
             unknown_bowling_teams = (df['bowling_team'] == 'Unknown').sum()
             if unknown_bowling_teams > 0:
                  print(f"  - Derived 'bowling_team' is 'Unknown' for {unknown_bowling_teams} rows (may include matches without 2 valid teams).")
        else:
             print("  - Warning (Tournament Data): Cannot derive 'bowling_team' ('match_teams_list' or 'batting_team' missing).")
             df['bowling_team'] = 'Unknown' # Ensure column exists

        # --- Final Check and Return ---
        if df.empty:
            print("Warning: Tournament DataFrame is empty after all processing.")
            return None

        print(f"--- load_tournament_data() successful, final shape: {df.shape} ---")
        # print(f"DEBUG Tournament Data Columns: {df.columns.tolist()}") # Optional Debug
        # print(df.info()) # Optional Debug
        return df

    except FileNotFoundError: # Already checked, keep for redundancy
        print(f"ERROR: Tournament data file '{data_path}' not found.")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error during tournament data loading/processing: {e}")
        traceback.print_exc()
        return None


# --- Helper Function: Load Player Analysis Data (from app6.csv) ---
# Keep the version from multi_page_app2.py as it seems okay.
# Add path check.
def load_player_analysis_data(filepath="data/app6.csv"):
    """Loads and preprocesses the cricket data specifically for the Player Analysis page."""
    print(f"--- Running load_player_analysis_data({filepath}) ---")
    actual_path = os.path.join('data', os.path.basename(filepath)) # Construct path relative to 'data' subdir
    if not os.path.exists(actual_path):
        print(f"ERROR: Player analysis data file '{actual_path}' not found.")
        return None
    try:
        df = pd.read_csv(actual_path)

        # Basic Cleaning & Type Conversion
        df['name'] = df['name'].astype(str).str.strip()
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')

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
                if df[col].apply(lambda x: x % 1 == 0).all(): # Check if all are whole numbers
                    if df[col].abs().max() < 2**31:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    df[col] = df[col].astype(float) # Keep as float
            else:
                 print(f"Warning (Player Analysis): Required numeric column '{col}' missing in {actual_path}. Filling with 0.")
                 df[col] = 0
                 missing_numeric.append(col)

        # Handle categorical/text columns
        cat_cols = ['out_kind', 'match_type', 'venue', 'city', 'match_id'] # Added match_id
        missing_cat = []
        for col in cat_cols:
            if col in df.columns:
                 if col == 'out_kind':
                     df[col] = df[col].fillna('not out').astype(str).str.lower().str.strip()
                 else:
                     df[col] = df[col].fillna('Unknown').astype(str).str.strip()
            else:
                print(f"Warning (Player Analysis): Required categorical column '{col}' missing in {actual_path}. Filling with 'Unknown'.")
                df[col] = 'Unknown'
                missing_cat.append(col)

        # Drop rows with invalid dates or missing player name
        df.dropna(subset=['start_date', 'name'], inplace=True)

        df.sort_values(by='start_date', inplace=True)

        if df.empty:
             print(f"Warning: Player analysis DataFrame became empty after cleaning/filtering {actual_path}.")
             return None

        print(f"--- load_player_analysis_data() successful, shape: {df.shape} ---")
        if missing_numeric or missing_cat:
            print(f"    -> Note: Missing columns filled: {missing_numeric + missing_cat}")
        # print(f"DEBUG Player Analysis Columns: {df.columns.tolist()}")
        return df

    except FileNotFoundError: # Already checked
        print(f"ERROR: Player analysis data file '{actual_path}' not found.")
        return None
    except Exception as e:
        print(f"ERROR: Error during player analysis data loading ({actual_path}): {e}")
        traceback.print_exc()
        return None


# --- Helper Function: Load Betting Analyzer Data ---
# Keep the version from multi_page_app2.py as it seems self-contained and correct for its purpose.
# Add path check.
def load_betting_data(file_path="data/style_data_with_start_date.csv"):
    """
    Loads and processes data for the Betting Analyzer page.
    Based on load_and_process_data from 08_pre_betting_analyzer.py.
    """
    print(f"--- Running load_betting_data({file_path}) ---")
    actual_path = os.path.join('data', os.path.basename(file_path)) # Construct path relative to 'data' subdir
    if not os.path.exists(actual_path):
         print(f"ERROR: Betting data file not found at '{actual_path}'.")
         return None

    try:
        df = pd.read_csv(actual_path)
        print(f"Betting Analyzer: Successfully read CSV '{actual_path}'. Shape: {df.shape}")

        required_cols = ['name', 'match_type', #'start_date', # Optional unless used for time filtering
                         'balls_against_spin', 'runs_against_spin', 'outs_against_spin',
                         'balls_against_right_fast', 'runs_against_right_fast', 'outs_against_right_fast',
                         'balls_against_left_fast', 'runs_against_left_fast', 'outs_against_left_fast']
        # Identify only the stat columns for processing
        stat_cols = [c for c in required_cols if c not in ['name', 'match_type', 'start_date']]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Betting data missing required columns: {', '.join(missing_cols)}")
            return None # Fail if core columns are missing

        # Basic Cleaning
        df['match_type'] = df['match_type'].fillna('Unknown').astype(str)
        df['name'] = df['name'].fillna('Unknown Player').astype(str)

        # Convert stats cols to numeric, coercing errors and filling NaN with 0
        for col in stat_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            # Ensure integer type
            if df[col].abs().max() < 2**31:
                 df[col] = df[col].astype(np.int32)
            else:
                 df[col] = df[col].astype(np.int64)

        # Aggregate raw stats per player/match_type (sums across all dates in the file)
        grouping_cols = ['name', 'match_type']
        agg_rules = {col: 'sum' for col in stat_cols}
        player_agg = df.groupby(grouping_cols, as_index=False).agg(agg_rules) # Use as_index=False

        # Reshape data: one row per player/match_type/bowling_type
        all_player_data = []
        bowling_types_config = {
            'Spin': ('balls_against_spin', 'runs_against_spin', 'outs_against_spin'),
            'Right Fast': ('balls_against_right_fast', 'runs_against_right_fast', 'outs_against_right_fast'),
            'Left Fast': ('balls_against_left_fast', 'runs_against_left_fast', 'outs_against_left_fast'),
        }

        for _, row in player_agg.iterrows():
            for bowling_label, (balls_col, runs_col, outs_col) in bowling_types_config.items():
                 # Only add rows if balls > 0 for that bowling type
                 if row[balls_col] > 0:
                     all_player_data.append({
                        'name': row['name'], 'match_type': row['match_type'],
                        'Bowling Type': bowling_label,
                        'Total Runs': row[runs_col],
                        'Total Balls': row[balls_col],
                        'Total Outs': row[outs_col]
                     })

        if not all_player_data:
            print("WARNING: Betting data: No valid player/bowling type combinations after aggregation (all balls <= 0?).")
            return pd.DataFrame(columns=['name', 'match_type', 'Bowling Type', 'Total Runs', 'Total Balls', 'Total Outs', 'run_rate', 'out_rate'])

        processed_df = pd.DataFrame(all_player_data)

        # Calculate rates safely
        processed_df['run_rate'] = np.where(
            processed_df['Total Balls'] > 0,
            (processed_df['Total Runs'] * 100.0) / processed_df['Total Balls'], # Ensure float division
            0.0
        )
        processed_df['out_rate'] = np.where(
            processed_df['Total Balls'] > 0,
            (processed_df['Total Outs'] * 100.0) / processed_df['Total Balls'], # Ensure float division
            0.0 # Or np.inf if preferred when balls=0, but 0 is safer for plotting
        )
        # Optional: Balls per dismissal
        # processed_df['balls_per_dismissal'] = np.where(
        #      processed_df['Total Outs'] > 0,
        #      processed_df['Total Balls'] / processed_df['Total Outs'],
        #      np.inf # Infinite balls per dismissal if never out
        # )

        player_options = sorted(processed_df['name'].unique())
        print(f"--- load_betting_data() successful, {len(processed_df)} rows processed, {len(player_options)} unique players ---")
        return processed_df

    except FileNotFoundError: # Already checked
        print(f"ERROR: Betting data file '{actual_path}' not found.")
        return None
    except Exception as e:
        print(f"ERROR during Betting data loading/processing: {e}")
        traceback.print_exc()
        return None

# --- Load Data on Startup ---
print("="*50)
print("Starting Data Loading Sequence...")
print("="*50)

df_main_data = load_main_data()
df_tournament_data = load_tournament_data()
df_player_analysis_data = load_player_analysis_data()
df_betting_data = load_betting_data()

print("="*50)
print("Data Loading Sequence Complete.")
print("="*50)

# --- Prepare Data for Stores (Serialization) ---
# Keep the robust serialize_df from multi_page_app2.py
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
        # Convert datetime columns to ISO format strings
        datetime_cols = df_copy.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
        if not datetime_cols.empty:
            # print(f"Serializing datetime columns: {', '.join(datetime_cols)}")
            for col in datetime_cols:
                 # Use ISO format, which pd.read_json handles well
                 # Handle NaT explicitly before formatting
                 df_copy[col] = df_copy[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
        # else: print("No datetime columns found for serialization.")

        # Convert Pandas nullable integers (Int64 etc.) to float for JSON compatibility
        # JSON standard doesn't have a native NaN/NA for integers
        nullable_int_cols = df_copy.select_dtypes(include=['Int64', 'Int32', 'Int16', 'Int8']).columns
        if not nullable_int_cols.empty:
            # print(f"Converting nullable Int columns to float for JSON: {', '.join(nullable_int_cols)}")
            for col in nullable_int_cols:
                 # Convert to float; <NA> becomes np.nan which JSON handles as 'null'
                 df_copy[col] = df_copy[col].astype(float)

        # Convert non-serializable types like 'Period' if they exist
        # period_cols = df_copy.select_dtypes(include=['period']).columns
        # if not period_cols.empty: ... convert to str ...

        # Handle potential list/array types in columns if they exist (e.g., match_teams_list)
        # They should serialize okay, but check if errors occur.
        for col in df_copy.columns:
            if df_copy[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                print(f"Note: Column '{col}' contains list/array types.")
                # Ensure they are basic lists if numpy arrays cause issues (unlikely)
                # df_copy[col] = df_copy[col].apply(lambda x: list(x) if isinstance(x, np.ndarray) else x)

        # Use 'split' orientation
        # Handle potential infinity values if using np.inf anywhere
        df_copy.replace([np.inf, -np.inf], [None, None], inplace=True) # Replace inf with None (null in JSON)

        data_for_store = df_copy.to_json(orient='split', date_format='iso', default_handler=str)
        # print("Serialization successful.")
        return data_for_store
    except Exception as e:
        print(f"ERROR: Failed to serialize DataFrame to JSON: {e}")
        traceback.print_exc()
        # Optionally, try to identify the problematic column:
        for col in df_copy.columns:
            try:
                df_copy[[col]].to_json(orient='split', date_format='iso', default_handler=str)
            except Exception as col_e:
                print(f"----> Failed on column: {col} ({df_copy[col].dtype}). Error: {col_e}")
                # print(f"----> Sample values: {df_copy[col].unique()[:10]}") # Show some unique values
        return None


print("\nSerializing data for stores...")
main_data_for_store = serialize_df(df_main_data)
tournament_data_for_store = serialize_df(df_tournament_data) # Includes Match Explorer data
player_analysis_data_for_store = serialize_df(df_player_analysis_data)
betting_data_for_store = serialize_df(df_betting_data)

# --- Initialize Dash App ---
app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder='pages', # Explicitly state the pages folder
    external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True, # Keep True for multi-page apps
    # Add meta_tags for responsiveness if needed
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server # Expose server for deployment

# --- Sidebar ---
sidebar = dbc.Nav(
    [
         dbc.NavLink(
            [html.I(className="fas fa-home me-2"), html.Span("Home")],
            href="/", active="exact"
        ),
        html.Hr(),
        html.P("Analysis Pages", className="text-muted small fw-bold ps-3"),
        # Dynamically create links for all registered pages EXCEPT home
        # Sort pages based on the number prefix in their filename (module name)
        *[
            dbc.NavLink(
                 html.Span(page["name"]), # Use the 'name' from register_page
                 href=page["relative_path"],
                 active="exact"
            )
            # Sort by module name (which includes the number prefix)
            # Ensure pages are in a 'pages' subdirectory and named like '01_home.py', '02_something.py'
            for page in sorted(dash.page_registry.values(), key=lambda p: p['module'])
            if page["path"] != "/" # Exclude Home page link from this section
        ],
    ],
    vertical=True, pills=True, className="bg-light",
    style={'position': 'fixed', 'top': 0, 'left': 0, 'bottom': 0, 'width': '16rem', 'padding': '2rem 1rem', 'overflowY': 'auto'} # Changed overflow-y to overflowY
)


# --- Main App Layout ---
app.layout = dbc.Container([
    # --- Add ALL Store components here ---
    # Ensure IDs match those used in pages
    dcc.Store(id='main-data-store', storage_type='memory', data=main_data_for_store),
    dcc.Store(id='tournament-data-store', storage_type='memory', data=tournament_data_for_store),
    dcc.Store(id='player-analysis-data-store', storage_type='memory', data=player_analysis_data_for_store),
    dcc.Store(id='betting-analyzer-data-store', storage_type='memory', data=betting_data_for_store),

    dbc.Row([
        dbc.Col(sidebar, width=2, style={'padding': '0'}), # Sidebar column
        dbc.Col(
            dash.page_container, # <<< Where page layouts will be displayed
            style={'marginLeft': '16rem', 'padding': '2rem 1rem', 'overflowX': 'hidden'} # Ensure margin matches sidebar, prevent horizontal scroll
        )
    ],
    # className="g-0" # Optional: remove row gutters
    )
], fluid=True) # Use fluid container for full width


# --- Run the App ---
if __name__ == '__main__':
    print("\n--- Data Loading & Serialization Summary ---")
    print(f"Main Data (total_data.csv): {'OK' if main_data_for_store else 'FAILED'}")
    print(f"Tournament Data (mw_overall.csv + Preprocessing): {'OK' if tournament_data_for_store else 'FAILED'}")
    print(f"Player Analysis Data (app6.csv): {'OK' if player_analysis_data_for_store else 'FAILED'}")
    print(f"Betting Analyzer Data (style_data_with_start_date.csv): {'OK' if betting_data_for_store else 'FAILED'}")
    print("------------------------------------------")

    # Add critical checks - Ensure the app doesn't start if essential data is missing
    critical_failures = []
    if not tournament_data_for_store:
         critical_failures.append("Tournament Data (mw_overall.csv)")
         print("\nCRITICAL WARNING: Tournament data (mw_overall.csv) could not be loaded/processed.")
         print("  -> Ensure 'data/mw_overall.csv' exists, is readable, and contains expected columns.")
         print("  -> The 'Match Explorer' page WILL LIKELY FAIL.")

    if not betting_data_for_store:
         critical_failures.append("Betting Analyzer Data (style_data_with_start_date.csv)")
         print("\nCRITICAL WARNING: Betting Analyzer data (style_data_with_start_date.csv) could not be loaded/processed.")
         print("  -> Ensure 'data/style_data_with_start_date.csv' exists, is readable, and contains expected columns.")
         print("  -> The 'Pre-Betting Analyzer' page WILL LIKELY FAIL.")

    # Add checks for other pages if they are critical
    # if not main_data_for_store: critical_failures.append("Main Data (total_data.csv)")
    # if not player_analysis_data_for_store: critical_failures.append("Player Analysis Data (app6.csv)")

    if critical_failures:
        print("\n" + "="*30 + " FATAL ERROR " + "="*30)
        print(f"Essential data failed to load: {', '.join(critical_failures)}")
        print("The application cannot start correctly. Please fix the data loading issues.")
        print("="*73)
        # Exit if critical data is missing
        # exit(1) # Uncomment to force exit on failure
        print("\nProceeding despite failures (some pages may not work)...") # Or allow proceeding with warning


    print("\nRegistered Pages:")
    if not dash.page_registry:
         print("  WARNING: No pages seem to be registered. Ensure pages are in the 'pages' directory and use dash.register_page().")
    else:
        for page in sorted(dash.page_registry.values(), key=lambda p: p['module']):
            print(f"  - Name: {page.get('name', 'N/A')}, Path: {page.get('path', 'N/A')}, Module: {page.get('module', 'N/A')}")

    print("\nLaunching Multi-Page Dash Server...")
    # Set use_reloader=False if debug mode causes issues with large data loading twice
    # Set host='0.0.0.0' to make accessible on network
    app.run(debug=True, use_reloader=True) # debug=True is helpful for development

# --- END OF FILE multi_page_app2.py ---