# pages/05_match_explorer.py

import dash
from dash import dcc, html, Input, Output, callback, State, ctx, Patch
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import traceback # For detailed error printing
import sys # Keep for potential use, but remove sys.exit()

# --- Register Page ---
dash.register_page(
    __name__,
    name='Match Explorer',           # Name for the sidebar
    path='/match-explorer',         # URL path
    title='Cricket Match Explorer'  # Browser tab title
)

# --- Helper Function: Deserialize Data ---
def deserialize_data(stored_data):
    """Deserializes JSON data (from dcc.Store) back into a Pandas DataFrame."""
    if stored_data is None:
        # print("Warning: No data found in store (match explorer).") # Less verbose
        return None
    try:
        df = pd.read_json(stored_data, orient='split')
        # Ensure required columns have correct types after deserialization if needed
        if df.empty:
            # print("Warning: Deserialized DataFrame is empty (match explorer).") # Less verbose
            return df
        return df
    except Exception as e:
        print(f"ERROR: Failed to deserialize data from store (match explorer): {e}")
        # traceback.print_exc() # Optional: Keep for debugging
        return None

# --- REMOVED Data Loading and Preprocessing ---
# --- REMOVED Get unique values for dropdowns ---
# --- REMOVED App Initialization ---

# --- App Layout ---
layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Cricket Match Explorer"), width=12)),
    dbc.Row([
        dbc.Col([
            html.Label("Select Event:", className="fw-bold"),
            dcc.Dropdown(
                id='me-event-dropdown', # Consider prefixing IDs (e.g., 'me-' for Match Explorer)
                placeholder="Select an event...", searchable=True, clearable=False
            )], width=4),
        dbc.Col([
            html.Label("Select Team 1:", className="fw-bold"),
            dcc.Dropdown(id='me-team1-dropdown', placeholder="Select team 1...", searchable=True, clearable=False)
            ], width=4),
        dbc.Col([
            html.Label("Select Team 2:", className="fw-bold"),
            dcc.Dropdown(id='me-team2-dropdown', placeholder="Select team 2...", searchable=True, clearable=False)
            ], width=4),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            html.Label("Countries Played:", className="fw-bold"),
            dcc.Graph(
                id='me-world-map',
                figure=go.Figure(layout=go.Layout(title="Select Event & Teams to see Map", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False)))
            )], width=12)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([html.Div(id='me-selected-country-stadium-info', children=[html.Div("Click on a country on the map to see stadiums.", className="text-muted")])], width=12)
    ], className="mb-3"),
    # Stats Section
    dbc.Row([
        dbc.Col(html.H3("Stadium Statistics", className="text-center"), width=12, id='me-stats-header', style={'display': 'none'})
    ]),
    # Row 1: Wins & Toss
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id='me-win-loss-chart')) , md=6), # Added Loading
        dbc.Col(dcc.Loading(dcc.Graph(id='me-toss-chart'))     , md=6)  # Added Loading
    ], className="mb-3", id='me-stats-row-1', style={'display': 'none'}),
    # Row 3: Runs & Wickets
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id='me-runs-chart'))         , md=6), # Added Loading
        dbc.Col(dcc.Loading(dcc.Graph(id='me-wickets-lost-chart')) , md=6)  # Added Loading
    ], className="mb-3", id='me-stats-row-3', style={'display': 'none'}),
    # Row 5: Runs Distribution & Scatter Plot
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id='me-runs-distribution-chart')), md=6), # Added Loading
        dbc.Col(dcc.Loading(dcc.Graph(id='me-scatter-runs-wickets'))   , md=6)  # Added Loading
    ], className="mb-3", id='me-stats-row-5', style={'display': 'none'}),
    # Combined Header for Sunbursts (Now just Extras)
    dbc.Row([
        dbc.Col(html.H4("Hierarchical Breakdowns", className="text-center"), width=12) # Consider renaming H4 if desired
    ], id='me-stats-sunburst-header', style={'display': 'none'}), # New Header Row
    # Modified Sunburst Row (Only Extras)
    dbc.Row([
        # REMOVED: dbc.Col(dcc.Loading(dcc.Graph(id='me-dismissal-sunburst-chart')), md=6),
        dbc.Col(dcc.Loading(dcc.Graph(id='me-extras-sunburst-chart'))    , md=12)  # Changed width to 12
    ], className="mb-3", id='me-stats-row-sunbursts', style={'display': 'none'}),
    # Store intermediate data - Use unique IDs if necessary
    dcc.Store(id='me-filtered-matches-ids'),
    dcc.Store(id='me-filtered-country-matches-ids'),
    dcc.Store(id='me-filtered-venue-matches-ids'),
    # NOTE: We will read data from 'tournament-data-store' defined in multi_page_app.py
], fluid=True)


# --- Callbacks ---

# Callback 0: Populate Event Dropdown using stored data
@callback(
    Output('me-event-dropdown', 'options'),
    Input('tournament-data-store', 'data'), # INPUT: Use the central store
    prevent_initial_call=False # Populate on load
)
def update_event_dropdown(stored_tournament_data):
    """Populates the event dropdown based on the centrally loaded tournament data."""
    df_tournament = deserialize_data(stored_tournament_data)

    if df_tournament is None or df_tournament.empty or 'event_name' not in df_tournament.columns:
        print("Match Explorer: Tournament data not available or missing 'event_name'.")
        return []

    unique_events = df_tournament['event_name'].dropna().unique()
    unique_events.sort()
    event_options = [{'label': i, 'value': i} for i in unique_events]
    return event_options

# Callback 1: Update ONLY Team 1 Dropdown based on Event AND stored data
@callback(
    Output('me-team1-dropdown', 'options'),
    Output('me-team1-dropdown', 'value'),
    Input('me-event-dropdown', 'value'),
    Input('tournament-data-store', 'data') # INPUT: Use the central store
)
def update_team1_options(selected_event, stored_tournament_data):
    if not selected_event:
        return [], None

    df_tournament = deserialize_data(stored_tournament_data)
    if df_tournament is None or df_tournament.empty or 'event_name' not in df_tournament.columns or 'match_teams_list' not in df_tournament.columns:
         print("Match Explorer: Can't update Team 1 - Data/columns missing.")
         return [], None

    event_df = df_tournament[df_tournament['event_name'] == selected_event]
    if event_df.empty:
        return [], None

    all_teams_in_event = set()
    valid_teams = event_df['match_teams_list'].dropna()
    for team_list in valid_teams:
        if isinstance(team_list, (list, tuple)):
             all_teams_in_event.update(team_list)

    sorted_teams = sorted(list(all_teams_in_event))
    team_options = [{'label': team, 'value': team} for team in sorted_teams]
    return team_options, None

# Callback 2: Update Team 2 Dropdown options based on Event, Team 1, AND stored data
@callback(
    Output('me-team2-dropdown', 'options'),
    Output('me-team2-dropdown', 'value'),
    Input('me-event-dropdown', 'value'),
    Input('me-team1-dropdown', 'value'),
    Input('tournament-data-store', 'data'), # INPUT: Use the central store
    prevent_initial_call=True
)
def update_team2_options_and_value(selected_event, selected_team1, stored_tournament_data):
    if not selected_event:
        return [], None

    df_tournament = deserialize_data(stored_tournament_data)
    if df_tournament is None or df_tournament.empty or 'event_name' not in df_tournament.columns or 'match_teams_list' not in df_tournament.columns:
        print("Match Explorer: Can't update Team 2 - Data/columns missing.")
        return [], None

    event_df = df_tournament[df_tournament['event_name'] == selected_event]
    if event_df.empty:
        return [], None

    all_teams_in_event = set()
    valid_teams = event_df['match_teams_list'].dropna()
    for team_list in valid_teams:
       if isinstance(team_list, (list, tuple)):
            all_teams_in_event.update(team_list)

    sorted_teams = sorted(list(all_teams_in_event))
    full_team_options = [{'label': team, 'value': team} for team in sorted_teams]

    if selected_team1:
        filtered_options = [opt for opt in full_team_options if opt['value'] != selected_team1]
        return filtered_options, None
    else:
        return full_team_options, None

# Callback 3: Filter match IDs based on event, teams, AND stored data
@callback(
    Output('me-filtered-matches-ids', 'data'), # Use prefixed ID
    Input('me-event-dropdown', 'value'),
    Input('me-team1-dropdown', 'value'),
    Input('me-team2-dropdown', 'value'),
    Input('tournament-data-store', 'data') # INPUT: Use the central store
)
def filter_matches_by_event_teams(event, team1, team2, stored_tournament_data):
    if not all([event, team1, team2]):
        return []
    if team1 == team2:
        return []

    df_tournament = deserialize_data(stored_tournament_data)
    if df_tournament is None or df_tournament.empty or 'event_name' not in df_tournament.columns or 'match_teams_list' not in df_tournament.columns or 'match_id' not in df_tournament.columns:
        print("Match Explorer: Can't filter matches - Data/columns missing.")
        return []

    event_df = df_tournament[df_tournament['event_name'] == event]
    if event_df.empty: return []

    selected_teams_set = {team1, team2}
    final_match_ids = event_df[
        event_df['match_teams_list'].apply(lambda lst: isinstance(lst, (list, tuple)) and set(lst) == selected_teams_set)
    ]['match_id'].unique().tolist()
    return final_match_ids

# Callback 4: Update World Map (Depends on filtered IDs, reads data again from store)
@callback(
    Output('me-world-map', 'figure'),            # Use prefixed ID
    Input('me-filtered-matches-ids', 'data'),  # Use prefixed ID
    State('me-team1-dropdown', 'value'),       # Use prefixed ID
    State('me-team2-dropdown', 'value'),       # Use prefixed ID
    State('tournament-data-store', 'data'),    # STATE: Get data store content
    prevent_initial_call=True
)
def update_world_map(filtered_match_ids, team1, team2, stored_tournament_data):
    default_fig = go.Figure(layout=go.Layout(title="Select Event & Teams to see Map", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False)))
    if not filtered_match_ids:
        return default_fig

    df_tournament = deserialize_data(stored_tournament_data)
    if df_tournament is None or df_tournament.empty or not all(c in df_tournament.columns for c in ['match_id', 'country', 'city', 'venue']):
        print("Match Explorer: Cannot update map - Data/columns missing.")
        return default_fig

    map_df_all = df_tournament.loc[df_tournament['match_id'].isin(filtered_match_ids), ['match_id', 'country', 'city', 'venue']].copy()
    map_df_unique_matches = map_df_all.drop_duplicates(subset=['match_id'])
    map_df_unique_matches = map_df_unique_matches.dropna(subset=['country'])

    if map_df_unique_matches.empty:
         t1_label = team1 if team1 else "Team 1"
         t2_label = team2 if team2 else "Team 2"
         return go.Figure(layout=go.Layout(title=f"No matches found in mappable locations for {t1_label} vs {t2_label}", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))

    country_counts = map_df_unique_matches['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'match_count']

    t1_label = team1 if team1 else "Team 1"
    t2_label = team2 if team2 else "Team 2"
    fig = px.scatter_geo(
        country_counts, locations="country", locationmode='country names', size="match_count",
        hover_name="country", hover_data={'country': True, 'match_count': True}, custom_data=['country'],
        projection="natural earth", title=f"Countries Where {t1_label} Played {t2_label}", size_max=40
    )
    fig.update_layout(
        geo=dict(bgcolor='rgba(0,0,0,0)', landcolor='rgb(217, 217, 217)', subunitcolor='rgb(255, 255, 255)',
                 showcoastlines=True, coastlinecolor="RebeccaPurple", showland=True, showocean=True, oceancolor="LightBlue"),
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    return fig

# Callback 5: Update Stadium Dropdown (Depends on map click, filtered IDs, reads data again)
@callback(
    Output('me-selected-country-stadium-info', 'children'), # Use prefixed ID
    Output('me-filtered-country-matches-ids', 'data'),     # Use prefixed ID
    Input('me-world-map', 'clickData'),                   # Use prefixed ID
    State('me-filtered-matches-ids', 'data'),             # Use prefixed ID
    State('tournament-data-store', 'data'),               # STATE: Get data store content
    prevent_initial_call=True
)
def update_stadium_dropdown(clickData, filtered_match_ids, stored_tournament_data):
    no_stadium_info = html.Div("Click on a country on the map to see stadiums.", className="text-muted")
    if not clickData or not filtered_match_ids:
        return no_stadium_info, []

    df_tournament = deserialize_data(stored_tournament_data)
    if df_tournament is None or df_tournament.empty or not all(c in df_tournament.columns for c in ['match_id', 'country', 'venue']):
        print("Match Explorer: Cannot update stadium dropdown - Data/columns missing.")
        return html.Div("Error: Data required for stadiums is missing.", className="text-danger"), []

    try:
        clicked_country = clickData['points'][0].get('customdata', [None])[0] or clickData['points'][0].get('location')
        if clicked_country is None:
             print("Error: Cannot determine clicked country from clickData.")
             return html.Div("Error: Could not identify clicked country.", className="text-danger"), []
        if not isinstance(clicked_country, str): clicked_country = str(clicked_country)
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error parsing clickData: {e}")
        return html.Div("Error processing map click.", className="text-danger"), []

    country_filtered_df = df_tournament[
        df_tournament['match_id'].isin(filtered_match_ids) & (df_tournament['country'] == clicked_country)
    ].copy()

    if country_filtered_df.empty:
         return html.Div(f"No matches found in {clicked_country} for the selected teams/event.", className="text-warning"), []

    unique_venues = country_filtered_df['venue'].dropna().unique()
    unique_venues.sort()

    if not unique_venues.any():
        return html.Div(f"No specific stadiums found in {clicked_country} for these matches.", className="text-warning"), country_filtered_df['match_id'].unique().tolist()

    venue_options = [{'label': venue, 'value': venue} for venue in unique_venues]
    country_match_ids = country_filtered_df['match_id'].unique().tolist()

    output_div = html.Div([
        html.H4(f"Stadiums in {clicked_country}"),
        html.Label("Select Stadium:", className="fw-bold"),
        dcc.Dropdown(id='me-venue-dropdown', options=venue_options, placeholder="Select a stadium...", clearable=False, searchable=True, value=None)
    ])
    return output_div, country_match_ids

# Callback 6: Filter by venue (Depends on country IDs, reads data again)
@callback(
    Output('me-filtered-venue-matches-ids', 'data'), # Use prefixed ID
    Input('me-venue-dropdown', 'value'),            # Use prefixed ID (Dynamically created)
    State('me-filtered-country-matches-ids', 'data'), # Use prefixed ID
    State('tournament-data-store', 'data'),         # STATE: Get data store content
    prevent_initial_call=True
)
def filter_by_venue(selected_venue, country_match_ids, stored_tournament_data):
    triggered_id = ctx.triggered_id
    if triggered_id != 'me-venue-dropdown' or not selected_venue or not country_match_ids:
        return []

    df_tournament = deserialize_data(stored_tournament_data)
    if df_tournament is None or df_tournament.empty or not all(c in df_tournament.columns for c in ['match_id', 'venue']):
        print("Match Explorer: Cannot filter by venue - Data/columns missing.")
        return []

    venue_filtered_df = df_tournament[
        df_tournament['match_id'].isin(country_match_ids) & (df_tournament['venue'] == selected_venue)
    ]
    venue_match_ids = venue_filtered_df['match_id'].unique().tolist()
    return venue_match_ids


# Callback 7: Generate Statistics (Depends on venue IDs, reads data again)
@callback(
    # Use prefixed IDs for all outputs
    Output('me-stats-header', 'style'),
    Output('me-stats-row-1', 'style'),
    Output('me-stats-row-3', 'style'),
    Output('me-stats-row-5', 'style'),
    Output('me-stats-sunburst-header', 'style'),
    Output('me-stats-row-sunbursts', 'style'),
    Output('me-win-loss-chart', 'figure'), Output('me-toss-chart', 'figure'),
    Output('me-runs-chart', 'figure'), Output('me-wickets-lost-chart', 'figure'),
    Output('me-runs-distribution-chart', 'figure'), Output('me-scatter-runs-wickets', 'figure'),
    # Output('me-dismissal-sunburst-chart', 'figure'), # REMOVED THIS OUTPUT
    Output('me-extras-sunburst-chart', 'figure'),
    Input('me-filtered-venue-matches-ids', 'data'), # Use prefixed ID
    State('me-team1-dropdown', 'value'),          # Use prefixed ID
    State('me-team2-dropdown', 'value'),          # Use prefixed ID
    State('tournament-data-store', 'data'),       # STATE: Get data store content
    prevent_initial_call=True
)
def update_stats(venue_match_ids, team1, team2, stored_tournament_data):
    # Define initial state for figures and styles
    no_data_layout = go.Layout(title="No data available", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False))
    no_data_fig = go.Figure(layout=no_data_layout)
    hidden_style = {'display': 'none'}
    visible_style = {'display': 'flex'}

    # Initialize all figures to no_data_fig
    figures = {
        'win_loss': no_data_fig, 'toss': no_data_fig,
        'runs': no_data_fig, 'wickets_lost': no_data_fig,
        'runs_dist': no_data_fig, 'scatter': no_data_fig,
        # 'sunburst': no_data_fig, # REMOVED THIS ENTRY
        'extras_sunburst': no_data_fig
    }
    # Initial styles are hidden
    styles = {
        'header': hidden_style, 'row1': hidden_style, 'row3': hidden_style,
        'row5': hidden_style, 'sunburst_header': hidden_style, 'sunburst_row': hidden_style
    }

    # --- Early exit conditions ---
    if not venue_match_ids or not team1 or not team2:
        return styles['header'], styles['row1'], styles['row3'], styles['row5'], styles['sunburst_header'], styles['sunburst_row'], \
               figures['win_loss'], figures['toss'], figures['runs'], figures['wickets_lost'], \
               figures['runs_dist'], figures['scatter'], figures['extras_sunburst'] # REMOVED figures['sunburst']

    df_tournament = deserialize_data(stored_tournament_data)
    if df_tournament is None or df_tournament.empty:
        print("Match Explorer Update Stats: Exiting early - data store empty or deserialization failed.")
        err_layout = go.Layout(title="Error: Data unavailable", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        err_fig = go.Figure(layout=err_layout)
        figures = {k: err_fig for k in figures}
        return styles['header'], styles['row1'], styles['row3'], styles['row5'], styles['sunburst_header'], styles['sunburst_row'], \
               figures['win_loss'], figures['toss'], figures['runs'], figures['wickets_lost'], \
               figures['runs_dist'], figures['scatter'], figures['extras_sunburst'] # REMOVED figures['sunburst']

    # --- Data Preparation for Stats ---
    stats_df = df_tournament[df_tournament['match_id'].isin(venue_match_ids)].copy()

    # Check if essential columns for stats exist
    # Removed 'out_kind' from this list as it was only used for the removed chart
    required_stat_cols = ['venue', 'match_id', 'winner', 'toss_winner', 'toss_decision',
                          'wides', 'noballs', 'byes', 'legbyes', 'penalty', 'runs_off_bat',
                          'player_dismissed', 'bowling_team', 'batting_team']
    missing_stat_cols = [col for col in required_stat_cols if col not in stats_df.columns]
    if missing_stat_cols:
        print(f"Match Explorer Update Stats: Exiting - Missing required columns in filtered data: {missing_stat_cols}")
        err_layout = go.Layout(title=f"Error: Missing data ({', '.join(missing_stat_cols)})", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        err_fig = go.Figure(layout=err_layout)
        figures = {k: err_fig for k in figures}
        return styles['header'], styles['row1'], styles['row3'], styles['row5'], styles['sunburst_header'], styles['sunburst_row'], \
               figures['win_loss'], figures['toss'], figures['runs'], figures['wickets_lost'], \
               figures['runs_dist'], figures['scatter'], figures['extras_sunburst'] # REMOVED figures['sunburst']

    # --- Derive venue_name ---
    venue_name = "Selected Venue"
    if not stats_df.empty and 'venue' in stats_df.columns:
        first_venue = stats_df['venue'].dropna().iloc[0] if not stats_df['venue'].dropna().empty else None
        if first_venue: venue_name = first_venue

    if stats_df.empty:
        print(f"Match Explorer Update Stats: Filtered stats_df is empty for venue IDs: {venue_match_ids}")
        no_match_layout = go.Layout(title=f"No match data found for {venue_name}", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        no_match_fig = go.Figure(layout=no_match_layout)
        figures = {k: no_match_fig for k in figures}
        return styles['header'], styles['row1'], styles['row3'], styles['row5'], styles['sunburst_header'], styles['sunburst_row'], \
               figures['win_loss'], figures['toss'], figures['runs'], figures['wickets_lost'], \
               figures['runs_dist'], figures['scatter'], figures['extras_sunburst'] # REMOVED figures['sunburst']

    # --- Start Calculations (Use try-except blocks for safety) ---
    styles = {k: visible_style for k in styles}
    tiny_value = 0.05

    try: # Stat 1: Win/Loss/Tie
        match_info_df = stats_df.drop_duplicates(subset=['match_id'])[['match_id', 'winner']].copy()
        wins = {team1: 0, team2: 0, 'Other (Tie/NR)': 0}
        for _, row in match_info_df.iterrows():
            winner = row['winner']
            if winner == team1: wins[team1] += 1
            elif winner == team2: wins[team2] += 1
            else: wins['Other (Tie/NR)'] += 1
        if sum(wins.values()) > 0:
            figures['win_loss'] = px.pie(names=list(wins.keys()), values=list(wins.values()), title=f"Match Results at {venue_name}", hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
            figures['win_loss'].update_traces(textinfo='percent+label+value', pull=[0.05 if k != 'Other (Tie/NR)' else 0 for k in wins.keys()], textposition='inside')
            figures['win_loss'].update_layout(showlegend=False, margin=dict(t=60, b=0, l=0, r=0))
        else: figures['win_loss'] = go.Figure(layout=go.Layout(title=f"No Match Results at {venue_name}"))
    except Exception as e: print(f"Stat Error (Win/Loss): {e}"); traceback.print_exc()

    try: # Stat 2: Toss Analysis
        toss_info_df = stats_df.drop_duplicates(subset=['match_id'])[['match_id', 'toss_winner', 'toss_decision']].copy()
        toss_wins = toss_info_df['toss_winner'].value_counts()
        toss_decision_df = toss_info_df[toss_info_df['toss_winner'].isin([team1, team2])]
        toss_decisions = toss_decision_df.groupby(['toss_winner', 'toss_decision']).size().unstack(fill_value=0)
        toss_data = pd.DataFrame({'Team': [team1, team2]})
        toss_data['Toss Wins'] = toss_data['Team'].map(toss_wins).fillna(0).astype(int)
        toss_data['Elected to Bat'] = toss_data['Team'].map(toss_decisions.get('bat', pd.Series(dtype=int))).fillna(0).astype(int)
        toss_data['Elected to Field'] = toss_data['Team'].map(toss_decisions.get('field', pd.Series(dtype=int))).fillna(0).astype(int)
        if toss_data['Toss Wins'].sum() > 0:
            figures['toss'] = go.Figure(data=[
                go.Bar(name='Toss Wins', x=toss_data['Team'], y=toss_data['Toss Wins'], marker_color='rgb(173, 216, 230)', text=toss_data['Toss Wins'], textposition='auto'),
                go.Bar(name='Elected to Bat', x=toss_data['Team'], y=toss_data['Elected to Bat'], marker_color='rgb(144, 238, 144)', text=toss_data['Elected to Bat'], textposition='auto'),
                go.Bar(name='Elected to Field', x=toss_data['Team'], y=toss_data['Elected to Field'], marker_color='rgb(250, 128, 114)', text=toss_data['Elected to Field'], textposition='auto')])
            figures['toss'].update_layout(barmode='group', title=f"Toss Analysis at {venue_name}", xaxis_title=None, yaxis_title="Count", legend_title_text="Toss Info", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        else: figures['toss'] = go.Figure(layout=go.Layout(title=f"No Toss Data for {team1}/{team2} at {venue_name}"))
    except Exception as e: print(f"Stat Error (Toss): {e}"); traceback.print_exc()

    # --- Stats 3 & 10: Extras --- (This section remains)
    extras_data_original = None
    try:
        extras_cols = ['wides', 'noballs', 'byes', 'legbyes', 'penalty']
        if 'bowling_team' in stats_df.columns and not stats_df['bowling_team'].isnull().all():
            extras_conceded = stats_df.groupby(['match_id', 'bowling_team'])[extras_cols].sum().reset_index()
            extras_conceded = extras_conceded[extras_conceded['bowling_team'].isin([team1, team2])]
            total_extras_by_team = extras_conceded.groupby('bowling_team')[extras_cols].sum()
            total_extras_by_team = total_extras_by_team.reindex([team1, team2], fill_value=0)
            extras_plot_data = total_extras_by_team.reset_index().melt(id_vars='bowling_team', var_name='Extra Type', value_name='Count')
            extras_data_original = extras_plot_data[extras_plot_data['Count'] > 0].copy()
        if extras_data_original is not None and not extras_data_original.empty:
            extras_data_original['Extra Type'] = extras_data_original['Extra Type'].str.title()
            figures['extras_sunburst'] = px.sunburst(extras_data_original, path=['bowling_team', 'Extra Type'], values='Count', title=f"Extras Conceded Breakdown at {venue_name}", color='bowling_team', color_discrete_sequence=px.colors.qualitative.Bold)
            figures['extras_sunburst'].update_layout(margin=dict(t=50, l=10, r=10, b=10))
        else: figures['extras_sunburst'] = go.Figure(layout=go.Layout(title=f"No Extras Data at {venue_name}"))
    except Exception as e: print(f"Stat Error (Extras): {e}"); traceback.print_exc()

    try: # Stat 4: Runs Scored Comparison
        runs_cols = ['runs_off_bat']
        if 'runs_off_bat' in stats_df.columns and 'batting_team' in stats_df.columns:
            runs_scored = stats_df.groupby('batting_team')[runs_cols].sum().reset_index()
            runs_scored = runs_scored[runs_scored['batting_team'].isin([team1, team2])]
            runs_data = pd.DataFrame({'batting_team': [team1, team2]})
            runs_data = pd.merge(runs_data, runs_scored, on='batting_team', how='left').fillna(0)
            if not runs_data.empty and runs_data['runs_off_bat'].sum() > 0 :
                original_runs_max = runs_data['runs_off_bat'].max()
                runs_data['Plot_Value'] = runs_data['runs_off_bat'].replace(0, tiny_value)
                figures['runs'] = px.bar(runs_data, x='batting_team', y='Plot_Value', color='batting_team', title=f"Total Runs Scored (Off Bat) at {venue_name}", labels={'batting_team': 'Team', 'Plot_Value': 'Runs Scored'}, color_discrete_sequence=px.colors.qualitative.Safe, text='runs_off_bat')
                yaxis_max_runs = max(original_runs_max * 1.1, tiny_value * 2)
                figures['runs'].update_traces(textposition='outside')
                figures['runs'].update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_range=[0, yaxis_max_runs])
            else: figures['runs'] = go.Figure(layout=go.Layout(title=f"No Runs Scored Data at {venue_name}"))
        else: figures['runs'] = go.Figure(layout=go.Layout(title=f"Run Data Missing"))
    except Exception as e: print(f"Stat Error (Runs): {e}"); traceback.print_exc()

    try: # Stat 5: Wickets Lost Comparison
        wickets_col = 'player_dismissed'
        if wickets_col in stats_df.columns and 'batting_team' in stats_df.columns:
            stats_df[wickets_col] = pd.to_numeric(stats_df[wickets_col], errors='coerce').fillna(0).astype(int)
            wickets_lost = stats_df.groupby('batting_team')[[wickets_col]].sum().reset_index()
            wickets_lost = wickets_lost[wickets_lost['batting_team'].isin([team1, team2])]
            wickets_data = pd.DataFrame({'batting_team': [team1, team2]})
            wickets_data = pd.merge(wickets_data, wickets_lost, on='batting_team', how='left').fillna(0)
            if not wickets_data.empty and wickets_data[wickets_col].sum() > 0:
                original_wickets_max = wickets_data[wickets_col].max()
                wickets_data['Plot_Value'] = wickets_data[wickets_col].replace(0, tiny_value)
                figures['wickets_lost'] = px.bar(wickets_data, x='batting_team', y='Plot_Value', color='batting_team', title=f"Total Wickets Lost at {venue_name}", labels={'batting_team': 'Team', 'Plot_Value': 'Wickets Lost'}, color_discrete_sequence=px.colors.qualitative.Pastel1, text=wickets_col)
                yaxis_max_wickets = max(original_wickets_max * 1.1, tiny_value * 2)
                figures['wickets_lost'].update_traces(textposition='outside')
                figures['wickets_lost'].update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_range=[0, yaxis_max_wickets])
            else: figures['wickets_lost'] = go.Figure(layout=go.Layout(title=f"No Wickets Lost Data at {venue_name}"))
        else: figures['wickets_lost'] = go.Figure(layout=go.Layout(title=f"Wicket Data Missing"))
    except Exception as e: print(f"Stat Error (Wickets): {e}"); traceback.print_exc()

    # --- Stats 6 & 9: Dismissals ---
    # REMOVED THIS ENTIRE BLOCK
    # dismissal_data_original = None
    # try:
    #     ...
    # except Exception as e: print(f"Stat Error (Dismissals): {e}"); traceback.print_exc()

    try: # Stat 7: Runs Distribution (Box Plot)
        if 'runs_off_bat' in stats_df.columns and 'batting_team' in stats_df.columns and 'match_id' in stats_df.columns:
            runs_per_match = stats_df.groupby(['match_id', 'batting_team'])['runs_off_bat'].sum().reset_index()
            runs_per_match = runs_per_match[runs_per_match['batting_team'].isin([team1, team2])]
            if not runs_per_match.empty:
                 if runs_per_match.groupby('batting_team')['runs_off_bat'].nunique().min() > 1 and len(runs_per_match['match_id'].unique()) > 1:
                     figures['runs_dist'] = px.box(runs_per_match, x='batting_team', y='runs_off_bat', color='batting_team', title=f"Distribution of Runs Scored per Match at {venue_name}", labels={'batting_team': 'Team', 'runs_off_bat': 'Runs Scored'}, points="all", color_discrete_sequence=px.colors.qualitative.Plotly)
                     figures['runs_dist'].update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title=None)
                 else:
                    figures['runs_dist'] = px.bar(runs_per_match, x='batting_team', y='runs_off_bat', color='batting_team', title=f"Runs Scored per Match at {venue_name}", labels={'batting_team':'Team','runs_off_bat':'Runs'})
                    figures['runs_dist'].update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title=None)
            else: figures['runs_dist'] = go.Figure(layout=go.Layout(title=f"No Runs per Match Data at {venue_name}"))
        else: figures['runs_dist'] = go.Figure(layout=go.Layout(title=f"Runs Distribution Data Missing"))
    except Exception as e: print(f"Stat Error (Runs Dist): {e}"); traceback.print_exc()

    try: # Stat 8: Runs vs Wickets Scatter Plot
        if 'runs_off_bat' in stats_df.columns and wickets_col in stats_df.columns and 'batting_team' in stats_df.columns and 'match_id' in stats_df.columns:
            match_summary = stats_df.groupby(['match_id', 'batting_team']).agg(total_runs=('runs_off_bat', 'sum'), total_wickets=(wickets_col, 'sum')).reset_index()
            match_summary = pd.merge(match_summary, stats_df.drop_duplicates(subset=['match_id'])[['match_id', 'winner']], on='match_id', how='left')
            match_summary = match_summary[match_summary['batting_team'].isin([team1, team2])]
            match_summary['winner'] = match_summary['winner'].fillna('Other (Tie/NR)')
            if not match_summary.empty:
                figures['scatter'] = px.scatter(match_summary, x="total_wickets", y="total_runs", color="batting_team", symbol="winner", hover_data=['match_id', 'batting_team', 'winner', 'total_runs', 'total_wickets'], title=f"Runs vs Wickets per Innings at {venue_name}", labels={"total_wickets": "Wickets Lost in Innings", "total_runs": "Runs Scored in Innings", "batting_team": "Batting Team"}, color_discrete_map={team1: px.colors.qualitative.Plotly[0], team2: px.colors.qualitative.Plotly[1], 'Other (Tie/NR)': px.colors.qualitative.Plotly[2]}, symbol_map={team1: "circle", team2: "square", "Other (Tie/NR)":"diamond", pd.NA:"x"})
                figures['scatter'].update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
                figures['scatter'].update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            else: figures['scatter'] = go.Figure(layout=go.Layout(title=f"No Runs vs Wickets Data at {venue_name}"))
        else: figures['scatter'] = go.Figure(layout=go.Layout(title=f"Runs vs Wickets Data Missing"))
    except Exception as e: print(f"Stat Error (Scatter): {e}"); traceback.print_exc()

    # --- Final Return Statement ---
    # REMOVED figures['sunburst'] from the return list
    return styles['header'], styles['row1'], styles['row3'], styles['row5'], styles['sunburst_header'], styles['sunburst_row'], \
           figures['win_loss'], figures['toss'], figures['runs'], figures['wickets_lost'], \
           figures['runs_dist'], figures['scatter'], figures['extras_sunburst']


# Callback 8 & 9: Reset Logic (Update IDs)
@callback(
    # Use prefixed IDs
    Output('me-world-map', 'clickData', allow_duplicate=True),
    Output('me-selected-country-stadium-info', 'children', allow_duplicate=True),
    Output('me-filtered-country-matches-ids', 'data', allow_duplicate=True),
    Output('me-filtered-venue-matches-ids', 'data', allow_duplicate=True),
    Output('me-stats-header', 'style', allow_duplicate=True),
    Output('me-stats-row-1', 'style', allow_duplicate=True),
    Output('me-stats-row-3', 'style', allow_duplicate=True),
    Output('me-stats-row-5', 'style', allow_duplicate=True),
    Output('me-stats-sunburst-header', 'style', allow_duplicate=True),
    Output('me-stats-row-sunbursts', 'style', allow_duplicate=True),
    # Also reset figures to avoid showing old data briefly
    Output('me-win-loss-chart', 'figure', allow_duplicate=True), Output('me-toss-chart', 'figure', allow_duplicate=True),
    Output('me-runs-chart', 'figure', allow_duplicate=True), Output('me-wickets-lost-chart', 'figure', allow_duplicate=True),
    Output('me-runs-distribution-chart', 'figure', allow_duplicate=True), Output('me-scatter-runs-wickets', 'figure', allow_duplicate=True),
    # Output('me-dismissal-sunburst-chart', 'figure', allow_duplicate=True), # REMOVED THIS OUTPUT
    Output('me-extras-sunburst-chart', 'figure', allow_duplicate=True),
    Input('me-event-dropdown', 'value'), Input('me-team1-dropdown', 'value'), Input('me-team2-dropdown', 'value'),
    prevent_initial_call=True
)
def reset_downstream_on_filter_change(event, team1, team2):
    hidden_style = {'display': 'none'}
    initial_stadium_info = html.Div("Click on a country on the map to see stadiums.", className="text-muted")
    no_data_layout = go.Layout(title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False))
    no_data_fig = go.Figure(layout=no_data_layout)
    # Return hidden style for all stat rows AND empty figures
    # REMOVED one no_data_fig from the return list
    return None, initial_stadium_info, [], [], hidden_style, hidden_style, hidden_style, hidden_style, hidden_style, hidden_style, \
           no_data_fig, no_data_fig, no_data_fig, no_data_fig, no_data_fig, no_data_fig, no_data_fig

@callback(
    # Use prefixed IDs
    Output('me-filtered-venue-matches-ids', 'data', allow_duplicate=True),
    Output('me-stats-header', 'style', allow_duplicate=True),
    Output('me-stats-row-1', 'style', allow_duplicate=True),
    Output('me-stats-row-3', 'style', allow_duplicate=True),
    Output('me-stats-row-5', 'style', allow_duplicate=True),
    Output('me-stats-sunburst-header', 'style', allow_duplicate=True),
    Output('me-stats-row-sunbursts', 'style', allow_duplicate=True),
    # Also reset figures
    Output('me-win-loss-chart', 'figure', allow_duplicate=True), Output('me-toss-chart', 'figure', allow_duplicate=True),
    Output('me-runs-chart', 'figure', allow_duplicate=True), Output('me-wickets-lost-chart', 'figure', allow_duplicate=True),
    Output('me-runs-distribution-chart', 'figure', allow_duplicate=True), Output('me-scatter-runs-wickets', 'figure', allow_duplicate=True),
    # Output('me-dismissal-sunburst-chart', 'figure', allow_duplicate=True), # REMOVED THIS OUTPUT
    Output('me-extras-sunburst-chart', 'figure', allow_duplicate=True),
    Input('me-world-map', 'clickData'),
    prevent_initial_call=True
)
def reset_stats_on_map_click(click_data):
    if not click_data: raise PreventUpdate
    hidden_style = {'display': 'none'}
    no_data_layout = go.Layout(title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False))
    no_data_fig = go.Figure(layout=no_data_layout)
    # Return empty venue IDs, hidden styles, and empty figures
    # REMOVED one no_data_fig from the return list
    return [], hidden_style, hidden_style, hidden_style, hidden_style, hidden_style, hidden_style, \
           no_data_fig, no_data_fig, no_data_fig, no_data_fig, no_data_fig, no_data_fig, no_data_fig


# --- REMOVED Run the App Block ---