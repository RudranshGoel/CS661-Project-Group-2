# --- pages/04_radar_bubble.py ---

import dash
from dash import dcc, html, Input, Output, State, callback, register_page # Ensure all are imported from dash
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
import traceback

# --- Register Page ---
register_page(__name__, name='Player Radar & Timeline', path='/radar-bubble', title='Player Radar & Timeline')

# --- Helper Functions ---
def deserialize_data_radar(stored_data):
    """Deserializes data from the store for this page, ensuring date conversion."""
    if stored_data is None:
        print("Radar/Bubble Page: Warning - No data found in store.")
        return None
    try:
        df = pd.read_json(stored_data, orient='split')
        print(f"Radar/Bubble Page: Data deserialized. Shape before date check: {df.shape}")

        if 'start_date' in df.columns:
            print(f"  - 'start_date' dtype before conversion: {df['start_date'].dtype}")
            # Attempt robust conversion (handles ISO strings, maybe epoch ms)
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
            print(f"  - 'start_date' dtype after conversion: {df['start_date'].dtype}")
            if df['start_date'].isnull().all():
                print("  - ERROR: All 'start_date' values are NaT after deserialization conversion.")
                # Return df, let callback handle message
        else:
            print("  - ERROR: 'start_date' column not found after deserialization.")
            return None # Cannot proceed without dates

        print(f"Radar/Bubble Page: Data deserialized and date checked successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"ERROR: Failed to deserialize data from store for radar/bubble page: {e}")
        traceback.print_exc()
        return None

def calculate_player_stats_aggregate(df_player_format):
    """Calculates aggregate stats for a player in a given format."""
    if df_player_format is None or df_player_format.empty:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stats = {}
        stats['total_runs'] = int(df_player_format['runs_scored'].sum())
        stats['total_wickets'] = int(df_player_format['wickets_taken'].sum())
        balls_faced = int(df_player_format['balls_faced'].sum())
        outs = int(df_player_format['player_out'].sum())
        balls_bowled = int(df_player_format['balls_bowled'].sum())
        runs_conceded = int(df_player_format['runs_conceded'].sum())
        stats['batting_avg'] = stats['total_runs'] / outs if outs > 0 else 0.0
        stats['batting_sr'] = (stats['total_runs'] / balls_faced) * 100 if balls_faced > 0 else 0.0
        stats['bowling_avg'] = runs_conceded / stats['total_wickets'] if stats['total_wickets'] > 0 else np.inf
        stats['bowling_econ'] = (runs_conceded / balls_bowled) * 6 if balls_bowled > 0 else np.inf
        return stats

def calculate_format_max_stats(df_format):
    """Calculates max stats across all players for normalization."""
    if df_format is None or df_format.empty or 'player_id' not in df_format.columns:
        print("Warning: Cannot calculate max stats, format_df is empty or missing 'player_id'.")
        return {}
    try:
        grouped = df_format.groupby('player_id').agg(
            total_runs=('runs_scored', 'sum'), total_wickets=('wickets_taken', 'sum'),
            balls_faced=('balls_faced', 'sum'), outs=('player_out', 'sum'),
            balls_bowled=('balls_bowled', 'sum'), runs_conceded=('runs_conceded', 'sum'),
        ).reset_index()
        if grouped.empty: return {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            grouped['batting_avg'] = grouped['total_runs'] / grouped['outs'].replace(0, np.nan)
            grouped['batting_sr'] = (grouped['total_runs'] / grouped['balls_faced'].replace(0, np.nan)) * 100
            grouped['bowling_avg'] = grouped['runs_conceded'] / grouped['total_wickets'].replace(0, np.nan)
            grouped['bowling_econ'] = (grouped['runs_conceded'] / grouped['balls_bowled'].replace(0, np.nan)) * 6
        max_stats = {
            'total_runs': grouped['total_runs'].max() if not grouped['total_runs'].empty else 1,
            'batting_avg': grouped['batting_avg'].dropna().max() if not grouped['batting_avg'].dropna().empty else 1,
            'batting_sr': grouped['batting_sr'].dropna().max() if not grouped['batting_sr'].dropna().empty else 1,
            'total_wickets': grouped['total_wickets'].max() if not grouped['total_wickets'].empty else 1,
            'bowling_avg': grouped['bowling_avg'].dropna().max() if not grouped['bowling_avg'].dropna().empty else 100, # Use a high number for avg/econ if no data
            'bowling_econ': grouped['bowling_econ'].dropna().max() if not grouped['bowling_econ'].dropna().empty else 20, # Use a high number for avg/econ if no data
        }
        for key, value in max_stats.items():
             if pd.isna(value) or not np.isfinite(value) or value <= 0:
                 max_stats[key] = 1 # Ensure positive divisor
        print(f"Max stats calculated for format: {max_stats}")
        return max_stats
    except Exception as e:
        print(f"Error in calculate_format_max_stats: {e}")
        traceback.print_exc()
        return {}

def normalize_stat(value, max_value, lower_is_better=False):
    """Normalizes a stat to 0-100 scale."""
    if pd.isna(value) or not np.isfinite(value): return 0
    if max_value is None or pd.isna(max_value) or not np.isfinite(max_value) or max_value <= 0:
        print(f"Warning: Invalid max_value ({max_value}) for normalization. Returning 0 for value {value}.")
        return 0
    normalized = (value / max_value) * 100
    if lower_is_better: normalized = max(0, 100 - normalized) # Invert score
    return max(0, min(normalized, 100)) # Clamp

# --- Default Placeholder Message ---
radar_placeholder = dbc.Alert(
    "Select a player and format to view visualizations.",
    color="info", className="text-center mt-4", id="radar-placeholder-message"
)

# --- Layout Definition ---
def layout():
    print("Generating Radar/Bubble page layout...")
    initial_player_opts = [{'label': "Loading players...", 'value': "", 'disabled': True}]
    ALLOWED_FORMATS = ["ODI", "T20", "T20I", "Test"]
    return dbc.Container([
        dbc.Row(dbc.Col(html.H2("Player Radar & Timeline Analysis"), width=12, className="mb-4")),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Select Player and Format"),
                    dbc.CardBody([
                        html.Div([
                            html.Label("Player:", className="fw-bold"),
                            dcc.Dropdown( id='radar-player-dropdown', options=initial_player_opts, value=None, placeholder="Select player...", clearable=False),
                        ], className="mb-3"),
                        html.Div([
                            html.Label("Format:", className="fw-bold"),
                            dbc.RadioItems(id='radar-format-radio', options=[{'label': fmt, 'value': fmt} for fmt in ALLOWED_FORMATS], value=ALLOWED_FORMATS[0], inline=True, labelStyle={'margin-right': '15px'}, inputClassName="me-1"),
                        ]),
                    ])
                ])
            ], width=12, md=4, className="mb-3 mb-md-0"),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Overall Performance Radar (Normalized 0-100)"),
                    dbc.CardBody(
                        dcc.Loading(id="loading-radar", type="circle", children=[
                            dcc.Graph(id='radar-plot', config={'displayModeBar': False}, style={'height': '400px'})
                        ])
                    )
                ])
            ], width=12, md=8),
        ], className="mb-4"),
        dbc.Row([
             dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Runs Scored Over Time"),
                    dbc.CardBody(
                        dcc.Loading(id="loading-bubble-runs", type="circle", children=[
                            dcc.Graph(id='bubble-runs-plot', config={'displayModeBar': False}, style={'height': '350px'})
                        ])
                    )
                ])
             ], width=12, md=6, className="mb-3 mb-md-0"),
             dbc.Col([
                 dbc.Card([
                    dbc.CardHeader("Wickets Taken Over Time"),
                    dbc.CardBody(
                        dcc.Loading(id="loading-bubble-wickets", type="circle", children=[
                            dcc.Graph(id='bubble-wickets-plot', config={'displayModeBar': False}, style={'height': '350px'})
                        ])
                    )
                ])
             ], width=12, md=6),
        ], className="mb-4"),
        html.Div(id='radar-plots-output-area', children=radar_placeholder)
    ], fluid=True)


# --- Callbacks ---

# Callback 1: Populate Player Dropdown
@callback(
    Output('radar-player-dropdown', 'options'),
    Output('radar-player-dropdown', 'value'),
    Input('main-data-store', 'data'),
    prevent_initial_call=False
)
def update_radar_player_options(stored_data):
    print("Callback triggered: update_radar_player_options")
    df = deserialize_data_radar(stored_data)
    if df is None or df.empty or 'name' not in df.columns:
        return ([{'label': "Data loading issue", 'value': "", 'disabled': True}], None)
    try:
        player_choices = sorted(df["name"].dropna().unique())
        if not player_choices: return ([{'label': "No players", 'value': "", 'disabled': True}], None)
        options = [{'label': p, 'value': p} for p in player_choices]
        default_value = options[0]['value'] if options else None
        print(f"  - Radar Page: Generated {len(options)} player options.")
        return options, default_value
    except Exception as e:
        print(f"  - Radar Page: Error generating player options: {e}")
        traceback.print_exc()
        return ([{'label': "Error", 'value': "", 'disabled': True}], None)

# Callback 2: Update Plots based on Player and Format Selection
@callback(
    Output('radar-plot', 'figure'),
    Output('bubble-runs-plot', 'figure'),
    Output('bubble-wickets-plot', 'figure'),
    Output('radar-plots-output-area', 'children'),
    Input('radar-player-dropdown', 'value'),
    Input('radar-format-radio', 'value'),
    State('main-data-store', 'data'),
    prevent_initial_call=True
)
def update_radar_bubble_plots(selected_player, selected_format, stored_data):
    print(f"Callback triggered: update_radar_bubble_plots (Player: {selected_player}, Format: {selected_format})")

    # Initialize empty figures with placeholder titles
    fig_radar = go.Figure().update_layout(title="Select Player/Format", title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig_bubble_runs = go.Figure().update_layout(title="Select Player/Format", title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig_bubble_wickets = go.Figure().update_layout(title="Select Player/Format", title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    if not selected_player or not selected_format:
        print("  - Player or format not selected.")
        return fig_radar, fig_bubble_runs, fig_bubble_wickets, radar_placeholder

    df_full = deserialize_data_radar(stored_data)
    if df_full is None or df_full.empty:
        print("  - Main data not loaded for plots.")
        return fig_radar, fig_bubble_runs, fig_bubble_wickets, dbc.Alert("Error: Data not available.", color="danger")

    try:
        # --- Filter Data ---
        player_mask = df_full["name"] == selected_player
        format_mask = df_full["match_type"] == selected_format
        player_format_df = df_full[player_mask & format_mask].copy()
        format_df = df_full[format_mask].copy()

        if player_format_df.empty:
            msg = f"No {selected_format} data found for {selected_player}."
            print(f"  - {msg}")
            fig_radar.update_layout(title=f"Radar Plot<br>({msg})")
            fig_bubble_runs.update_layout(title=f"Runs Timeline<br>({msg})")
            fig_bubble_wickets.update_layout(title=f"Wickets Timeline<br>({msg})")
            return fig_radar, fig_bubble_runs, fig_bubble_wickets, ""

        # --- Radar Plot ---
        max_stats = calculate_format_max_stats(format_df)
        stats = calculate_player_stats_aggregate(player_format_df)

        if not stats or not max_stats:
            print("  - Could not calculate stats or max stats for Radar plot.")
            fig_radar.update_layout(title=f"Normalized Radar Plot<br>(Stats/Normalization Data Unavailable)")
        else:
            categories = ['Total Runs', 'Batting Avg', 'Batting SR', 'Total Wickets', 'Bowling Avg', 'Bowling Econ']
            raw_values = [
                stats.get('total_runs', 0), stats.get('batting_avg', 0), stats.get('batting_sr', 0),
                stats.get('total_wickets', 0), stats.get('bowling_avg', np.inf), stats.get('bowling_econ', np.inf)
            ]
            normalized_values = [
                 normalize_stat(raw_values[0], max_stats.get('total_runs')),
                 normalize_stat(raw_values[1], max_stats.get('batting_avg')),
                 normalize_stat(raw_values[2], max_stats.get('batting_sr')),
                 normalize_stat(raw_values[3], max_stats.get('total_wickets')),
                 normalize_stat(raw_values[4], max_stats.get('bowling_avg'), lower_is_better=True),
                 normalize_stat(raw_values[5], max_stats.get('bowling_econ'), lower_is_better=True)
            ]
            hover_texts = []
            for cat, val in zip(categories, raw_values):
                if pd.isna(val) or not np.isfinite(val): text = f"{cat}: N/A"
                elif isinstance(val, float): text = f"{cat}: {val:.2f}"
                else: text = f"{cat}: {val}"
                hover_texts.append(text)

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=normalized_values, theta=categories, fill='toself', name=selected_player,
                hoverinfo='text', text=hover_texts
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False,
                title=f"Normalized Performance Radar ({selected_format})<br>{selected_player}",
                title_x=0.5, margin=dict(l=60, r=60, t=80, b=40)
            )
            print("  - Radar plot generated.")

        # --- Bubble Plots ---
        if 'start_date' not in player_format_df.columns or player_format_df['start_date'].isnull().all():
            print("  - ERROR: 'start_date' column missing or all NaN for bubble plots.")
            fig_bubble_runs.update_layout(title=f"Runs Timeline<br>(Date data unavailable)")
            fig_bubble_wickets.update_layout(title=f"Wickets Timeline<br>(Date data unavailable)")
        else:
            player_format_df_sorted = player_format_df.sort_values('start_date').copy()
            date_col_name = 'start_date' # Use the actual date column

            # Runs Plot
            if not player_format_df_sorted.empty:
                 try:
                    fig_bubble_runs = px.scatter(
                        player_format_df_sorted, x=date_col_name, y="runs_scored",
                        hover_data=['opposition_team', 'match_id', 'balls_faced', 'fours_scored', 'sixes_scored', 'out_kind'],
                        title=f"Runs Scored per Match ({selected_format})",
                        labels={date_col_name: 'Match Date', 'runs_scored': 'Runs Scored'}
                    )
                    fig_bubble_runs.update_traces(marker=dict(size=8))
                    fig_bubble_runs.update_layout(title_x=0.5, margin=dict(t=50, b=30, l=30, r=30))
                    print("  - Runs bubble plot generated.")
                 except Exception as e_run:
                    print(f"  - ERROR generating Runs bubble plot: {e_run}")
                    fig_bubble_runs.update_layout(title=f"Runs Timeline<br>(Plotting Error: {e_run})")
            else:
                print("  - No run scoring data found for Runs bubble plot.")
                fig_bubble_runs.update_layout(title=f"Runs Timeline<br>(No batting data)")

            # Wickets Plot
            bowling_df = player_format_df_sorted[player_format_df_sorted['balls_bowled'] > 0].copy()
            if not bowling_df.empty:
                 try:
                    fig_bubble_wickets = px.scatter(
                        bowling_df, x=date_col_name, y="wickets_taken",
                        hover_data=['opposition_team', 'match_id', 'balls_bowled', 'runs_conceded'],
                        title=f"Wickets Taken per Match ({selected_format})",
                        labels={date_col_name: 'Match Date', 'wickets_taken': 'Wickets Taken'}
                    )
                    fig_bubble_wickets.update_traces(marker=dict(size=8))
                    fig_bubble_wickets.update_layout(title_x=0.5, margin=dict(t=50, b=30, l=30, r=30), yaxis={'tickformat': ',.0f', 'dtick': 1})
                    max_wickets = bowling_df['wickets_taken'].max()
                    fig_bubble_wickets.update_yaxes(range=[-0.5, max(1, max_wickets + 0.5)]) # Ensure range is at least -0.5 to 1.5
                    print("  - Wickets bubble plot generated.")
                 except Exception as e_wkt:
                    print(f"  - ERROR generating Wickets bubble plot: {e_wkt}")
                    fig_bubble_wickets.update_layout(title=f"Wickets Timeline<br>(Plotting Error: {e_wkt})")
            else:
                 print("  - No bowling data found for Wickets bubble plot.")
                 fig_bubble_wickets.update_layout(title=f"Wickets Timeline<br>(No bowling data)")

        # Return figures and hide placeholder
        print("  - update_radar_bubble_plots finished successfully.")
        return fig_radar, fig_bubble_runs, fig_bubble_wickets, ""

    except Exception as e:
        print(f"ERROR during plot generation in update_radar_bubble_plots: {e}")
        traceback.print_exc()
        error_alert = dbc.Alert(f"An error occurred while generating plots: {e}", color="danger")
        return go.Figure(), go.Figure(), go.Figure(), error_alert