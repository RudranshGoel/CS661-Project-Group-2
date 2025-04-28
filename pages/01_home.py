import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='Home', title='Cricket Data Analysis Dashboard')

layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Welcome to the Cricket Data Analysis and Visualization Platform", className="text-center my-4"), width=12)),
    
    dbc.Row(dbc.Col([
        dbc.Card(dbc.CardBody([
            html.P(
                "This interactive dashboard transforms complex ball-by-ball cricket data into intuitive visual insights. "
                "Data is sourced from Cricsheet.org, covering both international matches and league tournaments.",
                className="lead"
            ),
            html.P(
                "Analyze player and team performances, identify strengths and weaknesses, track trends over time, "
                "and support strategic decision-making for coaching, auctions, or fantasy sports platforms like Dream11 and Stake."
            ),
            html.P("Use the navigation panel on the left or the links below to explore:"),
            html.Ul([
                html.Li(dcc.Link(html.B("Summary"), href="/summary", className="text-decoration-none"), className="mb-2"),
                html.Li(dcc.Link(html.B("Player Analyzer"), href="/analyzer", className="text-decoration-none"), className="mb-2"),
                html.Li(dcc.Link(html.B("Player Radar & Timeline"), href="/radar", className="text-decoration-none"), className="mb-2"),
            
                html.Li(dcc.Link(html.B("Recent Form"), href="/recent-form", className="text-decoration-none"), className="mb-2"),
                html.Li(dcc.Link(html.B("Pre-Betting Analyzer"), href="/prebetting", className="text-decoration-none"), className="mb-2"),
                html.Li(dcc.Link(html.B("Season Heatmap"), href="/season-heatmap", className="text-decoration-none"), className="mb-2"),
                
                html.Li(dcc.Link(html.B("All-Time Rankings"), href="/alltime-rankings", className="text-decoration-none"), className="mb-2"),
                html.Li(dcc.Link(html.B("Player Analysis"), href="/player-analysis", className="text-decoration-none"), className="mb-2"),
                html.Li(dcc.Link(html.B("Tournament Analysis"), href="/tournament", className="text-decoration-none"), className="mb-2"),
                html.Li(dcc.Link(html.B("Performance Analysis"), href="/performance", className="text-decoration-none"), className="mb-2"),
                html.Li(dcc.Link(html.B("Match Explorer"), href="/match-explorer", className="text-decoration-none"), className="mb-2"),
            ]),
            html.P(
                "Start by selecting a player, team, or tournament to dive deep into detailed statistics, "
                "compare against opponents, or visualize evolving career trajectories."
            ),
        ]), className="shadow-sm")
    ], width=12, md=10, lg=8, className="mx-auto")),
    
    html.Hr(className="my-5"),
    dbc.Row(dbc.Col([
        html.Small("Data Source: Cricsheet.org | Note: Data accuracy depends on the available datasets.", className="text-muted text-center d-block")
    ]))
], fluid=True)
