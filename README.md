# Interactive Cricket Performance Analysis Dashboard

## Introduction

Cricket, a globally celebrated sport, generates vast amounts of structured data, from individual player stats and match results to venue-specific details and temporal trends. While traditional scorecards provide basic facts, they often obscure deeper patterns and insights crucial for teams, players, analysts, and fans.

This project presents an **Interactive Cricket Performance Analysis Dashboard**, developed using Python's Dash framework and Plotly for visualizations. It transforms complex, raw ball-by-ball cricket data, sourced from [Cricsheet.org](https://cricsheet.org/), into intuitive and interactive visual insights. The process involved automated data scraping, thorough cleaning and standardization, extensive feature engineering (e.g., rolling averages, contextual metrics), and the creation of a multi-page web application.

**Need for Visualization:**

The sheer volume of cricket data makes manual interpretation challenging. This dashboard addresses this by:

*   Simplifying complex datasets into easily understandable visualizations.
*   Uncovering hidden performance patterns and anomalies.
*   Enabling dynamic comparison across players, teams, formats, and time periods.
*   Supporting strategic decision-making for various stakeholders, including team management, players, coaches, fantasy sports users (like Dream11, Stake), and betting analysts.

## Key Features

*   **Team-Based Visualizations:** Assess team strengths/weaknesses, inform tactical decisions, identify opposition vulnerabilities.
*   **Player-Based Visualizations:** Facilitate player self-assessment, support auction/scouting strategies, enhance fantasy sports selections with context-driven insights.
*   **Multi-Page Interface:** Organised analysis covering Summary, Player Analysis, Tournament Analysis, Performance Rankings, Recent Form, Pre-Betting Insights, Season Heatmaps, Match Exploration, All-Time Rankings, and more.
*   **Interactive Filtering:** Dynamically filter data by player, team, date range, match format, venue, opposition, etc.
*   **Diverse Chart Types:** Utilizes Choropleth Maps, Scatter Plots, Bar Charts, Line Graphs, Pie Charts, Radar Charts, Heatmaps, Box Plots, Indicator Gauges, Sunburst Diagrams, and Interactive DataTables.
*   **Data-Driven Insights:** Analyze performance trends, head-to-head records, venue statistics, player consistency, batting/bowling styles, and risk vs. aggression profiles.

## Technology Stack

*   **Backend:** Python
*   **Data Processing:** Pandas
*   **Web Framework:** Dash
*   **Visualization:** Plotly
*   **UI Components:** Dash Bootstrap Components
*   **Geospatial Mapping:** pycountry (potentially used for mapping)
*   **Reporting:** reportlab (potentially used for PDF export features)

## Setup and Run

Follow these steps to set up and run the Cricket Player Performance Analyzer locally:

1.  **Clone the Repository:**
    Clone this repository to your local machine or download the source code ZIP.
    ```bash
    git clone https://github.com/RudranshGoel/CS661-Project-Group-2
    cd CS661-Project-Group-2
    ```
    (If you downloaded the ZIP, extract it and navigate to the extracted folder in your terminal).

2.  **Download the Preprocessed Data Files:**
    The dashboard relies on preprocessed CSV data files. Download these files from the following link:
    [Link to download the necessary CSV files](https://drive.google.com/drive/folders/13cVUXclsawjvRq_vTQorFTzAwDpkKdzG)

3.  **Set up the Data Directory:**
    Create a folder named `data` inside the main project directory (the one you cloned or extracted). Place *all* the downloaded CSV files directly into this `data/` folder. The application expects the data to be present here.
    ```
    [repository-folder-name]/
    ├── data/
    │   ├── total_data.csv
    │   ├── mw_overall.csv
    │   ├── mw_pw.csv
    │   ├── mw_pw_profiles.csv
    │   ├── people.csv
    │   ├── style_based_features.csv
    │   ├── style_data_with_start_date.csv
    │   ├── app6.csv
    │   └── ... (any other required csv files)
    ├── pages/
    │   ├── 01_home.py
    │   └── ... (other page modules)
    ├── multi_page_app.py
    ├── feature_generation.py
    └── ... (other python scripts and files)
    ```

4.  **Install Required Python Packages:**
    It's recommended to use a virtual environment. Install all necessary libraries using pip:
    ```bash
    # Optional: Create and activate a virtual environment
    # python -m venv venv
    # source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    pip install gradio pandas matplotlib dash dash-bootstrap-components plotly pycountry reportlab
    ```

5.  **Launch the Application:**
    Navigate to the main project directory in your terminal (the one containing `multi_page_app.py`) and run the application script:
    ```bash
    python multi_page_app.py
    ```
    This will start the Dash development server. Open your web browser and go to the address provided in the terminal output (usually `http://127.0.0.1:8050/` or similar).

You can now explore the Interactive Cricket Performance Analysis Dashboard and interact with the various analysis modules.
