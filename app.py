import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# --- 1. SETTINGS & CONFIG ---
st.set_page_config(page_title="NBA Sharp Pro Hub", layout="wide", page_icon="🏀")

@st.cache_data(ttl=3600)
def load_nba_base_data():
    """Load Strength of Schedule (SOS) and League Averages."""
    try:
        # Fetching Advanced Team Stats for Defense Ratings
        team_stats_raw = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced', 
            season='2025-26'
        ).get_data_frames()[0]
        
        nba_teams = teams.get_teams()
        id_to_abbr = {t['id']: t['abbreviation'] for t in nba_teams}
        
        avg_drtg = team_stats_raw['DEF_RATING'].mean()
        
        # Map Team Abbreviation to a Strength Multiplier
        sos_map = {
            id_to_abbr[row['TEAM_ID']]: (row['DEF_RATING'] * 0.8 + avg_drtg * 0.2) / avg_drtg 
            for _, row in team_stats_raw.iterrows() if id_to_abbr.get(row['TEAM_ID'])
        }
        return sos_map, avg_drtg
    except:
        # Fallback if API is down
        return {"BOS": 1.0, "GSW": 1.0, "LAL": 1.0, "OKC": 1.0}, 115.0

@st.cache_data(ttl=600)
def get_player_data(player_full_name):
    """Fetch raw game logs for the selected player."""
    nba_players = players.get_players()
    # Direct matching for SGA and others
    match = [p for p in nba_players if p['full_name'].lower() == player_full_name.lower()]
    if not match:
        # Fuzzy matching if exact fails
        match = [p for p in nba_players if player_full_name.lower() in p['full_name'].lower()]
    
    if not match:
        return pd.DataFrame()
    
    p_id = match[0]['id']
    
    # Try current season first, fallback to previous if needed
    try:
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if log.empty:
            log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]
    except:
        return pd.DataFrame()

    # Standardization
    log = log.rename(columns={
        'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers',
        'FGA': 'fga', 'FG_PCT': 'fg_pct', 'FTA': 'fta', 'TOV': 'tov'
    })
    log['pra'] = log['points'] + log['rebounds'] + log['assists']
    log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
    return log

# --- 2. ADVANCED LOGIC ENGINES ---
def calculate_poisson_prob(lambda_val, line):
    """Calculates probability of exceeding a specific line."""
    return round((1 - poisson.cdf(line, lambda_val)) * 100, 1)

def run_monte_carlo(lambda_val, user_line, iterations=10000):
    """Simulates 10,000 games based on the sharp projection."""
    simulated_games = np.random.poisson(lambda_val, iterations)
    levels = [max(0, round(l, 1)) for l in [user_line - 1, user_line, user_line + 1]]
    results = [
        {"Stat Level": f"{l}+", "Hit Frequency": f"{(np.sum(simulated_games >= l)/iterations)*100:.1f}%"}
        for l in sorted(list(set(levels)))
    ]
    return pd.DataFrame(results), simulated_games

# --- 3. UI RENDERING ---
st.title("🏀 NBA Sharp Pro Hub (Legacy v2.0)")
sos_data, league_avg_drtg = load_nba_base_data()

with st.sidebar:
    st.header("🎯 Target Selection")
    search_query = st.text_input("Search Player", "Shai Gilgeous-Alexander")
    all_names = [p['full_name'] for p in players.get_players()]
    filtered = [p for p in all_names if search_query.lower() in p.lower()]
    selected_p = st.selectbox("Confirm Player", filtered if filtered else ["No Player Found"])
    
    st.divider()
    st.subheader("🎲 Manual Context Entry")
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    user_line = st.number_input(f"Sportsbook Line", value=25.5 if stat_category=="points" else 5.5, step=0.5)
    
    selected_opp = st.selectbox("Opponent", sorted(list(sos_data.keys())))
    is_home = st.toggle("Home Game", value=True)
    is_b2b = st.toggle("Back-to-Back (Fatigue)")
    star_out = st.toggle("Star Teammate Out?")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

# --- 4. DATA PROCESSING ---
p_df = get_player_data(selected_p)

if not p_df.empty:
    # 1. Base Average
    p_mean = p_df[stat_category].mean()
    
    # 2. Multipliers
    sos_mult = sos_data.get(selected_opp, 1.0)
    pace_mult = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]
    usage_mult = 1.15 if star_out else 1.0
    venue_mult = 1.03 if is_home else 0.97
    rest_mult = 0.95 if is_b2b else 1.0
    
    # 3. Final Sharp Lambda
    sharp_lambda = p_mean * pace_mult * sos_mult * usage_mult * venue_mult * rest_mult
    over_prob = calculate_poisson_prob(sharp_lambda, user_line)

    col_main, col_side = st.columns([2, 1])

    with col_main:
        # Trend Analysis
        st.subheader("📈 Last 10 Games Performance")
        last_10 = p_df.head(10).iloc[::-1]
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=list(range(1, 11)), y=last_10[stat_category], mode='lines+markers', name='Actual', line=dict(color='#00ff96', width=3)))
        trend_fig.add_hline(y=user_line, line_dash="dash", line_color="#ff4b4b", annotation_text="Vegas Line")
        trend_fig.add_hline(y=sharp_lambda, line_dash="dot", line_color="#ffcc00", annotation_text="Sharp Proj")
        trend_fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(trend_fig, use_container_width=True)

        # Monte Carlo
        st.subheader("🎲 10,000 Game Monte Carlo Simulation")
        sim_df, sim_raw = run_monte_carlo(sharp_lambda, user_line)
        c1, c2 = st.columns(2)
        with c1: st.table(sim_df)
        with c2:
            st.metric("Simulated Average", round(np.mean(sim_raw), 2))
            st.write(f"**90th Percentile Outcome:** {np.percentile(sim_raw, 90)}")

    with col_side:
        st.subheader("📊 Model Output")
        st.metric("Sharp Projection", round(sharp_lambda, 1))
        st.metric("Win Prob (Over)", f"{over_prob}%")
        
        st.divider()
        if over_prob > 60:
            st.success("🔥 **STRONG VALUE DETECTED: OVER**")
        elif over_prob < 40:
            st.error("❄️ **STRONG VALUE DETECTED: UNDER**")
        else:
            st.info("⚖️ **NEUTRAL: NO EDGE DETECTED**")

    st.caption(f"v2.0 stabilized | Dataset: {len(p_df)} games found | SOS: {round(sos_mult, 2)}")

else:
    st.warning("Player data not found. Please confirm the name in the sidebar search.")
