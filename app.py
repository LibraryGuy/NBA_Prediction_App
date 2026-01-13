import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# --- 1. SETTINGS & CONFIG ---
st.set_page_config(page_title="NBA Sharp: Intelligence Hub", layout="wide", page_icon="🏀")

# --- CORE LOGIC: NBA MULTIPLIERS ---
def get_rest_multiplier(days_rest, p_age):
    """Calculates a multiplier based on rest days and player fatigue."""
    multiplier = 1.0
    impact_reasons = []

    if days_rest == 0:  # Back-to-back
        penalty = 0.08 if p_age > 30 else 0.04
        multiplier -= penalty
        impact_reasons.append(f"Back-to-Back (-{int(penalty*100)}%)")
    elif days_rest >= 3:
        multiplier += 0.05
        impact_reasons.append("High Rest (+5%)")

    reason_str = " + ".join(impact_reasons) if impact_reasons else "Standard Rest"
    return round(multiplier, 2), reason_str

@st.cache_data(ttl=3600)
def load_nba_base_data():
    """Loads team defensive stats and maps them to abbreviations for SoS."""
    try:
        # Fetch the team stats (Advanced)
        team_stats_raw = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced',
            season='2025-26'
        ).get_data_frames()[0]
        
        # Static team list to map TeamID to Abbreviation (e.g., 1610612747 -> LAL)
        nba_teams = teams.get_teams()
        id_to_abbr = {t['id']: t['abbreviation'] for t in nba_teams}
        
        avg_drtg = team_stats_raw['DEF_RATING'].mean()
        
        # Build SoS Map: Higher DRTG = Worse Defense = Higher Multiplier for Offense
        sos_map = {}
        for _, row in team_stats_raw.iterrows():
            abbr = id_to_abbr.get(row['TEAM_ID'])
            if abbr:
                sos_map[abbr] = row['DEF_RATING'] / avg_drtg
                
        return sos_map
    except Exception as e:
        st.error(f"Data Sync Error: {e}")
        return {}

@st.cache_data(ttl=600)
def get_player_data(player_full_name):
    """Fetches recent game logs for a specific player."""
    nba_players = players.get_players()
    # Find active player matching the name
    player = [p for p in nba_players if p['full_name'] == player_full_name and p['is_active']]
    
    if not player:
        return pd.DataFrame()
    
    p_id = player[0]['id']
    # Fetch 2025-26 Game Logs
    log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
    
    # Standardize columns for consistency in calculations
    log = log.rename(columns={
        'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists',
        'STL': 'steals', 'BLK': 'blocks', 'MATCHUP': 'opponent'
    })
    return log

# --- 3. UI RENDERING ---
st.title("📊 NBA Sharp Intelligence Hub")

# Load Global SoS Data
sos_data = load_nba_base_data()

with st.sidebar:
    st.header("🎯 Target Selection")
    
    # Search functionality
    search_query = st.text_input("Search Player", "Jayson Tatum")
    all_active_names = [p['full_name'] for p in players.get_players() if p['is_active']]
    
    # Filter list based on search
    filtered_list = [p for p in all_active_names if search_query.lower() in p.lower()]
    selected_p = st.selectbox("Confirm Selection", filtered_list if filtered_list else all_active_names)
    
    # Opponent Selection (using Abbreviations like LAL, BOS)
    if sos_data:
        opp_list = sorted(list(sos_data.keys()))
        selected_opp = st.selectbox("Opponent Defense", opp_list)
    else:
        selected_opp = st.text_input("Opponent (Manual Entry)", "BOS")
    
    st.divider()
    st.subheader("🔋 Fatigue & Environment")
    days_off = st.slider("Days of Rest", 0, 4, 1)
    p_age = st.number_input("Player Age", 18, 45, 25)
    
    st.divider()
    st.subheader("⚙️ Market Settings")
    risk_pref = st.radio("Target Odds Profile", ["Conservative (-115)", "Standard (+100)", "Aggressive (+180)"], index=1)
    pace_script = st.select_slider("Expected Game Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

# --- 4. DATA PROCESSING & MODEL ---
p_df = get_player_data(selected_p)

if not p_df.empty:
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists"])
    
    # MODEL CALCULATION
    p_mean = p_df[stat_category].mean()
    p_std = p_df[stat_category].std() if len(p_df) > 1 else 1.0
    
    sos_multiplier = sos_data.get(selected_opp, 1.0)
    pace_boost = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]
    rest_multiplier, rest_reason = get_rest_multiplier(days_off, p_age)
    
    # Final Model Projection
    model_proj = p_mean * pace_boost * sos_multiplier * rest_multiplier

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.info(f"⚡ **Model Context:** {rest_reason} | Pace: {pace_script} (x{pace_boost})")
        
        # Risk Offsets
        risk_offsets = {"Conservative (-115)": -0.4, "Standard (+100)": 0, "Aggressive (+180)": 0.5}
        target_line = round(model_proj + (risk_offsets[risk_pref] * p_std))

        # Graphing
        last_10 = p_df.head(10).iloc[::-1] # Most recent 10 games
        last_10['hit'] = last_10[stat_category] >= target_line
        
        fig_hits = go.Figure(go.Bar(
            x=[f"Game {i+1}" for i in range(len(last_10))], 
            y=last_10[stat_category], 
            marker_color=['#00ff96' if hit else '#4a4a4a' for hit in last_10['hit']]
        ))
        fig_hits.add_hline(y=target_line, line_dash="dash", line_color="#ff4b4b")
        fig_hits.update_layout(title=f"Last 10 vs {risk_pref} Target ({target_line}+)", template="plotly_dark", height=300)
        st.plotly_chart(fig_hits, use_container_width=True)

    with col_side:
        st.subheader("📋 Metrics")
        avg_data = {
            "Metric": [stat_category.capitalize(), "SoS Multiplier"],
            "Season": [round(p_mean, 1), round(sos_multiplier, 2)],
            "Last 5": [round(p_df[stat_category].head(5).mean(), 1), "---"]
        }
        st.table(pd.DataFrame(avg_data))
        
        st.divider()
        st.subheader("🚀 Recommendation")
        confidence = (last_10['hit'].sum() / len(last_10)) * 100
        st.metric("Model Confidence", f"{int(confidence)}%", delta=f"{round(model_proj - p_mean, 1)} vs Avg")
        
        if confidence >= 70:
            st.success("🔥 HIGH CONVICTION: OVER")
        elif confidence <= 30:
            st.error("❄️ HIGH CONVICTION: UNDER")
        else:
            st.warning("⚖️ NEUTRAL/STAY AWAY")

    st.divider()
    st.caption(f"Data sourced from NBA Stats API. Model includes {selected_opp} defensive rating adjustment.")
else:
    st.warning(f"No 2025-26 game data found for {selected_p}. Please try another player.")
