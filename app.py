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
    """Loads team defensive stats to calculate Strength of Schedule (SoS)."""
    # Defensive Rating (DRTG) is the best proxy for NBA defense
    team_stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
    # Normalize DRTG so higher = better defense (which should lower the multiplier)
    avg_drtg = team_stats['DEF_RATING'].mean()
    sos_map = {row['TEAM_ABBREVIATION']: avg_drtg / row['DEF_RATING'] for _, row in team_stats.iterrows()}
    return sos_map

@st.cache_data(ttl=600)
def get_player_data(player_full_name):
    """Fetches recent game logs for a specific player."""
    nba_players = players.get_players()
    player = [p for p in nba_players if p['full_name'] == player_full_name and p['is_active']]
    
    if not player:
        return pd.DataFrame()
    
    p_id = player[0]['id']
    log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]
    
    # Standardize columns
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
    search_query = st.text_input("Search Player (e.g., LeBron James)", "Jayson Tatum")
    all_active = [p['full_name'] for p in players.get_players() if p['is_active']]
    selected_p = st.selectbox("Confirm Selection", [p for p in all_active if search_query.lower() in p.lower()])
    
    selected_opp = st.selectbox("Opponent Defense", sorted(list(sos_data.keys())))
    
    st.divider()
    st.subheader("🔋 Fatigue & Environment")
    days_off = st.slider("Days of Rest", 0, 4, 1)
    p_age = st.number_input("Player Age", 18, 45, 25)
    
    st.divider()
    st.subheader("⚙️ Market Settings")
    risk_pref = st.radio("Target Odds Profile", ["Conservative (-115)", "Standard (+100)", "Aggressive (+180)"], index=1)
    pace_script = st.select_slider("Expected Game Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

# DATA PROCESSING
p_df = get_player_data(selected_p)

if not p_df.empty:
    # NBA Stat Selection
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists"])
    
    # SHARP PROJECTION CALCULATION
    p_mean = p_df[stat_category].mean()
    p_std = p_df[stat_category].std() if len(p_df) > 1 else 1.0
    
    sos_multiplier = sos_data.get(selected_opp, 1.0)
    pace_boost = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]
    rest_multiplier, rest_reason = get_rest_multiplier(days_off, p_age)
    
    # Final Model Projection
    model_proj = p_mean * pace_boost * sos_multiplier * rest_multiplier

    col_main, col_side = st.columns([2, 1])

    with col_main:
        # Impact Badge
        st.info(f"⚡ **Model Context:** {rest_reason} | Pace: {pace_script} (x{pace_boost})")
        
        # Risk Offsets (Z-Score adjustment)
        risk_offsets = {"Conservative (-115)": -0.4, "Standard (+100)": 0, "Aggressive (+180)": 0.5}
        target_line = round(model_proj + (risk_offsets[risk_pref] * p_std))

        # Visualizing Last 10 Games
        last_10 = p_df.head(10).iloc[::-1] # Reverse to chronological
        last_10['hit'] = last_10[stat_category] >= target_line
        
        fig_hits = go.Figure(go.Bar(
            x=list(range(1, 11)), 
            y=last_10[stat_category], 
            marker_color=['#00ff96' if hit else '#4a4a4a' for hit in last_10['hit']]
        ))
        fig_hits.add_hline(y=target_line, line_dash="dash", line_color="#ff4b4b", annotation_text="Target")
        fig_hits.update_layout(title=f"Last 10 vs {risk_pref} Target ({target_line}+)", template="plotly_dark", height=300)
        st.plotly_chart(fig_hits, use_container_width=True)

    with col_side:
        st.subheader("📋 Performance Metrics")
        avg_data = {
            "Metric": [stat_category.capitalize(), "Efficiency"],
            "Season Avg": [round(p_mean, 1), f"{round(sos_multiplier, 2)} SoS"],
            "Last 5": [round(p_df[stat_category].head(5).mean(), 1), "---"]
        }
        st.table(pd.DataFrame(avg_data))
        
        st.divider()
        st.subheader("🚀 Sharp Recommendation")
        confidence = (last_10['hit'].sum() / 10) * 100
        st.metric("Model Confidence", f"{confidence}%", delta=f"{round(model_proj - p_mean, 1)} vs Avg")
        
        if confidence >= 70:
            st.success("🔥 HIGH CONVICTION: OVER")
        elif confidence <= 30:
            st.error("❄️ HIGH CONVICTION: UNDER")
        else:
            st.warning("⚖️ NEUTRAL/STAY AWAY")

    st.divider()
    st.caption(f"Data sourced via NBA_API. Projection includes {selected_opp} defensive rating adjustment.")
else:
    st.warning("No data found for this player in the 2024-25 season.")
