import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# --- 1. SETTINGS & CONFIG ---
st.set_page_config(page_title="NBA Sharp Pro Hub", layout="wide", page_icon="🏀")

def get_fatigue_score(df):
    """Next-Level Logic: The '3 games in 4 nights' rule."""
    if df.empty or len(df) < 3:
        return 1.0, "Fresh"
    
    # Get last 4 calendar days of history
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    recent_games = df.head(4)
    
    # Check if they played 3 games in the last 4 days
    last_date = recent_games['GAME_DATE'].iloc[0]
    four_days_ago = last_date - timedelta(days=4)
    games_in_stretch = recent_games[recent_games['GAME_DATE'] > four_days_ago]
    
    if len(games_in_stretch) >= 3:
        return 0.92, "🚨 Fatigue: 3-in-4 Nights"
    return 1.0, "Standard Cycle"

@st.cache_data(ttl=3600)
def load_nba_base_data():
    try:
        team_stats_raw = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced',
            season='2025-26'
        ).get_data_frames()[0]
        
        nba_teams = teams.get_teams()
        id_to_abbr = {t['id']: t['abbreviation'] for t in nba_teams}
        avg_drtg = team_stats_raw['DEF_RATING'].mean()
        
        sos_map = {id_to_abbr[row['TEAM_ID']]: row['DEF_RATING'] / avg_drtg 
                   for _, row in team_stats_raw.iterrows() if id_to_abbr.get(row['TEAM_ID'])}
        return sos_map
    except:
        return {}

@st.cache_data(ttl=600)
def get_player_data(player_full_name):
    nba_players = players.get_players()
    player = [p for p in nba_players if p['full_name'] == player_full_name and p['is_active']]
    if not player: return pd.DataFrame()
    
    p_id = player[0]['id']
    log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
    
    log = log.rename(columns={
        'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists',
        'FGA': 'fga', 'FTA': 'fta', 'TOV': 'tov'
    })
    log['usage_proxy'] = log['fga'] + (0.44 * log['fta']) + log['tov']
    return log

# --- 3. UI RENDERING ---
st.title("🏀 NBA Sharp: Intelligence Hub (v2.0)")
sos_data = load_nba_base_data()

with st.sidebar:
    st.header("🎯 Target Selection")
    search_query = st.text_input("Search Player", "Jayson Tatum")
    all_active = [p['full_name'] for p in players.get_players() if p['is_active']]
    filtered = [p for p in all_active if search_query.lower() in p.lower()]
    selected_p = st.selectbox("Confirm Player", filtered if filtered else all_active)
    
    selected_opp = st.selectbox("Opponent", sorted(list(sos_data.keys())) if sos_data else ["BOS"])

    st.divider()
    st.subheader("🎲 Game Script (Blowout Logic)")
    spread = st.number_input("Point Spread (Line)", value=0.0, step=0.5)
    blowout_risk = 0.90 if abs(spread) > 12.5 else 1.0
    
    st.subheader("🩹 Injury Impact")
    star_out = st.toggle("Star Teammate Out?", help="Applies a usage 'vacuum' boost (Standard +12%)")
    usage_multiplier = 1.12 if star_out else 1.0

# --- 4. DATA PROCESSING ---
p_df = get_player_data(selected_p)

if not p_df.empty:
    # --- AUTO FATIGUE LOGIC ---
    fatigue_mult, fatigue_label = get_fatigue_score(p_df)
    
    with st.sidebar:
        pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists"])
    
    # Calculation Components
    p_mean = p_df[stat_category].mean()
    p_std = p_df[stat_category].std() if len(p_df) > 1 else 1.0
    sos_mult = sos_data.get(selected_opp, 1.0)
    pace_mult = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]
    
    # FINAL AGGREGATED MODEL
    model_proj = p_mean * pace_mult * sos_mult * fatigue_mult * usage_multiplier * blowout_risk

    # --- UI LAYOUT ---
    col_main, col_side = st.columns([2, 1])

    with col_main:
        # Sharp Context Badges
        b1, b2, b3 = st.columns(3)
        b1.metric("Fatigue Status", fatigue_label)
        b2.metric("Script Risk", "High" if blowout_risk < 1 else "Normal")
        b3.metric("Opp. Strength", f"{round(sos_mult, 2)}x")

        # Visualization
        last_10 = p_df.head(10).iloc[::-1]
        target_line = round(model_proj)
        last_10['hit'] = last_10[stat_category] >= target_line
        
        fig = go.Figure(go.Bar(x=[f"G{i+1}" for i in range(len(last_10))], y=last_10[stat_category], 
                               marker_color=['#00ff96' if h else '#4a4a4a' for h in last_10['hit']]))
        fig.add_hline(y=target_line, line_dash="dash", line_color="#ff4b4b", annotation_text=f"SHARP PROJ: {target_line}")
        fig.update_layout(title=f"Performance vs Adjusted Projection ({stat_category.upper()})", template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.subheader("📋 Advanced Multipliers")
        st.json({
            "Pace Adj": pace_mult,
            "Defense Adj": round(sos_mult, 2),
            "Fatigue Adj": fatigue_mult,
            "Opportunity Adj": usage_multiplier,
            "Blowout Adj": blowout_risk
        })
        
        confidence = (last_10['hit'].sum() / len(last_10)) * 100
        st.metric("Model Confidence", f"{int(confidence)}%", delta=f"{round(model_proj - p_mean, 1)} vs Season Avg")
        
        if confidence >= 70: st.success("🔥 HIGH CONVICTION: OVER")
        elif confidence <= 30: st.error("❄️ HIGH CONVICTION: UNDER")
        else: st.warning("⚖️ NEUTRAL / NO EDGE")
