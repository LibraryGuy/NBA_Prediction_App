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
    if df.empty or len(df) < 3: return 1.0, "Fresh"
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    recent_games = df.head(4)
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
            measure_type_detailed_defense='Advanced', season='2025-26'
        ).get_data_frames()[0]
        nba_teams = teams.get_teams()
        id_to_abbr = {t['id']: t['abbreviation'] for t in nba_teams}
        avg_drtg = team_stats_raw['DEF_RATING'].mean()
        sos_map = {id_to_abbr[row['TEAM_ID']]: row['DEF_RATING'] / avg_drtg 
                   for _, row in team_stats_raw.iterrows() if id_to_abbr.get(row['TEAM_ID'])}
        return sos_map
    except: return {}

@st.cache_data(ttl=600)
def get_player_data(player_full_name):
    nba_players = players.get_players()
    player = [p for p in nba_players if p['full_name'] == player_full_name and p['is_active']]
    if not player: return pd.DataFrame()
    p_id = player[0]['id']
    log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
    log = log.rename(columns={'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FGA': 'fga', 'FTA': 'fta', 'TOV': 'tov'})
    log['usage_proxy'] = log['fga'] + (0.44 * log['fta']) + log['tov']
    return log

# --- NEW: PARLAY ENGINE LOGIC ---
def suggest_parlay_leg(player_name, stat_cat, confidence, star_out):
    """Suggests a correlated leg based on the primary bet's narrative."""
    if confidence < 60: return None
    
    if stat_cat == "assists":
        return {"leg": "Primary Scorer OVER Points", "reason": "High assists correlate with teammate bucket efficiency."}
    if stat_cat == "points" and star_out:
        return {"leg": "Opponent Star OVER Points", "reason": "Expected 'Shootout' script with star teammates sidelined."}
    if stat_cat == "rebounds":
        return {"leg": "Game Total UNDER", "reason": "High rebounds often stem from lower shooting percentages/more misses."}
    return {"leg": "Team Moneyline", "reason": "Model assumes player peak performance leads to team win."}

# --- 3. UI RENDERING ---
st.title("🏀 NBA Sharp: Intelligence Hub (v2.1)")
sos_data = load_nba_base_data()

with st.sidebar:
    st.header("🎯 Target Selection")
    search_query = st.text_input("Search Player", "Jayson Tatum")
    all_active = [p['full_name'] for p in players.get_players() if p['is_active']]
    filtered = [p for p in all_active if search_query.lower() in p.lower()]
    selected_p = st.selectbox("Confirm Player", filtered if filtered else all_active)
    selected_opp = st.selectbox("Opponent", sorted(list(sos_data.keys())) if sos_data else ["BOS"])

    st.divider()
    spread = st.number_input("Point Spread", value=0.0, step=0.5)
    star_out = st.toggle("Star Teammate Out?", help="Applies +12% usage vacuum.")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

# --- 4. DATA PROCESSING ---
p_df = get_player_data(selected_p)

if not p_df.empty:
    fatigue_mult, fatigue_label = get_fatigue_score(p_df)
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists"])
    
    # Model Calculation
    p_mean = p_df[stat_category].mean()
    p_std = p_df[stat_category].std() if len(p_df) > 1 else 1.0
    sos_mult = sos_data.get(selected_opp, 1.0)
    pace_mult = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]
    usage_multiplier = 1.12 if star_out else 1.0
    blowout_risk = 0.90 if abs(spread) > 12.5 else 1.0
    
    model_proj = p_mean * pace_mult * sos_mult * fatigue_mult * usage_multiplier * blowout_risk

    col_main, col_side = st.columns([2, 1])

    with col_main:
        # Performance Chart
        last_10 = p_df.head(10).iloc[::-1]
        target_line = round(model_proj)
        last_10['hit'] = last_10[stat_category] >= target_line
        
        fig = go.Figure(go.Bar(x=[f"G{i+1}" for i in range(len(last_10))], y=last_10[stat_category], 
                               marker_color=['#00ff96' if h else '#4a4a4a' for h in last_10['hit']]))
        fig.add_hline(y=target_line, line_dash="dash", line_color="#ff4b4b")
        fig.update_layout(title=f"{selected_p}: {stat_category.upper()} vs {target_line} Line", template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        # Recommendation & Parlay Engine
        confidence = (last_10['hit'].sum() / len(last_10)) * 100
        st.subheader("🚀 Sharp Recommendation")
        
        if confidence >= 70:
            st.success(f"**🔥 HIGH CONVICTION: OVER {target_line}**")
            parlay = suggest_parlay_leg(selected_p, stat_category, confidence, star_out)
            if parlay:
                st.info(f"➕ **Correlated Parlay Leg:**\n\n**{parlay['leg']}**\n\n*{parlay['reason']}*")
        elif confidence <= 30:
            st.error(f"**❄️ HIGH CONVICTION: UNDER {target_line}**")
        else:
            st.warning("⚖️ NEUTRAL / STAY AWAY")
        
        st.divider()
        st.subheader("📋 Data Breakdown")
        st.write(f"Confidence: {int(confidence)}%")
        st.write(f"Fatigue: {fatigue_label}")
        st.write(f"Proj vs Season: {round(model_proj - p_mean, 1)}")

    st.caption("Parlay suggestions are based on historical stat correlations (e.g., Assist volume vs Scorer efficiency).")
else:
    st.warning("No data found for this selection.")
