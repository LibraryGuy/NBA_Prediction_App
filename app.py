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
    player = [p for p in nba_players if p['full_name'].lower() == player_full_name.lower()]
    if not player:
        player = [p for p in nba_players if player_full_name.lower() in p['full_name'].lower()]
    
    if not player: return pd.DataFrame()
    p_id = player[0]['id']
    
    try:
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if log.empty:
            log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]
    except: return pd.DataFrame()

    log = log.rename(columns={
        'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 
        'FGA': 'fga', 'FTA': 'fta', 'TOV': 'tov', 'MATCHUP': 'matchup'
    })
    log['usage_proxy'] = log['fga'] + (0.44 * log['fta']) + log['tov']
    return log

def get_h2h_performance(df, opponent_abbr, stat_cat):
    h2h_df = df[df['matchup'].str.contains(opponent_abbr)]
    if h2h_df.empty: return None
    return round(h2h_df[stat_cat].mean(), 1)

def suggest_parlay_leg(player_name, stat_cat, confidence, star_out):
    if confidence < 60: return None
    if stat_cat == "assists":
        return {"leg": "Primary Scorer OVER Points", "reason": "High assists correlate with teammate efficiency."}
    if stat_cat == "points" and star_out:
        return {"leg": "Opponent Star OVER Points", "reason": "High-volume shootout expected."}
    if stat_cat == "rebounds":
        return {"leg": "Game Total UNDER", "reason": "High rebounds often follow low shooting percentages."}
    return {"leg": "Team Moneyline", "reason": "Model assumes peak player performance leads to a win."}

# --- 2. MOMENTUM CALCULATOR ---
def calculate_momentum(df, stat_cat):
    """Returns the % difference between L5 and L20 averages."""
    if len(df) < 10: return 0, "Insuff. Data"
    l5 = df[stat_cat].head(5).mean()
    l20 = df[stat_cat].head(20).mean()
    diff_pct = ((l5 - l20) / l20) * 100
    
    if diff_pct > 10: status = "🔥 Trending Up"
    elif diff_pct < -10: status = "❄️ Cooling Down"
    else: status = "⚖️ Stable"
    
    return round(diff_pct, 1), status

# --- 3. UI RENDERING ---
st.title("🏀 NBA Sharp: Intelligence Hub (v2.3)")
sos_data = load_nba_base_data()

with st.sidebar:
    st.header("🎯 Target Selection")
    search_query = st.text_input("Search Player", "Jayson Tatum")
    all_names = [p['full_name'] for p in players.get_players()]
    filtered = [p for p in all_names if search_query.lower() in p.lower()]
    selected_p = st.selectbox("Confirm Player", filtered if filtered else all_names)
    selected_opp = st.selectbox("Opponent", sorted(list(sos_data.keys())) if sos_data else ["BOS"])

    st.divider()
    st.subheader("🎲 Game Context")
    spread = st.number_input("Point Spread", value=0.0, step=0.5)
    star_out = st.toggle("Star Teammate Out?", help="Applies +12% usage vacuum.")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

# --- 4. DATA PROCESSING ---
p_df = get_player_data(selected_p)

if not p_df.empty:
    fatigue_mult, fatigue_label = get_fatigue_score(p_df)
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists"])
    
    # Momentum Logic
    mom_val, mom_status = calculate_momentum(p_df, stat_category)

    # Calculation
    p_mean = p_df[stat_category].mean()
    sos_mult = sos_data.get(selected_opp, 1.0)
    pace_mult = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]
    usage_multiplier = 1.12 if star_out else 1.0
    blowout_risk = 0.90 if abs(spread) > 12.5 else 1.0
    
    # Final Model Projection
    model_proj = p_mean * pace_mult * sos_mult * fatigue_mult * usage_multiplier * blowout_risk

    col_main, col_side = st.columns([2, 1])

    with col_main:
        last_10 = p_df.head(10).iloc[::-1]
        target_line = round(model_proj)
        last_10['hit'] = last_10[stat_category] >= target_line
        
        fig = go.Figure(go.Bar(x=[f"G{i+1}" for i in range(len(last_10))], y=last_10[stat_category], 
                               marker_color=['#00ff96' if h else '#4a4a4a' for h in last_10['hit']]))
        fig.add_hline(y=target_line, line_dash="dash", line_color="#ff4b4b", annotation_text=f"Sharp Line: {target_line}")
        fig.update_layout(title=f"{selected_p}: {stat_category.upper()} Trend (Last 10)", template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        confidence = (last_10['hit'].sum() / len(last_10)) * 100
        st.subheader("🚀 Recommendation")
        
        if confidence >= 70:
            st.success(f"**🔥 HIGH CONVICTION: OVER {target_line}**")
            parlay = suggest_parlay_leg(selected_p, stat_category, confidence, star_out)
            if parlay: st.info(f"➕ **Parlay Leg:** {parlay['leg']}\n\n*{parlay['reason']}*")
        elif confidence <= 30:
            st.error(f"**❄️ HIGH CONVICTION: UNDER {target_line}**")
        else:
            st.warning("⚖️ NEUTRAL / STAY AWAY")
        
        st.divider()
        st.subheader("📋 Momentum & Specs")
        # Visual Momentum Indicator
        st.metric(label=f"{stat_category.title()} Momentum", value=f"{mom_val}%", delta=mom_status)
        
        st.write(f"**Fatigue:** {fatigue_label}")
        h2h_avg = get_h2h_performance(p_df, selected_opp, stat_category)
        if h2h_avg: st.write(f"**H2H vs {selected_opp}:** {h2h_avg} {stat_category}")
        
        st.write(f"**Confidence Score:** {int(confidence)}%")

    st.caption("Model v2.3 | Momentum tracks L5 vs L20 average volatility.")
else:
    st.warning(f"Could not find data for {selected_p}.")
