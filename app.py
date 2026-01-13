import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# --- 1. SETTINGS & CONFIG ---
st.set_page_config(page_title="NBA Sharp: Intelligence Hub", layout="wide", page_icon="🏀")

def calculate_rest_days(df):
    """Calculates days since the last game based on the game log."""
    if df.empty:
        return 1
    try:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        last_game = df['GAME_DATE'].iloc[0]
        delta = (datetime.now() - last_game).days
        return max(0, delta)
    except:
        return 1

# --- CORE LOGIC: NBA MULTIPLIERS ---
def get_rest_multiplier(days_rest, p_age):
    multiplier = 1.0
    impact_reasons = []

    if days_rest == 0:
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
    try:
        team_stats_raw = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced',
            season='2025-26'
        ).get_data_frames()[0]
        
        nba_teams = teams.get_teams()
        id_to_abbr = {t['id']: t['abbreviation'] for t in nba_teams}
        avg_drtg = team_stats_raw['DEF_RATING'].mean()
        
        sos_map = {}
        for _, row in team_stats_raw.iterrows():
            abbr = id_to_abbr.get(row['TEAM_ID'])
            if abbr:
                sos_map[abbr] = row['DEF_RATING'] / avg_drtg
        return sos_map
    except:
        return {}

@st.cache_data(ttl=600)
def get_player_data(player_full_name):
    nba_players = players.get_players()
    player = [p for p in nba_players if p['full_name'] == player_full_name and p['is_active']]
    if not player:
        return pd.DataFrame()
    
    p_id = player[0]['id']
    # Getting Advanced stats for Usage Rate + Standard log
    log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
    
    log = log.rename(columns={
        'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists',
        'STL': 'steals', 'BLK': 'blocks', 'MATCHUP': 'opponent',
        'FGA': 'fga', 'FTA': 'fta', 'TOV': 'tov'
    })
    
    # Simple Usage Rate Formula: 100 * ((FGA + 0.44 * FTA + TOV))
    # Note: Real usage requires team stats, this is "Relative Usage Volume"
    log['usage_proxy'] = log['fga'] + (0.44 * log['fta']) + log['tov']
    return log

# --- 3. UI RENDERING ---
st.title("📊 NBA Sharp Intelligence Hub")
sos_data = load_nba_base_data()

with st.sidebar:
    st.header("🎯 Target Selection")
    search_query = st.text_input("Search Player", "Jayson Tatum")
    all_active_names = [p['full_name'] for p in players.get_players() if p['is_active']]
    filtered_list = [p for p in all_active_names if search_query.lower() in p.lower()]
    selected_p = st.selectbox("Confirm Selection", filtered_list if filtered_list else all_active_names)
    
    if sos_data:
        selected_opp = st.selectbox("Opponent Defense", sorted(list(sos_data.keys())))
    else:
        selected_opp = st.text_input("Opponent (Manual Entry)", "BOS")

    st.divider()
    st.subheader("⚙️ Market Settings")
    risk_pref = st.radio("Target Odds Profile", ["Conservative (-115)", "Standard (+100)", "Aggressive (+180)"], index=1)
    pace_script = st.select_slider("Expected Game Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

# --- 4. DATA PROCESSING ---
p_df = get_player_data(selected_p)

if not p_df.empty:
    # --- REST-O-METER LOGIC ---
    auto_rest = calculate_rest_days(p_df)
    
    with st.sidebar:
        st.subheader("🔋 Fatigue & Environment")
        days_off = st.slider("Days of Rest (Auto-detected)", 0, 4, auto_rest)
        p_age = st.number_input("Player Age", 18, 45, 25)

    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists"])
    
    # Calculations
    p_mean = p_df[stat_category].mean()
    p_std = p_df[stat_category].std() if len(p_df) > 1 else 1.0
    sos_multiplier = sos_data.get(selected_opp, 1.0)
    pace_boost = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]
    rest_multiplier, rest_reason = get_rest_multiplier(days_off, p_age)
    
    # Usage Trend
    avg_usage = p_df['usage_proxy'].mean()
    recent_usage = p_df['usage_proxy'].head(3).mean()
    usage_factor = 1.05 if recent_usage > avg_usage else 0.98
    
    model_proj = p_mean * pace_boost * sos_multiplier * rest_multiplier * usage_factor

    col_main, col_side = st.columns([2, 1])

    with col_main:
        # Custom Rest-o-Meter Display
        rest_cols = st.columns(3)
        rest_cols[0].metric("Detected Rest", f"{auto_rest} Days")
        rest_cols[1].metric("Weather/Air", "Indoor (N/A)")
        rest_cols[2].metric("Usage Trend", f"{round((recent_usage/avg_usage - 1)*100, 1)}%", delta_color="normal")

        st.info(f"⚡ **Context:** {rest_reason} | Usage Boost: {round((usage_factor-1)*100)}%")
        
        # Risk Offsets
        risk_offsets = {"Conservative (-115)": -0.4, "Standard (+100)": 0, "Aggressive (+180)": 0.5}
        target_line = round(model_proj + (risk_offsets[risk_pref] * p_std))

        # Main Performance Chart
        last_10 = p_df.head(10).iloc[::-1]
        last_10['hit'] = last_10[stat_category] >= target_line
        
        fig_hits = go.Figure()
        fig_hits.add_trace(go.Bar(x=[f"G{i+1}" for i in range(len(last_10))], y=last_10[stat_category], marker_color=['#00ff96' if hit else '#4a4a4a' for hit in last_10['hit']], name="Stat Value"))
        fig_hits.add_hline(y=target_line, line_dash="dash", line_color="#ff4b4b", annotation_text="Target Line")
        fig_hits.update_layout(title=f"Last 10 vs {target_line}+ {stat_category}", template="plotly_dark", height=300, showlegend=False)
        st.plotly_chart(fig_hits, use_container_width=True)

        # Usage Tracker Chart
        fig_usage = go.Figure()
        fig_usage.add_trace(go.Scatter(x=[f"G{i+1}" for i in range(len(last_10))], y=last_10['usage_proxy'], mode='lines+markers', line=dict(color='#00d4ff', width=3), name="Usage Proxy"))
        fig_usage.update_layout(title="Usage Volume Tracker (Possessions Used)", template="plotly_dark", height=200, margin=dict(t=30, b=0))
        st.plotly_chart(fig_usage, use_container_width=True)

    with col_side:
        st.subheader("📋 Prop Intelligence")
        st.table(pd.DataFrame({
            "Metric": ["Projection", "Season Avg", "SoS Multi"],
            "Value": [round(model_proj, 1), round(p_mean, 1), round(sos_multiplier, 2)]
        }))
        
        st.divider()
        confidence = (last_10['hit'].sum() / len(last_10)) * 100
        st.metric("Model Confidence", f"{int(confidence)}%", delta=f"{round(model_proj - p_mean, 1)} vs Avg")
        
        if confidence >= 70: st.success("🔥 HIGH CONVICTION: OVER")
        elif confidence <= 30: st.error("❄️ HIGH CONVICTION: UNDER")
        else: st.warning("⚖️ NEUTRAL/STAY AWAY")

    st.caption("Usage Proxy calculated via FGA + 0.44*FTA + TOV. High usage indicates primary offensive engine.")
else:
    st.warning(f"No 2025-26 data found for {selected_p}.")
