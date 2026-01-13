import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime, timedelta
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# --- 1. SETTINGS & CONFIG ---
st.set_page_config(page_title="NBA Sharp Pro Hub", layout="wide", page_icon="🏀")

@st.cache_data(ttl=3600)
def load_nba_base_data():
    """Loads league-wide defensive ratings for Bayesian SoS multipliers."""
    try:
        team_stats_raw = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced', season='2025-26'
        ).get_data_frames()[0]
        nba_teams = teams.get_teams()
        id_to_abbr = {t['id']: t['abbreviation'] for t in nba_teams}
        avg_drtg = team_stats_raw['DEF_RATING'].mean()
        
        # Bayesian Adjustment: Mix team DRtg with League Average (Shrinkage)
        # Weight recent performance (if available) higher, but here we use a stability constant
        sos_map = {id_to_abbr[row['TEAM_ID']]: (row['DEF_RATING'] * 0.8 + avg_drtg * 0.2) / avg_drtg 
                   for _, row in team_stats_raw.iterrows() if id_to_abbr.get(row['TEAM_ID'])}
        return sos_map, avg_drtg
    except: return {}, 115.0

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
        'FGA': 'fga', 'FG_PCT': 'fg_pct', 'FTA': 'fta', 'TOV': 'tov', 'MATCHUP': 'matchup'
    })
    # Feature Engineering for Regression Analysis
    log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
    log['ts_pct'] = log['points'] / (2 * (log['fga'] + 0.44 * log['fta']))
    return log

# --- 2. ADVANCED LOGIC ENGINES ---
def calculate_poisson_prob(lambda_val, line):
    """Calculates the mathematical probability of hitting the OVER."""
    # Probability of X > line is 1 - CDF(line)
    prob_over = 1 - poisson.cdf(line, lambda_val)
    return round(prob_over * 100, 1)

def get_regression_signal(df, stat_cat):
    """Detects if a player is shooting unsustainably well (Shot Quality Proxy)."""
    recent_ts = df['ts_pct'].head(5).mean()
    season_ts = df['ts_pct'].mean()
    
    if recent_ts > season_ts * 1.15: return "⚠️ Regression: Negative (Overperforming)"
    if recent_ts < season_ts * 0.85: return "✅ Regression: Positive (Due for Bounce)"
    return "⚖️ Efficiency: Stable"

# --- 3. UI RENDERING ---
st.title("🏀 NBA Sharp: Intelligence Hub (v2.4)")
sos_data, league_avg_drtg = load_nba_base_data()

with st.sidebar:
    st.header("🎯 Target Selection")
    search_query = st.text_input("Search Player", "Jayson Tatum")
    all_names = [p['full_name'] for p in players.get_players()]
    filtered = [p for p in all_names if search_query.lower() in p.lower()]
    selected_p = st.selectbox("Confirm Player", filtered if filtered else all_names)
    selected_opp = st.selectbox("Opponent", sorted(list(sos_data.keys())) if sos_data else ["BOS"])

    st.divider()
    st.subheader("🎲 Contextual Inputs")
    user_line = st.number_input("Sportsbook Line", value=25.5, step=0.5)
    star_out = st.toggle("Star Teammate Out?", help="Applies Bayesian Usage Split (+15% volume).")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

# --- 4. DATA PROCESSING ---
p_df = get_player_data(selected_p)

if not p_df.empty:
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists"])
    
    # 1. Bayesian Baseline
    p_mean = p_df[stat_category].mean()
    sos_mult = sos_data.get(selected_opp, 1.0)
    pace_mult = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]
    
    # 2. On/Off Usage Split Logic
    usage_mult = 1.15 if star_out else 1.0
    
    # 3. Final Projected Lambda (Mean)
    sharp_lambda = p_mean * pace_mult * sos_mult * usage_mult
    
    # 4. Poisson Probability
    over_prob = calculate_poisson_prob(sharp_lambda, user_line)
    reg_signal = get_regression_signal(p_df, stat_category)

    col_main, col_side = st.columns([2, 1])

    with col_main:
        # Probabilistic Hit Rate Chart
        x_vals = np.arange(max(0, int(user_line - 15)), int(user_line + 15))
        y_vals = [poisson.pmf(x, sharp_lambda) for x in x_vals]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, fill='tozeroy', name='Prob. Density', line_color='#00ff96'))
        fig.add_vline(x=user_line, line_dash="dash", line_color="#ff4b4b", annotation_text="Vegas Line")
        fig.update_layout(title=f"Probability Distribution for {selected_p} {stat_category.title()}", template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.subheader("📊 Model Output")
        st.metric("Sharp Projection", round(sharp_lambda, 1))
        st.metric("Win Probability (Over)", f"{over_prob}%")
        
        st.divider()
        st.subheader("🧠 Analytics signals")
        st.info(f"**Regression:** {reg_signal}")
        
        # Confidence logic based on Probability and Regression
        if over_prob > 60 and "Positive" in reg_signal:
            st.success("🔥 **ULTRA CONVICTION OVER**")
        elif over_prob < 40 and "Negative" in reg_signal:
            st.error("❄️ **ULTRA CONVICTION UNDER**")
        else:
            st.warning("⚖️ High Variance: Tread Carefully")

    st.caption(f"v2.4 | Poisson Modeling | Bayesian SoS ({selected_opp}) | Regression Signal Active")
else:
    st.warning("No data found.")
