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
    try:
        team_stats_raw = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced', season='2025-26'
        ).get_data_frames()[0]
        nba_teams = teams.get_teams()
        id_to_abbr = {t['id']: t['abbreviation'] for t in nba_teams}
        avg_drtg = team_stats_raw['DEF_RATING'].mean()
        
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
        'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers',
        'FGA': 'fga', 'FG_PCT': 'fg_pct', 'FTA': 'fta', 'TOV': 'tov', 'MATCHUP': 'matchup'
    })
    log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
    log['ts_pct'] = log['points'] / (2 * (log['fga'] + 0.44 * log['fta']))
    return log

# --- 2. ADVANCED LOGIC ENGINES ---
def calculate_poisson_prob(lambda_val, line):
    prob_over = 1 - poisson.cdf(line, lambda_val)
    return round(prob_over * 100, 1)

def run_monte_carlo(lambda_val, user_line, iterations=10000):
    """Simulates 10k games based on the adjusted lambda (mean)."""
    simulated_games = np.random.poisson(lambda_val, iterations)
    
    # Calculate intervals for the table
    levels = [user_line - 2, user_line - 1, user_line, user_line + 1, user_line + 2]
    # Ensure three-pointer lines don't go negative or look weird
    levels = [max(0, round(l, 1)) for l in levels]
    
    results = []
    for level in sorted(list(set(levels))):
        hits = np.sum(simulated_games >= level)
        pct = (hits / iterations) * 100
        results.append({"Stat Level": f"{level}+", "Hit Frequency": f"{pct:.1f}%"})
    
    return pd.DataFrame(results), simulated_games

# --- 3. UI RENDERING ---
st.title("🏀 NBA Sharp: Intelligence Hub (v2.5)")
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
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers"])
    default_line = 25.5 if stat_category == "points" else 2.5
    user_line = st.number_input(f"Sportsbook Line ({stat_category})", value=float(default_line), step=0.5)
    
    star_out = st.toggle("Star Teammate Out?", help="Applies Bayesian Usage Split.")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

# --- 4. DATA PROCESSING ---
p_df = get_player_data(selected_p)

if not p_df.empty:
    p_mean = p_df[stat_category].mean()
    sos_mult = sos_data.get(selected_opp, 1.0)
    pace_mult = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]
    usage_mult = 1.15 if star_out else 1.0
    
    sharp_lambda = p_mean * pace_mult * sos_mult * usage_mult
    over_prob = calculate_poisson_prob(sharp_lambda, user_line)

    col_main, col_side = st.columns([2, 1])

    with col_main:
        # Charting Distribution
        x_vals = np.arange(max(0, int(user_line - 10)), int(user_line + 15))
        y_vals = [poisson.pmf(x, sharp_lambda) for x in x_vals]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, fill='tozeroy', name='Prob. Density', line_color='#00ff96'))
        fig.add_vline(x=user_line, line_dash="dash", line_color="#ff4b4b", annotation_text="Target Line")
        fig.update_layout(title=f"Distribution: {selected_p} {stat_category.replace('_',' ').title()}", template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

        # 10k Simulation Table
        st.subheader("🎲 10,000 Game Monte Carlo Simulation")
        sim_df, sim_raw = run_monte_carlo(sharp_lambda, user_line)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.table(sim_df)
        with c2:
            st.write(f"**Mean Result:** {np.mean(sim_raw):.2f}")
            st.write(f"**High (90th Percentile):** {np.percentile(sim_raw, 90)}")
            st.write(f"**Low (10th Percentile):** {np.percentile(sim_raw, 10)}")

    with col_side:
        st.subheader("📊 Model Output")
        st.metric("Sharp Projection", round(sharp_lambda, 1))
        st.metric("Win Prob (Over)", f"{over_prob}%")
        
        st.divider()
        st.info(f"Simulation shows this player hits **{user_line}+** in **{over_prob}%** of virtual outcomes.")
        
        if over_prob > 60:
            st.success("🔥 **VALUE DETECTED: OVER**")
        elif over_prob < 40:
            st.error("❄️ **VALUE DETECTED: UNDER**")

    st.caption(f"v2.5 | Monte Carlo Engine (10k trials) | Added 3PM Support")
else:
    st.warning("No data found.")
