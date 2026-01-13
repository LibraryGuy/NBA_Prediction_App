import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, leaguegamefinder

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
    
    if not player: return pd.DataFrame(), None
    p_id = player[0]['id']
    
    try:
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if log.empty:
            log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]
        
        team_id = log['TEAM_ID'].iloc[0]
    except: return pd.DataFrame(), None

    log = log.rename(columns={
        'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers',
        'FGA': 'fga', 'FG_PCT': 'fg_pct', 'FTA': 'fta', 'TOV': 'tov', 'MATCHUP': 'matchup', 'GAME_DATE': 'date'
    })
    log['pra'] = log['points'] + log['rebounds'] + log['assists']
    log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
    log['ts_pct'] = log['points'] / (2 * (log['fga'] + 0.44 * log['fta']))
    return log, team_id

@st.cache_data(ttl=3600)
def get_auto_context(team_id):
    """Detects Home/Away and B2B status for the upcoming game."""
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
        games = gamefinder.get_data_frames()[0]
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        today = datetime.now()
        
        upcoming = games[games['GAME_DATE'] >= today].sort_values('GAME_DATE')
        past = games[games['GAME_DATE'] < today].sort_values('GAME_DATE', ascending=False)
        
        if upcoming.empty: return True, False, "UNK"
        
        next_g = upcoming.iloc[0]
        is_home = "vs." in next_g['MATCHUP']
        opp_abbr = next_g['MATCHUP'].split(' ')[-1]
        
        is_b2b = False
        if not past.empty:
            days_diff = (next_g['GAME_DATE'] - past.iloc[0]['GAME_DATE']).days
            is_b2b = days_diff <= 1
            
        return is_home, is_b2b, opp_abbr
    except: return True, False, "BOS"

# --- 2. ADVANCED LOGIC ENGINES ---
def calculate_poisson_prob(lambda_val, line):
    return round((1 - poisson.cdf(line, lambda_val)) * 100, 1)

def run_monte_carlo(lambda_val, user_line, iterations=10000):
    simulated_games = np.random.poisson(lambda_val, iterations)
    levels = [max(0, round(l, 1)) for l in [user_line - 1, user_line, user_line + 1]]
    results = [{"Stat Level": f"{l}+", "Hit Frequency": f"{(np.sum(simulated_games >= l)/iterations)*100:.1f}%"} for l in sorted(list(set(levels)))]
    return pd.DataFrame(results), simulated_games

# --- 3. UI RENDERING ---
st.title("🏀 NBA Sharp: Auto-Pilot Suite (v3.0)")
sos_data, league_avg_drtg = load_nba_base_data()

with st.sidebar:
    st.header("🎯 Target Selection")
    search_query = st.text_input("Search Player", "Jayson Tatum")
    all_names = [p['full_name'] for p in players.get_players()]
    filtered = [p for p in all_names if search_query.lower() in p.lower()]
    selected_p = st.selectbox("Confirm Player", filtered if filtered else all_names)
    
    p_df, t_id = get_player_data(selected_p)
    auto_home, auto_b2b, auto_opp = get_auto_context(t_id) if t_id else (True, False, "BOS")

    st.divider()
    st.subheader("🎲 Auto-Context Engine")
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    user_line = st.number_input(f"Sportsbook Line", value=25.5 if stat_category=="points" else 5.5, step=0.5)
    
    selected_opp = st.selectbox("Opponent", sorted(list(sos_data.keys())), index=sorted(list(sos_data.keys())).index(auto_opp) if auto_opp in sos_data else 0)
    is_home = st.toggle("Home Game", value=auto_home)
    is_b2b = st.toggle("Back-to-Back (Fatigue)", value=auto_b2b)
    star_out = st.toggle("Star Teammate Out?")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

# --- 4. DATA PROCESSING ---
if not p_df.empty:
    p_mean = p_df[stat_category].mean()
    sos_mult = sos_data.get(selected_opp, 1.0)
    pace_mult = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]
    # Pro adjustments: Home/Away Split (3%) and B2B Fatigue (5%)
    sharp_lambda = p_mean * pace_mult * sos_mult * (1.15 if star_out else 1.0) * (1.03 if is_home else 0.97) * (0.95 if is_b2b else 1.0)
    over_prob = calculate_poisson_prob(sharp_lambda, user_line)

    col_main, col_side = st.columns([2, 1])

    with col_main:
        # NEW: Last 10 Games Heater/Slump Chart
        st.subheader("🔥 Momentum: Last 10 Games Trend")
        last_10 = p_df.head(10).iloc[::-1] # Reverse to chronological
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=list(range(1, 11)), y=last_10[stat_category], mode='lines+markers', name='Actual Result', line=dict(color='#00ff96', width=3)))
        trend_fig.add_hline(y=sharp_lambda, line_dash="dash", line_color="#ffcc00", annotation_text="Model Proj")
        trend_fig.add_hline(y=user_line, line_dash="dot", line_color="#ff4b4b", annotation_text="Vegas Line")
        trend_fig.update_layout(title="Heater/Slump Analysis (Last 10 Games)", template="plotly_dark", height=300, xaxis_title="Recent Games", yaxis_title=stat_category.title())
        st.plotly_chart(trend_fig, use_container_width=True)

        st.subheader("🎲 10,000 Game Monte Carlo Simulation")
        sim_df, sim_raw = run_monte_carlo(sharp_lambda, user_line)
        c1, c2 = st.columns(2)
        with c1: st.table(sim_df)
        with c2:
            st.metric("Simulated Mean", round(np.mean(sim_raw), 2))
            st.write(f"**90th Percentile:** {np.percentile(sim_raw, 90)}")

    with col_side:
        st.subheader("📊 Model Output")
        st.metric("Sharp Projection", round(sharp_lambda, 1))
        st.metric("Win Prob (Over)", f"{over_prob}%")
        
        st.divider()
        status = "HOME" if is_home else "AWAY"
        rest = "FATIGUED (B2B)" if is_b2b else "RESTED"
        st.info(f"**Context:** {status} | {rest} | vs {selected_opp}")
        
        if over_prob > 60: st.success("🔥 **VALUE DETECTED: OVER**")
        elif over_prob < 40: st.error("❄️ **VALUE DETECTED: UNDER**")

    st.caption("v3.0 | Auto-Context Engine | Last 10 Momentum Chart")
else:
    st.warning("No data found.")
