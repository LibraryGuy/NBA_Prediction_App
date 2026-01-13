import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, leaguegamefinder
from datetime import datetime

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
    if not player: return pd.DataFrame(), None, None
    
    p_id = player[0]['id']
    try:
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if log.empty:
            log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]
        
        # Determine team for schedule lookup
        team_id = log['TEAM_ID'].iloc[0]
        team_abbr = log['MATCHUP'].iloc[0][:3] # Simplified team abbr extraction
    except: return pd.DataFrame(), None, None

    log = log.rename(columns={
        'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers',
        'FGA': 'fga', 'FG_PCT': 'fg_pct', 'FTA': 'fta', 'TOV': 'tov', 'MATCHUP': 'matchup'
    })
    log['pra'] = log['points'] + log['rebounds'] + log['assists']
    log['ts_pct'] = log['points'] / (2 * (log['fga'] + 0.44 * log['fta']))
    return log, team_id, team_abbr

@st.cache_data(ttl=3600)
def get_upcoming_context(team_id):
    """Automates Home/Away and B2B detection."""
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
        games = gamefinder.get_data_frames()[0]
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        today = datetime.now()
        
        # Get next game and previous game
        upcoming = games[games['GAME_DATE'] >= today].sort_values('GAME_DATE')
        past = games[games['GAME_DATE'] < today].sort_values('GAME_DATE', ascending=False)
        
        if upcoming.empty: return True, False, "N/A" # Default if no games found
        
        next_game = upcoming.iloc[0]
        last_game = past.iloc[0] if not past.empty else None
        
        is_home = "vs." in next_game['MATCHUP']
        is_b2b = False
        if last_game is not None:
            days_diff = (next_game['GAME_DATE'] - last_game['GAME_DATE']).days
            is_b2b = days_diff <= 1
            
        opp = next_game['MATCHUP'][-3:] # Extract Opponent
        return is_home, is_b2b, opp
    except:
        return True, False, "UNK"

# --- 2. LOGIC ENGINES ---
def calculate_poisson_prob(lambda_val, line):
    prob_over = 1 - poisson.cdf(line, lambda_val)
    return round(prob_over * 100, 1)

def run_monte_carlo(lambda_val, user_line, iterations=10000):
    simulated_games = np.random.poisson(lambda_val, iterations)
    levels = [max(0, round(l, 1)) for l in [user_line - 1, user_line, user_line + 1]]
    results = [{"Stat Level": f"{l}+", "Hit Frequency": f"{(np.sum(simulated_games >= l)/iterations)*100:.1f}%"} for l in levels]
    return pd.DataFrame(results), simulated_games

# --- 3. UI RENDERING ---
st.title("🏀 NBA Sharp: Auto-Pilot Suite (v3.0)")
sos_data, _ = load_nba_base_data()

with st.sidebar:
    st.header("🎯 Target Selection")
    search_query = st.text_input("Search Player", "Jayson Tatum")
    all_names = [p['full_name'] for p in players.get_players()]
    filtered = [p for p in all_names if search_query.lower() in p.lower()]
    selected_p = st.selectbox("Confirm Player", filtered if filtered else all_names)
    
    # FETCH DATA & CONTEXT
    p_df, t_id, t_abbr = get_player_data(selected_p)
    auto_home, auto_b2b, auto_opp = get_upcoming_context(t_id) if t_id else (True, False, "UNK")

    st.divider()
    st.subheader("🎲 Intelligence Context")
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    user_line = st.number_input(f"Sportsbook Line", value=25.5, step=0.5)
    
    # Display automated values but allow override
    st.write(f"**Auto-Detected Opponent:** {auto_opp}")
    is_home = st.toggle("Home Game?", value=auto_home)
    is_b2b = st.toggle("Back-to-Back?", value=auto_b2b)
    star_out = st.toggle("Star Teammate Out?")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

# --- 4. DATA PROCESSING ---
if not p_df.empty:
    p_mean = p_df[stat_category].mean()
    sos_mult = sos_data.get(auto_opp, 1.0)
    pace_mult = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]
    sharp_lambda = p_mean * pace_mult * sos_mult * (1.15 if star_out else 1.0) * (1.03 if is_home else 0.97) * (0.95 if is_b2b else 1.0)
    over_prob = calculate_poisson_prob(sharp_lambda, user_line)

    col_main, col_side = st.columns([2, 1])
    with col_main:
        # Charting & Simulation
        x_vals = np.arange(max(0, int(user_line - 10)), int(user_line + 15))
        y_vals = [poisson.pmf(x, sharp_lambda) for x in x_vals]
        fig = go.Figure(go.Scatter(x=x_vals, y=y_vals, fill='tozeroy', line_color='#00ff96'))
        fig.add_vline(x=user_line, line_dash="dash", line_color="#ff4b4b")
        fig.update_layout(title=f"Distribution: {selected_p}", template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎲 10,000 Game Monte Carlo")
        sim_df, sim_raw = run_monte_carlo(sharp_lambda, user_line)
        st.table(sim_df)

    with col_side:
        st.subheader("📊 Model Output")
        st.metric("Win Prob (Over)", f"{over_prob}%")
        st.info(f"Auto-detected {t_abbr} {'Home' if auto_home else 'Away'} game. B2B status: {'ACTIVE' if auto_b2b else 'None'}.")

    st.caption("v3.0 | Auto-Pilot Context Active")
