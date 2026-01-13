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
    # Normalize naming for SGA and others
    match = [p for p in nba_players if p['full_name'].lower() == player_full_name.lower()]
    if not match:
        match = [p for p in nba_players if player_full_name.lower() in p['full_name'].lower()]
    
    if not match: return pd.DataFrame(), None
    p_id = match[0]['id']
    
    log = pd.DataFrame()
    # Attempt multi-season fetch to handle API gaps
    for season in ['2025-26', '2024-25']:
        try:
            log = playergamelog.PlayerGameLog(player_id=p_id, season=season).get_data_frames()[0]
            if not log.empty: break
        except: continue

    # CRITICAL FIX: Check if log is empty before accessing columns
    if log.empty:
        return pd.DataFrame(), None

    team_id = log['TEAM_ID'].iloc[0]
    
    log = log.rename(columns={
        'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers',
        'FGA': 'fga', 'FG_PCT': 'fg_pct', 'FTA': 'fta', 'TOV': 'tov', 'MATCHUP': 'matchup', 'GAME_DATE': 'date'
    })
    log['pra'] = log['points'] + log['rebounds'] + log['assists']
    log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
    return log, team_id

@st.cache_data(ttl=3600)
def get_auto_context(team_id):
    if not team_id: return True, False, "BOS"
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
        games = gamefinder.get_data_frames()[0]
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        
        today = datetime.now()
        upcoming = games[games['GAME_DATE'] >= today].sort_values('GAME_DATE')
        past = games[games['GAME_DATE'] < today].sort_values('GAME_DATE', ascending=False)
        
        if upcoming.empty: return True, False, "BOS"
        
        next_g = upcoming.iloc[0]
        is_home = "vs." in next_g['MATCHUP']
        # Safer opponent extraction
        matchup_parts = next_g['MATCHUP'].split(' ')
        opp_abbr = matchup_parts[-1] if len(matchup_parts) > 1 else "BOS"
        
        is_b2b = False
        if not past.empty:
            days_diff = (next_g['GAME_DATE'] - past.iloc[0]['GAME_DATE']).days
            is_b2b = days_diff <= 1
            
        return is_home, is_b2b, opp_abbr
    except: return True, False, "BOS"

# --- 2. LOGIC ENGINES ---
def calculate_poisson_prob(lambda_val, line):
    return round((1 - poisson.cdf(line, lambda_val)) * 100, 1)

def run_monte_carlo(lambda_val, user_line, iterations=10000):
    simulated_games = np.random.poisson(lambda_val, iterations)
    levels = [max(0, round(l, 1)) for l in [user_line - 1, user_line, user_line + 1]]
    results = [{"Stat Level": f"{l}+", "Hit Frequency": f"{(np.sum(simulated_games >= l)/iterations)*100:.1f}%"} for l in sorted(list(set(levels)))]
    return pd.DataFrame(results), simulated_games

# --- 3. UI RENDERING ---
st.title("🏀 NBA Sharp: Auto-Pilot Suite (v3.2)")
sos_data, _ = load_nba_base_data()

with st.sidebar:
    st.header("🎯 Target Selection")
    search_query = st.text_input("Search Player", "Shai Gilgeous-Alexander")
    all_names = [p['full_name'] for p in players.get_players()]
    filtered = [p for p in all_names if search_query.lower() in p.lower()]
    selected_p = st.selectbox("Confirm Player", filtered if filtered else all_names)
    
    # Fetch data and context with error-handling logic
    p_df, t_id = get_player_data(selected_p)
    auto_home, auto_b2b, auto_opp = get_auto_context(t_id)

    st.divider()
    st.subheader("🎲 Intelligence Controls")
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    user_line = st.number_input(f"Sportsbook Line", value=25.5 if stat_category=="points" else 5.5, step=0.5)
    
    # Validation for selectbox index
    opp_list = sorted(list(sos_data.keys()))
    opp_idx = opp_list.index(auto_opp) if auto_opp in opp_list else 0
    
    selected_opp = st.selectbox("Opponent", opp_list, index=opp_idx)
    is_home = st.toggle("Home Game", value=auto_home)
    is_b2b = st.toggle("Back-to-Back", value=auto_b2b)
    star_out = st.toggle("Star Teammate Out?")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

# --- 4. DATA PROCESSING ---
if not p_df.empty:
    p_mean = p_df[stat_category].mean()
    sos_mult = sos_data.get(selected_opp, 1.0)
    pace_mult = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]
    # Weightings for Home (3%) and B2B (-5%)
    sharp_lambda = p_mean * pace_mult * sos_mult * (1.15 if star_out else 1.0) * (1.03 if is_home else 0.97) * (0.95 if is_b2b else 1.0)
    over_prob = calculate_poisson_prob(sharp_lambda, user_line)

    col_main, col_side = st.columns([2, 1])

    with col_main:
        # MOMENTUM CHART
        st.subheader(f"📈 {selected_p} Trend (Last 10)")
        last_10 = p_df.head(10).iloc[::-1]
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=list(range(1, len(last_10)+1)), y=last_10[stat_category], mode='lines+markers', line=dict(color='#00ff96', width=3)))
        trend_fig.add_hline(y=user_line, line_dash="dash", line_color="#ff4b4b", annotation_text="Line")
        trend_fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(trend_fig, use_container_width=True)

        # SIMULATION
        st.subheader("🎲 Monte Carlo Simulation")
        sim_df, sim_raw = run_monte_carlo(sharp_lambda, user_line)
        st.table(sim_df)

    with col_side:
        st.subheader("📊 Output")
        st.metric("Sharp Projection", round(sharp_lambda, 1))
        st.metric("Win Prob (Over)", f"{over_prob}%")
        st.info(f"Context: {'Home' if is_home else 'Away'} | {'B2B' if is_b2b else 'Rested'} | vs {selected_opp}")
        
        if over_prob > 60: st.success("🔥 VALUE: OVER")
        elif over_prob < 40: st.error("❄️ VALUE: UNDER")

    st.caption("v3.2 | KeyError:TEAM_ID Patched | Auto-Pilot Context")
else:
    st.warning("⚠️ Data could not be retrieved. The NBA API may be experiencing high traffic or the player name is misspelled.")
