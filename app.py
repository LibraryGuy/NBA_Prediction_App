import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from scipy.stats import poisson
from datetime import datetime
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, playernextngames, commonplayerinfo, commonteamroster

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
        abbr_to_id = {t['abbreviation']: t['id'] for t in nba_teams}
        avg_drtg = team_stats_raw['DEF_RATING'].mean()
        
        sos_map = {id_to_abbr[row['TEAM_ID']]: (row['DEF_RATING'] * 0.8 + avg_drtg * 0.2) / avg_drtg 
                   for _, row in team_stats_raw.iterrows() if id_to_abbr.get(row['TEAM_ID'])}
        return sos_map, abbr_to_id
    except:
        fallback_teams = ["BOS", "GSW", "LAL", "OKC", "DET", "MIL", "PHX", "DAL", "NYK", "PHI"]
        return {t: 1.0 for t in fallback_teams}, {"BOS": 1610612738, "GSW": 1610612744}

@st.cache_data(ttl=600)
def get_player_data(player_input, is_id=False):
    if not is_id:
        nba_players = players.get_players()
        match = [p for p in nba_players if p['full_name'].lower() == player_input.lower()]
        if not match:
            match = [p for p in nba_players if player_input.lower() in p['full_name'].lower()]
        if not match: return pd.DataFrame(), None, None
        p_id = match[0]['id']
    else:
        p_id = player_input
    
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
        team_abbr = info['TEAM_ABBREVIATION'].iloc[0]
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if log.empty:
            log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]
    except: return pd.DataFrame(), None, None

    log = log.rename(columns={'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'FGA': 'fga', 'FG_PCT': 'fg_pct', 'FTA': 'fta', 'TOV': 'tov'})
    log['pra'] = log['points'] + log['rebounds'] + log['assists']
    log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
    log['pps'] = log['points'] / log['fga'].replace(0, 1)
    return log, p_id, team_abbr

# --- 2. ENGINES ---
def calculate_sharp_lambda(p_mean, pace_mult, sos_mult, star_out, is_home, is_b2b):
    return p_mean * pace_mult * sos_mult * (1.15 if star_out else 1.0) * (1.03 if is_home else 0.97) * (0.95 if is_b2b else 1.0)

def calculate_poisson_prob(lambda_val, line):
    return round((1 - poisson.cdf(line, lambda_val)) * 100, 1)

def run_monte_carlo(lambda_val, user_line, iterations=10000):
    simulated_games = np.random.poisson(lambda_val, iterations)
    levels = [max(0, round(l, 1)) for l in [user_line - 1, user_line, user_line + 1]]
    results = [{"Stat Level": f"{l}+", "Hit Frequency": f"{(np.sum(simulated_games >= l)/iterations)*100:.1f}%"} 
               for l in sorted(list(set(levels)))]
    return pd.DataFrame(results), simulated_games

# --- 3. UI RENDERING ---
st.title("🏀 NBA Sharp Pro Hub")
sos_data, abbr_to_id = load_nba_base_data()

if 'auto_opp' not in st.session_state: st.session_state.auto_opp = "BOS"
if 'auto_home' not in st.session_state: st.session_state.auto_home = True
if 'auto_b2b' not in st.session_state: st.session_state.auto_b2b = False

with st.sidebar:
    st.header("🎮 Analysis Mode")
    mode = st.radio("Switch View", ["Single Player", "Team Scout Radar"])
    st.divider()
    
    if mode == "Single Player":
        search_query = st.text_input("Search Player", "Shai Gilgeous-Alexander")
        all_names = [p['full_name'] for p in players.get_players()]
        filtered = [p for p in all_names if search_query.lower() in p.lower()]
        selected_p = st.selectbox("Confirm Player", filtered if filtered else ["No Player Found"])
        p_df, p_id, team_abbr = get_player_data(selected_p)
    else:
        selected_team_abbr = st.selectbox("Select Team", sorted(list(abbr_to_id.keys())))
        p_df = pd.DataFrame() 

    st.subheader("🎲 Game Context")
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    
    opp_options = sorted(list(sos_data.keys()))
    opp_idx = opp_options.index(st.session_state.auto_opp) if st.session_state.auto_opp in opp_options else 0
    selected_opp = st.selectbox("Opponent", opp_options, index=opp_idx)
    
    is_home = st.toggle("Home Game", value=st.session_state.auto_home)
    is_b2b = st.toggle("Back-to-Back", value=st.session_state.auto_b2b)
    star_out = st.toggle("Star Teammate Out?")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

# --- 4. MAIN DASHBOARD ---
sos_mult = sos_data.get(selected_opp, 1.0)
pace_mult = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]

if mode == "Single Player" and not p_df.empty:
    p_mean = p_df[stat_category].mean()
    sharp_lambda = calculate_sharp_lambda(p_mean, pace_mult, sos_mult, star_out, is_home, is_b2b)
    
    col_main, col_side = st.columns([2, 1])
    with col_main:
        # Last 10 Trend
        last_10 = p_df.head(10).iloc[::-1]
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=list(range(1, 11)), y=last_10[stat_category], mode='lines+markers', line=dict(color='#00ff96', width=3)))
        trend_fig.update_layout(template="plotly_dark", height=300, title=f"Last 10: {stat_category.capitalize()}")
        st.plotly_chart(trend_fig, use_container_width=True)

    with col_side:
        st.subheader("📊 Model Output")
        st.metric("Sharp Projection", round(sharp_lambda, 1))
        st.metric("Context SOS", f"{round(sos_mult, 2)}x")

elif mode == "Team Scout Radar":
    st.subheader(f"📡 {selected_team_abbr} Roster Radar")
    if st.button("🚀 Run Team Scan"):
        t_id = abbr_to_id.get(selected_team_abbr)
        if t_id:
            with st.status("Scanning Roster...", expanded=True) as status:
                roster = commonteamroster.CommonTeamRoster(team_id=t_id).get_data_frames()[0].head(10)
                results = []
                for _, row in roster.iterrows():
                    t_df, _, _ = get_player_data(row['PLAYER_ID'], is_id=True)
                    if not t_df.empty:
                        t_mean = t_df[stat_category].mean()
                        t_proj = calculate_sharp_lambda(t_mean, pace_mult, sos_mult, star_out, is_home, is_b2b)
                        results.append({
                            "Player": row['PLAYER'],
                            "Avg": round(t_mean, 1),
                            "Proj": round(t_proj, 1),
                            "Context Bump": round(t_proj - t_mean, 1)
                        })
                    time.sleep(0.1) # Rate limit safety
                status.update(label="Scan Complete!", state="complete")

            res_df = pd.DataFrame(results).sort_values(by="Context Bump", ascending=False)
            
            # --- FIX: SAFE STYLING BLOCK ---
            try:
                st.dataframe(
                    res_df.style.background_gradient(subset=['Context Bump'], cmap='RdYlGn'), 
                    use_container_width=True
                )
            except ImportError:
                st.warning("⚠️ Heatmap disabled. Add `matplotlib` to requirements.txt for full visuals.")
                st.dataframe(res_df, use_container_width=True)
