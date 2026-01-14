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
    # ULTIMATE SAFETY: All 30 teams mapped to IDs (Ensures dropdowns never shrink)
    all_30_teams = {
        'ATL': 1610612737, 'BOS': 1610612738, 'CLE': 1610612739, 'NOP': 1610612740,
        'CHI': 1610612741, 'DAL': 1610612742, 'DEN': 1610612743, 'GSW': 1610612744,
        'HOU': 1610612745, 'LAC': 1610612746, 'LAL': 1610612747, 'MIA': 1610612748,
        'MIL': 1610612749, 'MIN': 1610612750, 'BKN': 1610612751, 'NYK': 1610612752,
        'ORL': 1610612753, 'IND': 1610612754, 'PHI': 1610612755, 'PHX': 1610612756,
        'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'OKC': 1610612760,
        'TOR': 1610612761, 'UTA': 1610612762, 'MEM': 1610612763, 'WAS': 1610612764,
        'DET': 1610612765, 'CHA': 1610612766
    }
    
    try:
        # ATTEMPT LIVE 2025-26 STATS (Approx. 40 games in)
        team_stats_raw = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced', season='2025-26'
        ).get_data_frames()[0]
        
        avg_drtg = team_stats_raw['DEF_RATING'].mean()
        # Create SOS mapping based on current Defensive Rating
        sos_map = {}
        for _, row in team_stats_raw.iterrows():
            abbr = [k for k, v in all_30_teams.items() if v == row['TEAM_ID']][0]
            # SOS > 1 means harder defense than average
            sos_map[abbr] = (row['DEF_RATING'] * 0.7 + avg_drtg * 0.3) / avg_drtg
            
        return sos_map, all_30_teams
        
    except Exception as e:
        # Fallback to neutral SOS for all 30 teams (Avoids the 2-team dropdown bug)
        return {abbr: 1.0 for abbr in all_30_teams.keys()}, all_30_teams

@st.cache_data(ttl=600)
def get_player_data(player_input, is_id=False):
    # Locate Player
    if not is_id:
        nba_players = players.get_players()
        match = [p for p in nba_players if player_input.lower() in p['full_name'].lower()]
        if not match: return pd.DataFrame(), None, None
        p_id = match[0]['id']
        p_name = match[0]['full_name']
    else:
        p_id = player_input
    
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
        team_abbr = info['TEAM_ABBREVIATION'].iloc[0]
        # PRIMARY: 2025-26 Season Data
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if log.empty:
            log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]
            
        if not log.empty:
            log = log.rename(columns={'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'FGA': 'fga', 'FG_PCT': 'fg_pct', 'FTA': 'fta', 'TOV': 'tov'})
            log['pra'] = log['points'] + log['rebounds'] + log['assists']
            log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
            log['pps'] = log['points'] / log['fga'].replace(0, 1)
        return log, p_id, team_abbr
    except:
        return pd.DataFrame(), None, None

# --- 2. ENGINES ---
def calculate_sharp_lambda(p_mean, pace_mult, sos_mult, star_out, is_home, is_b2b):
    return p_mean * pace_mult * sos_mult * (1.12 if star_out else 1.0) * (1.03 if is_home else 0.97) * (0.95 if is_b2b else 1.0)

def calculate_poisson_prob(lambda_val, line):
    return round((1 - poisson.cdf(line - 0.5, lambda_val)) * 100, 1)

def run_monte_carlo(lambda_val, user_line, iterations=10000):
    simulated_games = np.random.poisson(lambda_val, iterations)
    levels = [user_line - 1, user_line, user_line + 1]
    results = [{"Stat Level": f"{max(0, round(l, 1))}+", "Hit Frequency": f"{(np.sum(simulated_games >= l)/iterations)*100:.1f}%"} 
               for l in sorted(list(set(levels)))]
    return pd.DataFrame(results), simulated_games

# --- 3. UI RENDERING ---
st.title("🏀 NBA Sharp Pro Hub (v2.9)")
sos_data, abbr_to_id = load_nba_base_data()

with st.sidebar:
    st.header("🎮 Analysis Mode")
    mode = st.radio("Switch View", ["Single Player", "Team Scout Radar"])
    st.divider()
    
    if mode == "Single Player":
        search_query = st.text_input("Search Player", "Shai Gilgeous-Alexander")
        p_df, p_id, team_abbr = get_player_data(search_query)
    else:
        selected_team_abbr = st.selectbox("Select Team (All 30 Available)", sorted(list(abbr_to_id.keys())))
        p_df = pd.DataFrame() 

    st.subheader("🎲 Game Context")
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    selected_opp = st.selectbox("Opponent (All 30 Available)", sorted(list(sos_data.keys())))
    is_home = st.toggle("Home Game", value=True)
    is_b2b = st.toggle("Back-to-Back")
    star_out = st.toggle("Star Teammate Out?")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

# --- 4. MAIN DASHBOARD ---
sos_mult = sos_data.get(selected_opp, 1.0)
pace_mult = {"Snail": 0.94, "Balanced": 1.0, "Track Meet": 1.06}[pace_script]

if mode == "Single Player" and not p_df.empty:
    p_mean = p_df[stat_category].mean()
    sharp_lambda = calculate_sharp_lambda(p_mean, pace_mult, sos_mult, star_out, is_home, is_b2b)
    
    col_main, col_side = st.columns([2, 1])
    with col_main:
        # 1. Volume vs Efficiency
        st.subheader("📊 Volume vs. Efficiency Matrix")
        eff_fig = go.Figure(go.Scatter(x=p_df['usage'].head(15), y=p_df['pps'].head(15), mode='markers+text', text=p_df['points'].head(15), marker=dict(size=12, color=p_df['points'], colorscale='Viridis')))
        eff_fig.update_layout(template="plotly_dark", height=300, xaxis_title="Usage", yaxis_title="Efficiency (PPS)")
        st.plotly_chart(eff_fig, use_container_width=True)

        # 2. Monte Carlo
        st.subheader("🎲 Monte Carlo Distribution")
        user_line = st.number_input("Sportsbook Line", value=float(round(p_mean, 1)), step=0.5)
        sim_df, sim_raw = run_monte_carlo(sharp_lambda, user_line)
        mc_fig = go.Figure(go.Histogram(x=sim_raw, nbinsx=30, marker_color='#00ff96', opacity=0.7, histnorm='probability'))
        mc_fig.add_vline(x=user_line, line_width=3, line_dash="dash", line_color="#ff4b4b")
        mc_fig.update_layout(template="plotly_dark", height=250)
        st.plotly_chart(mc_fig, use_container_width=True)
        st.table(sim_df)

    with col_side:
        st.subheader("📊 Model Output")
        st.metric("Sharp Projection", round(sharp_lambda, 1))
        st.metric("Win Prob (Over)", f"{calculate_poisson_prob(sharp_lambda, user_line)}%")
        st.info("Current projection uses live 2025-26 data for SOS and player averages.")

elif mode == "Team Scout Radar":
    st.subheader(f"📡 {selected_team_abbr} Radar Scan")
    if st.button("🚀 Run Full Roster Analysis"):
        t_id = abbr_to_id.get(selected_team_abbr)
        with st.status(f"Scanning 2025-26 data for {selected_team_abbr}...", expanded=True) as status:
            roster = commonteamroster.CommonTeamRoster(team_id=t_id).get_data_frames()[0].head(12)
            results = []
            for _, row in roster.iterrows():
                p_log, _, _ = get_player_data(row['PLAYER_ID'], is_id=True)
                if not p_log.empty:
                    m = p_log[stat_category].mean()
                    p = calculate_sharp_lambda(m, pace_mult, sos_mult, star_out, is_home, is_b2b)
                    results.append({"Player": row['PLAYER'], "Avg": round(m, 1), "Proj": round(p, 1), "Edge": round(p - m, 1)})
                time.sleep(0.1)
            status.update(label="Analysis Complete", state="complete")
        
        res_df = pd.DataFrame(results).sort_values(by="Edge", ascending=False)
        st.dataframe(res_df.style.background_gradient(subset=['Edge'], cmap='RdYlGn'), use_container_width=True)
