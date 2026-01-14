import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from scipy.stats import poisson
from datetime import datetime
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, commonteamroster, playernextngames

# --- 1. SETTINGS & STYLES ---
st.set_page_config(page_title="NBA Sharp Pro Hub v3.1", layout="wide", page_icon="🏀")

@st.cache_data(ttl=3600)
def load_nba_base_data():
    all_30 = {
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
        # Fetching current 2025-26 Season Stats for all 30 teams
        team_stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced', season='2025-26').get_data_frames()[0]
        avg_drtg = team_stats['DEF_RATING'].mean()
        sos_map = {}
        for _, row in team_stats.iterrows():
            abbr = [k for k, v in all_30.items() if v == row['TEAM_ID']][0]
            sos_map[abbr] = row['DEF_RATING'] / avg_drtg
        return sos_map, all_30
    except:
        return {k: 1.0 for k in all_30.keys()}, all_30

@st.cache_data(ttl=600)
def get_player_data(player_input, is_id=False):
    if not is_id:
        nba_players = players.get_players()
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
        
        if not log.empty:
            log = log.rename(columns={'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'FGA': 'fga', 'FTA': 'fta', 'TOV': 'tov'})
            log['pra'] = log['points'] + log['rebounds'] + log['assists']
            log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
            log['pps'] = log['points'] / log['fga'].replace(0, 1)
        return log, p_id, team_abbr
    except: return pd.DataFrame(), None, None

# --- 2. ENGINES ---
def calculate_sharp_lambda(p_mean, pace_mult, sos_mult, star_out, is_home, is_b2b):
    # SOS > 1 means defense is worse (easier scoring), hence 2 - sos_mult logic or direct mult
    # Using direct mult here assuming sos_mult = Opp_Drtg / Avg_Drtg (Higher = easier)
    return p_mean * pace_mult * sos_mult * (1.15 if star_out else 1.0) * (1.03 if is_home else 0.97) * (0.95 if is_b2b else 1.0)

def run_monte_carlo(lambda_val, user_line, iterations=10000):
    sims = np.random.poisson(lambda_val, iterations)
    over_prob = (np.sum(sims >= user_line) / iterations) * 100
    return sims, over_prob

# --- 3. DASHBOARD ---
sos_data, team_map = load_nba_base_data()

with st.sidebar:
    st.header("🎮 Operation Mode")
    mode = st.radio("View", ["Single Player", "Team Scout Radar"])
    st.divider()
    stat_cat = st.selectbox("Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    opp_list = sorted(list(team_map.keys()))
    selected_opp = st.selectbox("Opponent", opp_list, index=opp_list.index("SAC") if "SAC" in opp_list else 0)
    is_home = st.toggle("Home Game", value=True)
    is_b2b = st.toggle("Back-to-Back")
    star_out = st.toggle("Star Out? (Usage Boost)")
    pace_script = st.select_slider("Pace", options=[0.92, 1.0, 1.08], value=1.0)

# --- 4. MAIN LOGIC ---
sos_mult = sos_data.get(selected_opp, 1.0)

if mode == "Single Player":
    search_q = st.text_input("Search Player", "Jalen Brunson")
    p_df, p_id, team_abbr = get_player_data(search_q)
    
    if not p_df.empty:
        p_mean = p_df[stat_cat].mean()
        sharp_lambda = calculate_sharp_lambda(p_mean, pace_script, sos_mult, star_out, is_home, is_b2b)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"📊 {stat_cat.capitalize()} Trend & Projection")
            user_line = st.number_input("Sportsbook Line", value=float(round(p_mean, 1)), step=0.5)
            
            # Monte Carlo Logic
            sim_data, over_prob = run_monte_carlo(sharp_lambda, user_line)
            fig = go.Figure(data=[go.Histogram(x=sim_data, histnorm='probability', marker_color='#00ff96')])
            fig.add_vline(x=user_line, line_dash="dash", line_color="red")
            fig.update_layout(template="plotly_dark", title="Poisson Distribution (10,000 Games)")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("🎯 Market Edge")
            st.metric("Sharp Projection", round(sharp_lambda, 1))
            st.metric("Win Probability", f"{round(over_prob, 1)}%")
            
            # Kelly Criterion
            odds = st.number_input("Odds (American)", value=-110)
            dec_odds = (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
            win_p = over_prob / 100
            kelly = ((dec_odds - 1) * win_p - (1 - win_p)) / (dec_odds - 1) if dec_odds > 1 else 0
            st.metric("Kelly Suggestion", f"{max(0, round(kelly * 100, 1))}%")

elif mode == "Team Scout Radar":
    team_to_scout = st.selectbox("Select Team", sorted(list(team_map.keys())), index=sorted(list(team_map.keys())).index("NYK"))
    
    if st.button(f"🚀 Scan {team_to_scout} Roster"):
        t_id = team_map[team_to_scout]
        with st.status("Fetching Live 2026 Roster Stats...") as s:
            roster = commonteamroster.CommonTeamRoster(team_id=t_id).get_data_frames()[0].head(10)
            results = []
            for _, row in roster.iterrows():
                p_log, _, _ = get_player_data(row['PLAYER_ID'], is_id=True)
                if not p_log.empty:
                    m = p_log[stat_cat].mean()
                    p = calculate_sharp_lambda(m, pace_script, sos_mult, star_out, is_home, is_b2b)
                    results.append({"Player": row['PLAYER'], "Avg": round(m, 1), "Proj": round(p, 1), "Edge": round(p - m, 1)})
                time.sleep(0.1)
            s.update(label="Radar Scan Complete", state="complete")
        
        res_df = pd.DataFrame(results).sort_values("Edge", ascending=False)
        st.table(res_df)
        
        # Automatic Drill-down for the top edge
        if not res_df.empty:
            top_p = res_df.iloc[0]['Player']
            top_proj = res_df.iloc[0]['Proj']
            st.subheader(f"🎲 Simulation Drill-Down: {top_p}")
            sims, prob = run_monte_carlo(top_proj, res_df.iloc[0]['Avg'])
            st.write(f"Based on 10,000 simulations, **{top_p}** has a **{prob}%** chance to beat their season average in this matchup.")
