import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from scipy.stats import poisson
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, commonteamroster

# --- 1. CORE DATA ENGINE ---
@st.cache_data(ttl=3600)
def load_nba_universe():
    # Guaranteed 30-team map for 2025-26 Season
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
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced', season='2025-26').get_data_frames()[0]
        avg_d = stats['DEF_RATING'].mean()
        sos = { [k for k,v in all_30.items() if v==row['TEAM_ID']][0]: row['DEF_RATING']/avg_d for _,row in stats.iterrows() }
        return sos, all_30
    except:
        return {k: 1.0 for k in all_30.keys()}, all_30

# --- 2. THE SIMULATOR ---
def run_monte_carlo(lambda_val, iterations=10000):
    return np.random.poisson(lambda_val, iterations)

def calculate_sharp_lambda(mean, pace, sos, star_out, home):
    # Sharp adjustment formula
    return mean * pace * (2 - sos) * (1.15 if star_out else 1.0) * (1.03 if home else 0.97)

# --- 3. UI ---
st.set_page_config(page_title="Sharp Pro 3.0", layout="wide")
sos_data, team_map = load_nba_universe()

with st.sidebar:
    st.header("🎯 System Controls")
    mode = st.radio("Mode", ["Single Player", "Team Scout Radar"])
    stat_cat = st.selectbox("Category", ["PTS", "REB", "AST", "PRA"])
    opp = st.selectbox("Opponent", sorted(list(team_map.keys())), index=sorted(list(team_map.keys())).index("SAC"))
    is_home = st.toggle("Home Game", value=True)
    pace_val = st.select_slider("Pace", options=[0.9, 1.0, 1.1], value=1.0)

if mode == "Team Scout Radar":
    team_to_scan = st.selectbox("Select Team", sorted(list(team_map.keys())), index=sorted(list(team_map.keys())).index("NYK"))
    
    if st.button(f"🚀 Scan {team_to_scan} Roster"):
        t_id = team_map[team_to_scan]
        with st.status("Analyzing 2025-26 Roster Trends...") as s:
            roster = commonteamroster.CommonTeamRoster(team_id=t_id).get_data_frames()[0]
            results = []
            for _, row in roster.head(10).iterrows():
                try:
                    log = playergamelog.PlayerGameLog(player_id=row['PLAYER_ID'], season='2025-26').get_data_frames()[0]
                    col = 'PTS' if stat_cat == 'PTS' else 'REB' if stat_cat == 'REB' else 'AST'
                    if stat_cat == 'PRA': log['PRA'] = log['PTS'] + log['REB'] + log['AST']; col = 'PRA'
                    
                    avg = log[col].mean()
                    proj = calculate_sharp_lambda(avg, pace_val, sos_data[opp], False, is_home)
                    results.append({"Player": row['PLAYER'], "Avg": round(avg,1), "Proj": round(proj,1), "Edge": round(proj-avg, 1)})
                except: pass
            s.update(label="Scan Complete!", state="complete")
        
        df = pd.DataFrame(results).sort_values("Edge", ascending=False)
        st.table(df)

        # --- RESTORED MONTE CARLO FOR TEAM VIEW ---
        if not df.empty:
            top_player = df.iloc[0]['Player']
            top_proj = df.iloc[0]['Proj']
            st.subheader(f"🎲 Simulation Drill-Down: {top_player}")
            sim_data = run_monte_carlo(top_proj)
            
            fig = go.Figure(data=[go.Histogram(x=sim_data, histnorm='probability', marker_color='#00ff96')])
            fig.update_layout(title=f"Probability Distribution for {top_player} ({stat_cat})", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # Hit Rates
            line = df.iloc[0]['Avg']
            hit_rate = (np.sum(sim_data > line) / 10000) * 100
            st.write(f"**Model Insight:** {top_player} has a **{hit_rate:.1f}%** chance of exceeding their season average of {line}.")
            
