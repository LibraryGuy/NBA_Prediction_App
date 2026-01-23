import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import requests
from nba_api.stats.endpoints import (playergamelog, scoreboardv2, 
                                     commonplayerinfo, leaguedashteamstats, 
                                     commonteamroster)
from nba_api.stats.static import players, teams

# --- 1. THE TRUTH LAYER (FIXED INJURIES) ---

@st.cache_data(ttl=600)
def get_intel():
    # 2026 Season Hard-Lock List
    intel = {
        "injuries": [
            "Nikola Jokic", "Nikola Jokić", "Jokic, Nikola", 
            "Cameron Johnson", "Christian Braun", "Tamar Bates",
            "Joel Embiid", "Kevin Durant", "Ja Morant", "Trae Young"
        ],
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96},
            "Jacyn Goble": {"type": "Over", "impact": 1.06},
            "Marc Davis": {"type": "Over", "impact": 1.05}
        }
    }
    # Dynamic Scraper
    try:
        url = "https://www.cbssports.com/nba/injuries/"
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        tables = pd.read_html(resp.text)
        for table in tables:
            if 'Player' in table.columns:
                out_list = table[table['Status'].str.contains('Out|Sidelined|Surgery', case=False, na=False)]['Player'].tolist()
                intel["injuries"].extend(out_list)
    except: pass
    return list(set(intel["injuries"]))

# --- 2. THE SCANNER (UNIFIED WITH FULL UI) ---

st.set_page_config(page_title="Sharp Pro v9.7", layout="wide")
injury_list = get_intel()

with st.sidebar:
    st.title("🏀 Sharp Pro v9.7")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Target Line", value=22.5, step=0.5)

# --- 3. LOGIC FOR BOTH MODES ---

def run_analysis(player_id, player_name, team_id):
    # THE ABSOLUTE KILL-SWITCH
    if any(alias in player_name for alias in injury_list):
        return None # Exit immediately if injured

    # Fetch stats
    log = playergamelog.PlayerGameLog(player_id=player_id, season='2025-26').get_data_frames()[0]
    if log.empty: return None
    
    if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
    
    # Calculate Injury Cascade (Usage Boost)
    # Check if a star on THEIR team is in our injury list
    roster = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
    team_out = [p for p in roster['PLAYER'] if p in injury_list]
    usage_boost = 1.15 if len(team_out) > 0 else 1.0
    
    # Final Proj
    raw_avg = log[stat_cat].head(10).mean()
    proj = raw_avg * usage_boost # Simplified for brevity, add pace/ref logic here
    prob = (1 - poisson.cdf(line - 0.5, proj)) * 100
    
    return {"proj": proj, "prob": prob, "log": log, "boost": usage_boost, "team_out": team_out}

# --- 4. RENDER MODES ---

if mode == "Single Player Analysis":
    search = st.text_input("Player Search", "Shai Gilgeous-Alexander")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    if matches:
        sel_p = st.selectbox("Confirm Player", matches, format_func=lambda x: x['full_name'])
        res = run_analysis(sel_p['id'], sel_p['full_name'], commonplayerinfo.CommonPlayerInfo(player_id=sel_p['id']).get_data_frames()[0]['TEAM_ID'].iloc[0])
        
        if res is None:
            st.error(f"🛑 {sel_p['full_name']} is currently OUT.")
        else:
            # RENDER DASHBOARD (All original visual features here)
            c1, c2, c3 = st.columns(3)
            c1.metric("Projection", round(res['proj'], 1), f"{res['boost']}x Usage")
            c2.metric("Win Prob", f"{round(res['prob'], 1)}%")
            c3.metric("Injury Impact", "High" if res['boost'] > 1 else "Normal")
            
            st.plotly_chart(px.bar(x=np.arange(0, 40), y=poisson.pmf(np.arange(0, 40), res['proj'])))

elif mode == "Team Scanner":
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    if st.button("Run Scan"):
        roster = commonteamroster.CommonTeamRoster(team_id=sel_team['id']).get_data_frames()[0]
        results = []
        for _, p in roster.iterrows():
            data = run_analysis(p['PLAYER_ID'], p['PLAYER'], sel_team['id'])
            if data:
                results.append({"Player": p['PLAYER'], "Proj": round(data['proj'], 1), "Prob": f"{round(data['prob'], 1)}%"})
        
        st.table(pd.DataFrame(results))
