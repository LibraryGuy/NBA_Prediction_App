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

# --- 1. THE INJURY & REF DATA ---
@st.cache_data(ttl=600)
def get_intel():
    # MANDATORY: These players are filtered out NO MATTER WHAT
    intel = {
        "injuries": [
            "Nikola Jokic", "Joel Embiid", "Kevin Durant", "Ja Morant", 
            "Cameron Johnson", "Christian Braun", "Tamar Bates", "Trae Young",
            "Jayson Tatum", "Kyrie Irving", "Jimmy Butler"
        ],
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96},
            "Jacyn Goble": {"type": "Over", "impact": 1.06},
            "Marc Davis": {"type": "Over", "impact": 1.05}
        }
    }
    # Dynamic update from CBS
    try:
        url = "https://www.cbssports.com/nba/injuries/"
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        tables = pd.read_html(resp.text)
        for table in tables:
            if 'Player' in table.columns and 'Status' in table.columns:
                out = table[table['Status'].str.contains('Out|Sidelined', case=False, na=False)]
                intel["injuries"].extend(out['Player'].tolist())
    except: pass
    intel["injuries"] = list(set(intel["injuries"]))
    return intel

# --- 2. DASHBOARD INIT ---
st.set_page_config(page_title="Sharp Pro v9.5", layout="wide")
intel = get_intel()
p_map, avg_p = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0], 100 # Pace Placeholder

# --- 3. TEAM SCANNER (REWRITTEN FOR EXCLUSION) ---
if st.sidebar.radio("Navigation", ["Single Player", "Team Scanner"]) == "Team Scanner":
    st.header("🔍 Team Value Scanner")
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    stat_cat = st.sidebar.selectbox("Category", ["PTS", "REB", "AST", "PRA"])
    line = st.sidebar.number_input("Line", value=22.5)

    if st.button("📡 Scan Roster"):
        roster = commonteamroster.CommonTeamRoster(team_id=sel_team['id']).get_data_frames()[0]
        # Pre-identify team-specific injuries
        team_out = [p for p in roster['PLAYER'] if p in intel["injuries"]]
        scan_data = []

        for _, p in roster.iterrows():
            p_name = p['PLAYER']
            
            # --- CRITICAL FILTER: SHUT DOWN IF INJURED ---
            if p_name in intel["injuries"]:
                continue # Skip all calculations/API calls for this player
            
            try:
                log = playergamelog.PlayerGameLog(player_id=p['PLAYER_ID']).get_data_frames()[0]
                if not log.empty:
                    if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                    
                    # CASCADING LOGIC: Apply boost if stars are out
                    boost = 1.15 if len(team_out) > 0 else 1.0
                    proj = log[stat_cat].head(7).mean() * boost
                    prob = (1 - poisson.cdf(line - 0.5, proj)) * 100
                    
                    scan_data.append({
                        "Player": p_name, "Proj": round(proj, 1), 
                        "Usage Boost": f"{boost}x", "Win Prob": f"{round(prob, 1)}%",
                        "Signal": "🔥 SMART" if prob > 68 else "Neutral"
                    })
            except: continue

        st.info(f"**Ruled OUT for {sel_team['abbreviation']}:** {', '.join(team_out) if team_out else 'None'}")
        st.table(pd.DataFrame(scan_data).sort_values("Proj", ascending=False))
