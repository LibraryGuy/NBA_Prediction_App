import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import requests
import time
from requests.exceptions import ReadTimeout, ConnectionError
from nba_api.stats.endpoints import (playergamelog, scoreboardv2, 
                                     commonplayerinfo, leaguedashteamstats, 
                                     commonteamroster)
from nba_api.stats.static import players, teams

# --- 1. THE RECOVERY ENGINE (Fixes the blank screen) ---
def safe_nba_request(endpoint_class, **kwargs):
    headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
    }
    for attempt in range(2): # Quick retry
        try:
            request = endpoint_class(**kwargs, headers=headers, timeout=30)
            df = request.get_data_frames()[0]
            if not df.empty: return df
        except:
            time.sleep(1)
    return pd.DataFrame()

# --- 2. THE INTELLIGENCE ENGINE (Hard-Coded Jokic Rule) ---
@st.cache_data(ttl=600)
def get_intel():
    # Jan 23, 2026: Jokic is OUT.
    return {
        "injuries": ["Nikola Jokic", "Nikola Jokić", "Jokic, Nikola", "Joel Embiid", "Ja Morant"],
        "pace": 100.2 # 2026 League Average Pace
    }

# --- 3. APP UI ---
st.set_page_config(page_title="Sharp Pro v9.8", layout="wide")
intel = get_intel()

with st.sidebar:
    st.title("🏀 Sharp Pro v9.8")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)

# --- 4. CORE ANALYTICS ---
def analyze_player(p_id, p_name, team_id):
    # MANDATORY: The Jokic Filter
    if any(alias.lower() in p_name.lower() for alias in intel["injuries"]):
        return "OUT"

    # DATA FETCH
    log = safe_nba_request(playergamelog.PlayerGameLog, player_id=p_id, season='2025-26')
    
    # If the API is failing us, provide a generic baseline so the screen isn't blank
    if log.empty:
        return "API_LIMIT" # New flag for the UI to handle

    if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
    
    # Usage redistribution logic
    usage_boost = 1.15 if "Denver" in str(team_id) or "Nuggets" in str(team_id) else 1.0
    raw_avg = log[stat_cat].head(10).mean()
    proj = raw_avg * usage_boost
    prob = (1 - poisson.cdf(line - 0.5, proj)) * 100

    return {"proj": proj, "prob": prob, "avg": raw_avg, "boost": usage_boost, "log": log}

# --- 5. THE RENDERERS ---
if mode == "Single Player Analysis":
    search = st.text_input("Player Search", "Shai Gilgeous-Alexander")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Select", matches, format_func=lambda x: x['full_name'])
        res = analyze_player(sel_p['id'], sel_p['full_name'], "Any")
        
        if res == "OUT":
            st.error(f"🛑 {sel_p['full_name']} is OUT.")
        elif res == "API_LIMIT":
            st.warning("The NBA servers are currently throttling this request. Try again in 30 seconds.")
        elif res:
            st.metric("Final Projection", round(res['proj'], 1), f"{res['boost']}x Usage")
            st.plotly_chart(px.line(res['log'].head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, title="Trend"))

elif mode == "Team Scanner":
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    if st.button("📡 Scan Roster"):
        roster = safe_nba_request(commonteamroster.CommonTeamRoster, team_id=sel_team['id'])
        
        if not roster.empty:
            scan_results = []
            for _, p in roster.iterrows():
                data = analyze_player(p['PLAYER_ID'], p['PLAYER'], sel_team['id'])
                if data == "OUT": continue
                
                if data == "API_LIMIT":
                    scan_results.append({"Player": p['PLAYER'], "Status": "Throttled by NBA", "Proj": "N/A"})
                elif data:
                    scan_results.append({
                        "Player": p['PLAYER'], "Proj": round(data['proj'], 1), 
                        "Prob": f"{round(data['prob'], 1)}%", "Signal": "🔥" if data['prob'] > 65 else ""
                    })
            
            if scan_results:
                st.table(pd.DataFrame(scan_results))
            else:
                st.info("All players for this team are currently filtered out (Injuries/Inactives).")
        else:
            st.error("Could not load roster. The NBA servers are blocking this cloud request. Try refreshing the page.")
