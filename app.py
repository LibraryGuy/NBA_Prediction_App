import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
import time
from nba_api.stats.endpoints import (playergamelog, commonplayerinfo, 
                                     leaguedashteamstats, commonteamroster)
from nba_api.stats.static import players, teams
from nba_api.stats.library.http import NBAStatsHTTP

# --- 1. THE "IDENTITY" MASK ---
NBAStatsHTTP.headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.nba.com/',
}

# --- 2. THE BULLETPROOF CALLER ---
def fetch_with_fallback(endpoint_class, fallback_val, **kwargs):
    """Tries the API, but returns a fallback immediately on failure."""
    try:
        # We give the NBA 15 seconds. If they don't answer, we move on.
        return endpoint_class(**kwargs, timeout=15).get_data_frames()[0]
    except Exception:
        st.sidebar.warning(f"⚠️ API blocked for {endpoint_class.__name__}. Using fallback data.")
        return fallback_val

# --- 3. DATA ENGINES ---

@st.cache_data(ttl=3600)
def get_pace_data():
    # If API fails, we return a standard dict with 100 pace for all teams
    default_pace = {t['id']: 100.0 for t in teams.get_teams()}
    df = fetch_with_fallback(leaguedashteamstats.LeagueDashTeamStats, None, measure_type_detailed_defense='Advanced')
    
    if df is not None:
        return {row['TEAM_ID']: row['PACE'] for _, row in df.iterrows()}, df['PACE'].mean()
    return default_pace, 100.0

# --- 4. DASHBOARD UI ---
st.set_page_config(page_title="Sharp Pro v10.5", layout="wide")

with st.sidebar:
    st.title("🏀 Sharp Pro")
    # THE CONSISTENCY SWITCH
    app_mode = st.toggle("Offline/Simulated Mode", help="Use this if the NBA servers are blocking you.")
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Line", value=22.5)

pace_map, avg_pace = get_pace_data()

# --- 5. LOGIC WITH SIMULATION FALLBACK ---
search = st.text_input("Search Player", "Peyton Watson")
player_matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]

if player_matches:
    sel_p = st.selectbox("Confirm", player_matches, format_func=lambda x: x['full_name'])
    
    if st.button("🚀 Run Analysis"):
        if app_mode:
            # SIMULATED DATA (App always works)
            st.info("Using Simulated Data Mode")
            final_proj = 24.2
            prob_over = 58.4
            raw_avg = 21.0
        else:
            # LIVE DATA
            with st.spinner("Attempting to bypass NBA firewall..."):
                log = fetch_with_fallback(playergamelog.PlayerGameLog, pd.DataFrame(), player_id=sel_p['id'], season='2025-26')
                
                if not log.empty:
                    if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                    raw_avg = log[stat_cat].head(10).mean()
                    final_proj = raw_avg * (pace_map.get(0, 100) / avg_pace)
                    prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100
                else:
                    st.error("NBA Servers Blocked this Player. Switch to 'Offline Mode' to test UI.")
                    st.stop()

        # UI (This part is consistent regardless of where data comes from)
        c1, c2, c3 = st.columns(3)
        c1.metric("Projection", round(final_proj, 1))
        c2.metric("L10 Avg", round(raw_avg, 1))
        c3.metric("Win Prob", f"{round(prob_over, 1)}%")
