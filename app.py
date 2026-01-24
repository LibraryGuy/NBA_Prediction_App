import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import pytz
import requests
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import players

# --- 1. CONFIG & HEADERS ---
st.set_page_config(page_title="Sharp Pro v10.9", layout="wide")

# This free API key is for Balldontlie - a more stable cloud source
# You can get your own for free at balldontlie.io if this one hits limits
BDL_API_URL = "https://api.balldontlie.io/v1"
BDL_HEADERS = {"Authorization": "YOUR_FREE_KEY_HERE"} # Optional: App works without it for small traffic

# --- 2. DATA ENGINES ---

@st.cache_data(ttl=3600)
def get_stable_schedule():
    """Uses a more stable source for the schedule to prevent perpetual loading."""
    try:
        today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
        # Using a public CDN for game schedule
        url = f"https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
        res = requests.get(url, timeout=5).json()
        
        m_map = {}
        for day in res['leagueSchedule']['gameDates']:
            if day['gameDate'].split('T')[0] == today:
                for g in day['games']:
                    home_id = g['homeTeam']['teamId']
                    away_id = g['visitorTeam']['teamId']
                    m_map[home_id] = {'opp_id': away_id, 'opp_name': g['visitorTeam']['teamName']}
                    m_map[away_id] = {'opp_id': home_id, 'opp_name': g['homeTeam']['teamName']}
        return m_map
    except: return {}

@st.cache_data(ttl=1800)
def get_official_refs():
    """Only uses the heavy NBA_API for Ref data, with a fast-fail timeout."""
    try:
        today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
        # We give the NBA server only 5 seconds to respond. If it hangs, we skip it.
        sb = scoreboardv2.ScoreboardV2(game_date=today, timeout=5).get_data_frames()
        if len(sb) > 2:
            officials = sb[2]
            return dict(zip(officials['GAME_ID'], officials['OFFICIAL_NAME']))
    except: return {}
    return {}

# --- 3. THE ANALYTICS ENGINE ---

st.sidebar.title("🏀 Sharp Pro v10.9")
stat_cat = st.sidebar.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
line = st.sidebar.number_input("Sportsbook Line", value=22.5)

search = st.text_input("Search Player", "Peyton Watson")
matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]

if matches:
    sel_p = st.selectbox("Confirm Player", matches, format_func=lambda x: x['full_name'])
    
    if st.button("🚀 Run Analysis"):
        # Layer 1: Schedule (Stable)
        schedule = get_stable_schedule()
        # Layer 2: Refs (Attempts to connect, skips if blocked)
        refs = get_official_refs()
        
        with st.status("Performing Data Extraction...") as status:
            try:
                # Fetching 2025-26 Stats
                from nba_api.stats.endpoints import playergamelog
                logs = playergamelog.PlayerGameLog(player_id=sel_p['id'], season='2025-26', timeout=10).get_data_frames()[0]
                
                if logs.empty:
                    st.error("Could not retrieve game logs. NBA servers may be throttling.")
                    st.stop()

                # Calculate Stats
                if stat_cat == "PRA": logs['PRA'] = logs['PTS'] + logs['REB'] + logs['AST']
                avg_10 = logs[stat_cat].head(10).mean()
                
                # Contextual Adjustment
                ref_name = "Ref Data Blocked" if not refs else list(refs.values())[0]
                # (You can expand the ref_bias logic here)
                proj = avg_10 * 1.02 # Base projection with slight boost for pace
                prob = (1 - poisson.cdf(line - 0.5, proj)) * 100

                status.update(label="Analysis Complete!", state="complete")

                # --- DASHBOARD RENDER ---
                c1, c2, c3 = st.columns(3)
                c1.metric("Projection", round(proj, 1))
                c2.metric("Official Ref", ref_name)
                c3.metric("Win Prob (Over)", f"{round(prob, 1)}%")

                st.divider()
                st.subheader("Last 10 Games Performance")
                fig = px.line(logs.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                fig.add_hline(y=line, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error("Cloud Connection Error. NBA.com is blocking this session.")
