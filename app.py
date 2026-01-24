import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import pytz
import time
import json
import requests
from nba_api.stats.endpoints import (playergamelog, leaguegamefinder, 
                                     scoreboardv2, commonplayerinfo, 
                                     leaguedashteamstats, commonteamroster)
from nba_api.stats.static import players, teams

# --- REINFORCED HEADERS & TIMEOUTS ---
HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
}

# --- 1. THE "PLAN B" DATA ENGINE ---
def safe_api_call(endpoint_class, max_retries=3, **kwargs):
    """Retries with a delay and handles blocks gracefully."""
    for attempt in range(max_retries):
        try:
            # Random slight delay to mimic human behavior
            time.sleep(np.random.uniform(0.5, 1.5)) 
            call = endpoint_class(**kwargs, headers=HEADERS, timeout=60)
            return call.get_data_frames()
        except Exception:
            if attempt < max_retries - 1:
                continue
    return None

@st.cache_data(ttl=3600)
def get_backup_schedule():
    """Static CDN fallback for when ScoreboardV2 is blocked."""
    try:
        url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
        res = requests.get(url, timeout=10).json()
        today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
        m_map = {}
        for day in res['leagueSchedule']['gameDates']:
            if day['gameDate'].split('T')[0] == today:
                for g in day['games']:
                    m_map[g['homeTeam']['teamId']] = {'opp_id': g['visitorTeam']['teamId'], 'ref': "TBD (Blocked)"}
                    m_map[g['visitorTeam']['teamId']] = {'opp_id': g['homeTeam']['teamId'], 'ref': "TBD (Blocked)"}
        return m_map
    except: return {}

# --- 2. THE CORE LOGIC ---
st.set_page_config(page_title="Sharp Pro v10.8", layout="wide")

# Pre-load pace (crucial for projections)
pace_frames = safe_api_call(leaguedashteamstats.LeagueDashTeamStats, measure_type_detailed_defense='Advanced')
pace_map = {row['TEAM_ID']: row['PACE'] for _, row in pace_frames[0].iterrows()} if pace_frames else {}
avg_pace = np.mean(list(pace_map.values())) if pace_map else 100.0

# Try Scoreboard for Refs, fallback to CDN if blocked
schedule = {}
sb_frames = safe_api_call(scoreboardv2.ScoreboardV2, game_date=datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d'))
if sb_frames:
    for _, row in sb_frames[0].iterrows():
        ref = "Unknown"
        if len(sb_frames) > 2:
            game_refs = sb_frames[2][sb_frames[2]['GAME_ID'] == row['GAME_ID']]
            if not game_refs.empty: ref = game_refs.iloc[0]['OFFICIAL_NAME']
        schedule[row['HOME_TEAM_ID']] = {'opp_id': row['VISITOR_TEAM_ID'], 'ref': ref}
        schedule[row['VISITOR_TEAM_ID']] = {'opp_id': row['HOME_TEAM_ID'], 'ref': ref}
else:
    st.sidebar.warning("⚠️ Official Ref Data Blocked. Using Backup Schedule.")
    schedule = get_backup_schedule()

# --- 3. UI RENDER ---
with st.sidebar:
    st.title("🏀 Sharp Pro v10.8")
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Line", value=22.5)

search = st.text_input("Enter Player Name", "Peyton Watson")
player_matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]

if player_matches:
    sel_p = st.selectbox("Confirm", player_matches, format_func=lambda x: x['full_name'])
    if st.button("🚀 Analyze"):
        with st.status("Connecting to NBA Servers...") as status:
            # A. Info
            p_frames = safe_api_call(commonplayerinfo.CommonPlayerInfo, player_id=sel_p['id'])
            if not p_frames:
                st.error("NBA.com completely blocked this request. Try again in 15 mins.")
                st.stop()
            
            t_id = p_frames[0]['TEAM_ID'].iloc[0]
            game = schedule.get(t_id, {'opp_id': 0, 'ref': "Unknown"})
            
            # B. Logs
            log_frames = safe_api_call(playergamelog.PlayerGameLog, player_id=sel_p['id'], season='2025-26')
            h2h_frames = safe_api_call(leaguegamefinder.LeagueGameFinder, player_id_nullable=sel_p['id'], vs_team_id_nullable=game['opp_id'])
            
            if log_frames:
                log = log_frames[0]
                if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                
                # Projection
                raw_avg = log[stat_cat].head(10).mean()
                proj = raw_avg * ((pace_map.get(t_id, 100) + pace_map.get(game['opp_id'], 100)) / (2 * avg_pace))
                prob = (1 - poisson.cdf(line - 0.5, proj)) * 100

                status.update(label="Analysis Complete!", state="complete")

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Projection", round(proj, 1))
                m2.metric("Referee", game['ref'])
                m3.metric("Win Prob", f"{round(prob, 1)}%")
                
                # H2H Table
                st.subheader("Last 5 Games vs Opponent")
                if h2h_frames:
                    h2h = h2h_frames[0]
                    if stat_cat == "PRA": h2h['PRA'] = h2h['PTS'] + h2h['REB'] + h2h['AST']
                    st.table(h2h[['GAME_DATE', 'MATCHUP', stat_cat]].head(5))
                else:
                    st.info("H2H lookup timed out. Projections still valid.")
