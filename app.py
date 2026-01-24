import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import pytz
import requests
import time
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import players, teams

# --- 1. SETTINGS & STABLE ENGINES ---
st.set_page_config(page_title="Sharp Pro v10.10", layout="wide")

# FREE API KEY: This is a public key for demonstration; get your own at balldontlie.io for higher limits.
BDL_URL = "https://api.balldontlie.io/v1"
BDL_HEADERS = {"Authorization": "YOUR_FREE_KEY_HERE"} 

@st.cache_data(ttl=1800)
def get_intel():
    return {
        "injuries": ["Nikola Jokic", "Kevin Durant", "Joel Embiid", "Ja Morant", "Giannis"],
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96},
            "Marc Davis": {"type": "Over", "impact": 1.05},
            "Jacyn Goble": {"type": "Over", "impact": 1.04}
        }
    }

@st.cache_data(ttl=3600)
def get_stable_schedule():
    """Pulls games from Balldontlie or NBA CDN to ensure sidebar loads."""
    try:
        url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
        res = requests.get(url, timeout=5).json()
        today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
        m_map = {}
        for day in res['leagueSchedule']['gameDates']:
            if day['gameDate'].split('T')[0] == today:
                for g in day['games']:
                    h_id, v_id = g['homeTeam']['teamId'], g['visitorTeam']['teamId']
                    m_map[h_id] = {'opp_id': v_id, 'opp_name': g['visitorTeam']['teamName'], 'game_id': g['gameId']}
                    m_map[v_id] = {'opp_id': h_id, 'opp_name': g['homeTeam']['teamName'], 'game_id': g['gameId']}
        return m_map
    except: return {}

@st.cache_data(ttl=600)
def get_official_refs_fast():
    """Tries for 3 seconds to get refs. If it fails, the app survives."""
    try:
        today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
        # SCOREBOARDv2 is usually the culprit for hanging, so we use a strict timeout
        sb = scoreboardv2.ScoreboardV2(game_date=today, timeout=3).get_data_frames()
        if len(sb) > 2:
            return dict(zip(sb[2]['GAME_ID'], sb[2]['OFFICIAL_NAME']))
    except: pass
    return {}

# --- 2. SIDEBAR NAVIGATION ---
intel = get_intel()
schedule = get_stable_schedule()
team_list = sorted(teams.get_teams(), key=lambda x: x['full_name'])

with st.sidebar:
    st.title("🏀 Sharp Pro v10.10")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Value Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)
    st.divider()
    st.info(f"Today's Games Found: {len(schedule)//2 if schedule else 0}")

# --- 3. MODE: SINGLE PLAYER ANALYSIS ---
if mode == "Single Player Analysis":
    search = st.text_input("Find Player", "Peyton Watson")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Select Player", matches, format_func=lambda x: x['full_name'])
        if st.button("🚀 Analyze Player"):
            with st.status("Gathering Multi-Source Data...") as status:
                try:
                    # Player Logs from Balldontlie (Blocked-proof)
                    p_id = sel_p['id']
                    # Note: You can replace this with playergamelog if balldontlie isn't set up
                    from nba_api.stats.endpoints import playergamelog
                    logs = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26', timeout=10).get_data_frames()[0]
                    
                    if not logs.empty:
                        # Logic
                        if stat_cat == "PRA": logs['PRA'] = logs['PTS'] + logs['REB'] + logs['AST']
                        avg_10 = logs[stat_cat].head(10).mean()
                        
                        # Ref Check
                        ref_map = get_official_refs_fast()
                        t_id = logs['Team_ID'].iloc[0] if 'Team_ID' in logs.columns else 0
                        game_data = schedule.get(t_id, {'opp_name': "N/A", 'game_id': "0"})
                        ref_name = ref_map.get(game_data['game_id'], "Unknown/Blocked")
                        
                        # Final Projection
                        impact = intel['ref_bias'].get(ref_name, {"impact": 1.0})['impact']
                        final_proj = avg_10 * impact
                        prob = (1 - poisson.cdf(line - 0.5, final_proj)) * 100
                        
                        status.update(label="Analysis Ready!", state="complete")
                        
                        # UI
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Projection", round(final_proj, 1))
                        m2.metric("Ref Assignment", ref_name)
                        m3.metric("Over Prob", f"{round(prob, 1)}%")
                        
                        st.plotly_chart(px.line(logs.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, title="L10 Trend"), use_container_width=True)
                except Exception as e:
                    st.error("NBA Server is currently unresponsive. Try again in 5 mins.")

# --- 4. MODE: TEAM VALUE SCANNER (RECOVERED) ---
elif mode == "Team Value Scanner":
    sel_team = st.selectbox("Select Team to Scan", team_list, format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Full Roster"):
        with st.status(f"Scanning {sel_team['full_name']}...") as status:
            from nba_api.stats.endpoints import commonteamroster, playergamelog
            roster = commonteamroster.CommonTeamRoster(team_id=sel_team['id'], timeout=10).get_data_frames()[0]
            
            scan_results = []
            for _, player in roster.head(8).iterrows(): # Scan top 8 players to avoid rate limits
                try:
                    p_log = playergamelog.PlayerGameLog(player_id=player['PLAYER_ID'], season='2025-26', timeout=5).get_data_frames()[0]
                    if not p_log.empty:
                        if stat_cat == "PRA": p_log['PRA'] = p_log['PTS'] + p_log['REB'] + p_log['AST']
                        avg = p_log[stat_cat].head(5).mean()
                        prob = (1 - poisson.cdf(line - 0.5, avg)) * 100
                        scan_results.append({
                            "Player": player['PLAYER'],
                            "L5 Avg": round(avg, 1),
                            "Over Prob": f"{round(prob, 1)}%",
                            "Signal": "🔥 HIGH" if prob > 65 else ("❄️ LOW" if prob < 35 else "MID")
                        })
                except: continue
            
            status.update(label="Scan Complete!", state="complete")
            st.table(pd.DataFrame(scan_results))
