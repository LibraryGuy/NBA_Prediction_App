import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import time
import random
from nba_api.stats.endpoints import (playergamelog, leaguegamefinder, 
                                     scoreboardv2, commonplayerinfo, 
                                     leaguedashteamstats, commonteamroster)
from nba_api.stats.static import players, teams
from nba_api.stats.library.http import NBAStatsHTTP

# --- 1. GLOBAL API CONFIGURATION ---
custom_headers = {
    'Host': 'stats.nba.com',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Referer': 'https://www.nba.com/',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}

NBAStatsHTTP.headers = custom_headers

# --- 2. THE BYPASS HELPER ---
def safe_api_call(endpoint_class, **kwargs):
    """Retries an API call, but fails quickly to avoid hanging the UI."""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Shortened timeout to 25s so the UI stays responsive
            return endpoint_class(**kwargs, timeout=25).get_data_frames()[0]
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                raise Exception("API Timeout")

# --- 3. DATA ENGINES (With Fallbacks) ---

@st.cache_data(ttl=3600)
def get_pace():
    """Fetches league pace but falls back to 100.0 if the server is stubborn."""
    try:
        # Attempt to get real-time pace
        df = safe_api_call(leaguedashteamstats.LeagueDashTeamStats, measure_type_detailed_defense='Advanced')
        pace_dict = {row['TEAM_ID']: row['PACE'] for _, row in df.iterrows()}
        avg_pace = df['PACE'].mean()
        return pace_dict, avg_pace, "Live"
    except:
        # FALLBACK: If the API hangs, return standard league pace so the app loads
        return {}, 100.0, "Static (Fallback)"

@st.cache_data(ttl=600)
def get_daily_schedule():
    """Fetches today's games or returns an empty schedule if blocked."""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        df = safe_api_call(scoreboardv2.ScoreboardV2, game_date=today)
        m_map = {}
        refs = ["Scott Foster", "Marc Davis", "Jacyn Goble", "Bill Kennedy"]
        for i, row in df.iterrows():
            ref = refs[i % len(refs)]
            m_map[row['HOME_TEAM_ID']] = {'opp_id': row['VISITOR_TEAM_ID'], 'ref': ref}
            m_map[row['VISITOR_TEAM_ID']] = {'opp_id': row['HOME_TEAM_ID'], 'ref': ref}
        return m_map, "Live"
    except:
        return {}, "Unavailable"

# --- 4. DASHBOARD SETUP ---
st.set_page_config(page_title="Sharp Pro v10.5", layout="wide")

# Load data with status indicators
pace_map, avg_pace, pace_status = get_pace()
schedule, schedule_status = get_daily_schedule()
intel = {"injuries": ["Nikola Jokic", "Kevin Durant", "Joel Embiid", "Ja Morant"],
         "ref_bias": {"Scott Foster": {"type": "Under", "impact": 0.96}, "Marc Davis": {"type": "Over", "impact": 1.05}}}

team_lookup = {t['id']: t['full_name'] for t in teams.get_teams()}

with st.sidebar:
    st.title("🏀 Sharp Pro v10.5")
    st.write(f"Pace Data: `{pace_status}`")
    st.write(f"Schedule: `{schedule_status}`")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)

# --- 5. MODE: SINGLE PLAYER ANALYSIS ---
if mode == "Single Player Analysis":
    search = st.text_input("Search Player", "Peyton Watson")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Confirm Player", matches, format_func=lambda x: x['full_name'])
        if st.button("🚀 Run Full Analysis"):
            with st.spinner("Analyzing Matchup..."):
                try:
                    # Player Info
                    p_info = safe_api_call(commonplayerinfo.CommonPlayerInfo, player_id=sel_p['id'])
                    t_id = p_info['TEAM_ID'].iloc[0]
                    game_context = schedule.get(t_id, {'opp_id': 0, 'ref': "Unknown"})
                    opp_name = team_lookup.get(game_context['opp_id'], "Opponent")
                    
                    # Logs
                    log = safe_api_call(playergamelog.PlayerGameLog, player_id=sel_p['id'], season='2025-26')
                    
                    if not log.empty:
                        if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                        
                        raw_avg = log[stat_cat].head(10).mean()
                        # Use fallback pace if live data failed
                        p_pace = pace_map.get(t_id, 100.0)
                        o_pace = pace_map.get(game_context['opp_id'], 100.0)
                        comp_pace = (p_pace + o_pace) / 2
                        
                        final_proj = raw_avg * (comp_pace / avg_pace)
                        prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

                        st.header(f"{sel_p['full_name']} vs {opp_name}")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Final Projection", round(final_proj, 1))
                        c2.metric("L10 Average", round(raw_avg, 1))
                        c3.metric("Win Prob (Over)", f"{round(prob_over, 1)}%")

                        st.divider()
                        st.subheader("Last 10 Game Trend")
                        fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                        fig_t.add_hline(y=line, line_color="red", line_dash="dash")
                        st.plotly_chart(fig_t, use_container_width=True)
                except Exception as e:
                    st.error("The NBA servers are currently blocking requests. Please try again in a few minutes.")

# --- 6. MODE: TEAM SCANNER ---
elif mode == "Team Scanner":
    st.header("🔍 Value Scanner")
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Roster"):
        try:
            roster = safe_api_call(commonteamroster.CommonTeamRoster, team_id=sel_team['id'])
            scan_data = []
            with st.status("Scanning..."):
                for i, p in roster.head(8).iterrows(): # Scan top 8 to avoid rate limits
                    time.sleep(1.5)
                    try:
                        p_log = safe_api_call(playergamelog.PlayerGameLog, player_id=p['PLAYER_ID'])
                        if not p_log.empty:
                            if stat_cat == "PRA": p_log['PRA'] = p_log['PTS'] + p_log['REB'] + p_log['AST']
                            raw = p_log[stat_cat].head(5).mean()
                            scan_data.append({"Player": p['PLAYER'], "L5 Avg": round(raw, 1)})
                    except: continue
            st.table(pd.DataFrame(scan_data))
        except:
            st.error("Scanner Timed Out.")
