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

# --- 1. ROBUST API HANDLER (FIXES TIMEOUTS) ---
def safe_nba_request(endpoint_class, **kwargs):
    """Mimics a browser and handles retries to prevent ReadTimeout."""
    headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Origin': 'https://www.nba.com',
        'Referer': 'https://www.nba.com/',
        'Connection': 'keep-alive',
    }
    for attempt in range(3):
        try:
            # We set a high timeout (60s) because stats.nba.com is slow in 2026
            request = endpoint_class(**kwargs, headers=headers, timeout=60)
            return request.get_data_frames()[0]
        except (ReadTimeout, ConnectionError):
            time.sleep(2) # Wait before retry
    return pd.DataFrame()

# --- 2. THE INTELLIGENCE ENGINE ---
@st.cache_data(ttl=600)
def get_intel():
    # Hard-coded kill-switch for Jan 23, 2026
    intel = {
        "injuries": [
            "Nikola Jokic", "Nikola Jokić", "Jokic, Nikola", 
            "Joel Embiid", "Kevin Durant", "Ja Morant", "Trae Young",
            "Christian Braun", "Tamar Bates", "Jayson Tatum"
        ],
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96},
            "Jacyn Goble": {"type": "Over", "impact": 1.06},
            "Marc Davis": {"type": "Over", "impact": 1.05}
        }
    }
    # Dynamic Scraper Fallback
    try:
        url = "https://www.cbssports.com/nba/injuries/"
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        tables = pd.read_html(resp.text)
        for table in tables:
            if 'Player' in table.columns:
                out = table[table['Status'].str.contains('Out|Sidelined', case=False, na=False)]['Player'].tolist()
                intel["injuries"].extend(out)
    except: pass
    return list(set(intel["injuries"]))

# --- 3. APP UI SETUP ---
st.set_page_config(page_title="Sharp Pro v9.7", layout="wide")
injury_list = get_intel()

with st.sidebar:
    st.title("🏀 Sharp Pro v9.7")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)
    st.divider()
    st.info(f"System Date: Jan 23, 2026")

# --- 4. CORE ANALYTICS LOGIC ---
def analyze_player(p_id, p_name, team_id):
    # 1. THE JOKIC FILTER (Kill process if name matches injury list)
    if any(alias.lower() in p_name.lower() for alias in injury_list):
        return "OUT"

    # 2. DATA FETCHING (Using the safe wrapper)
    log = safe_nba_request(playergamelog.PlayerGameLog, player_id=p_id, season='2025-26')
    if log.empty: return None

    # 3. STATS PROCESSING
    if stat_cat == "PRA": 
        log['PRA'] = log['PTS'] + log['REB'] + log['AST']
    
    # 4. INJURY CASCADING (Redistribute usage if stars are out)
    roster = safe_nba_request(commonteamroster.CommonTeamRoster, team_id=team_id)
    team_out = [name for name in roster['PLAYER'].tolist() if name in injury_list] if not roster.empty else []
    
    usage_boost = 1.15 if len(team_out) > 0 else 1.0
    raw_avg = log[stat_cat].head(10).mean()
    proj = raw_avg * usage_boost
    prob = (1 - poisson.cdf(line - 0.5, proj)) * 100

    return {
        "proj": proj, "prob": prob, "avg": raw_avg, 
        "boost": usage_boost, "team_out": team_out, "log": log
    }

# --- 5. RENDER MODES ---
if mode == "Single Player Analysis":
    search = st.text_input("Search Player", "Shai Gilgeous-Alexander")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Select Result", matches, format_func=lambda x: x['full_name'])
        p_info = safe_nba_request(commonplayerinfo.CommonPlayerInfo, player_id=sel_p['id'])
        
        if not p_info.empty:
            res = analyze_player(sel_p['id'], sel_p['full_name'], p_info['TEAM_ID'].iloc[0])
            
            if res == "OUT":
                st.error(f"🛑 {sel_p['full_name']} is confirmed OUT for today.")
            elif res:
                # ALL DASHBOARD FEATURES RETAINED
                st.header(f"{sel_p['full_name']} Analysis")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Sharp Projection", round(res['proj'], 1), f"{res['boost']}x Usage")
                c2.metric("Win Probability", f"{round(res['prob'], 1)}%")
                c3.metric("Last 10 Avg", round(res['avg'], 1))
                c4.metric("Team Injuries", len(res['team_out']))

                v1, v2 = st.columns(2)
                with v1:
                    st.subheader("Poisson Probability Distribution")
                    x = np.arange(max(0, int(res['proj']-15)), int(res['proj']+15))
                    fig = px.bar(x=x, y=poisson.pmf(x, res['proj']), labels={'x': stat_cat, 'y': 'Prob'})
                    fig.add_vline(x=line, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                with v2:
                    st.subheader("Performance Trend (L10)")
                    fig_t = px.line(res['log'].head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                    fig_t.add_hline(y=line, line_color="red", line_dash="dot")
                    st.plotly_chart(fig_t, use_container_width=True)

elif mode == "Team Scanner":
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    if st.button("📡 Scan Roster"):
        roster = safe_nba_request(commonteamroster.CommonTeamRoster, team_id=sel_team['id'])
        if not roster.empty:
            scan_data = []
            with st.status("Calculating boosted projections..."):
                for _, p in roster.iterrows():
                    data = analyze_player(p['PLAYER_ID'], p['PLAYER'], sel_team['id'])
                    if data and data != "OUT":
                        scan_data.append({
                            "Player": p['PLAYER'], "Proj": round(data['proj'], 1), 
                            "Usage": f"{data['boost']}x", "Prob": f"{round(data['prob'], 1)}%",
                            "Signal": "🔥 VALUE" if data['prob'] > 68 else "Neutral"
                        })
            st.table(pd.DataFrame(scan_data).sort_values("Proj", ascending=False))
