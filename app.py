import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
import requests
import time
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, commonteamroster
from nba_api.stats.static import players, teams

# --- 1. THE RELIABILITY LAYER ---
def safe_api_call(endpoint_class, **kwargs):
    """Bypasses cloud-blocks with updated 2026 headers and retry logic."""
    headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    for _ in range(2):
        try:
            req = endpoint_class(**kwargs, headers=headers, timeout=15)
            df = req.get_data_frames()[0]
            if not df.empty: return df
        except: time.sleep(1)
    return pd.DataFrame()

# --- 2. GLOBAL SETTINGS ---
st.set_page_config(page_title="Sharp Pro v9.9", layout="wide")
JOKIC_ALIASES = ["Nikola Jokic", "Nikola Jokić", "Jokic, Nikola"]

with st.sidebar:
    st.title("🏀 Sharp Pro v9.9")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)
    st.warning("⚠️ JAN 23, 2026: Jokic is OUT")

# --- 3. THE ANALYZER ---
def get_player_data(p_id, p_name, is_denver=False):
    # IMMEDIATE JOKIC KILL-SWITCH
    if any(alias.lower() in p_name.lower() for alias in JOKIC_ALIASES):
        return "OUT"

    log = safe_api_call(playergamelog.PlayerGameLog, player_id=p_id, season='2025-26')
    if log.empty: return None

    if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
    
    # Apply 1.15x Boost for Denver players because Jokic is out
    boost = 1.15 if is_denver else 1.0
    avg = log[stat_cat].head(10).mean()
    proj = avg * boost
    prob = (1 - poisson.cdf(line - 0.5, proj)) * 100
    
    return {"proj": proj, "prob": prob, "avg": avg, "log": log, "boost": boost}

# --- 4. RENDER MODES ---
if mode == "Single Player Analysis":
    search = st.text_input("Search Player", "Shai Gilgeous-Alexander")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Select", matches, format_func=lambda x: x['full_name'])
        is_nugget = sel_p['full_name'] in ["Jamal Murray", "Michael Porter Jr.", "Aaron Gordon", "Peyton Watson"]
        res = get_player_data(sel_p['id'], sel_p['full_name'], is_denver=is_nugget)
        
        if res == "OUT":
            st.error(f"🛑 {sel_p['full_name']} is currently OUT.")
        elif res:
            st.header(f"{sel_p['full_name']} Dashboard")
            c1, c2 = st.columns(2)
            c1.metric("Projection", round(res['proj'], 1), f"{res['boost']}x Usage")
            c2.metric("Win Probability", f"{round(res['prob'], 1)}%")
            st.plotly_chart(px.line(res['log'].head(10).iloc[::-1], x='GAME_DATE', y=stat_cat))

elif mode == "Team Scanner":
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Run Analysis"):
        # FALLBACK: If roster API fails, we use a static list for Denver to prevent blank screens
        if sel_team['full_name'] == "Denver Nuggets":
            st.info("Using Denver Static Roster (API Bypass Active)")
            # 2026 Core Rotation
            roster_names = ["Jamal Murray", "Michael Porter Jr.", "Aaron Gordon", "Peyton Watson", "Russell Westbrook", "Dario Saric"]
            roster_ids = [players.find_players_by_full_name(n)[0]['id'] for n in roster_names]
            roster_df = pd.DataFrame({"PLAYER": roster_names, "PLAYER_ID": roster_ids})
        else:
            roster_df = safe_api_call(commonteamroster.CommonTeamRoster, team_id=sel_team['id'])

        if not roster_df.empty:
            results = []
            for _, p in roster_df.iterrows():
                data = get_player_data(p['PLAYER_ID'], p['PLAYER'], is_denver=(sel_team['abbreviation'] == 'DEN'))
                if data and data != "OUT":
                    results.append({
                        "Player": p['PLAYER'], "Proj": round(data['proj'], 1),
                        "Prob": f"{round(data['prob'], 1)}%", "Signal": "🔥" if data['prob'] > 65 else ""
                    })
            st.table(pd.DataFrame(results).sort_values("Proj", ascending=False))
        else:
            st.error("NBA Server Blocked Request. Please try Single Player mode or refresh.")
