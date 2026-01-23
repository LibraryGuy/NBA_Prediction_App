import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
import requests
import time
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, commonteamroster
from nba_api.stats.static import players, teams

# --- 1. CORE CONFIG & JOKIC SETTINGS ---
st.set_page_config(page_title="Sharp Pro v10.0", layout="wide")
JOKIC_ALIASES = ["Nikola Jokic", "Nikola Jokić", "Jokic, Nikola"]

# --- 2. ROBUST API WRAPPER ---
def safe_api_call(endpoint_class, **kwargs):
    """Bypasses cloud blocks with 2026-compliant browser headers."""
    headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    try:
        req = endpoint_class(**kwargs, headers=headers, timeout=20)
        df = req.get_data_frames()[0]
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# --- 3. THE ANALYZER ---
def get_player_data(p_id, p_name, is_denver=False, stat_cat="PTS", line=22.5):
    # MANDATORY JOKIC KILL-SWITCH
    if any(alias.lower() in p_name.lower() for alias in JOKIC_ALIASES):
        return "OUT"

    log = safe_api_call(playergamelog.PlayerGameLog, player_id=p_id, season='2025-26')
    if log.empty: return None

    if stat_cat == "PRA": 
        log['PRA'] = log['PTS'] + log['REB'] + log['AST']
    
    # Apply 1.15x Boost for Denver players because Jokic is out
    boost = 1.15 if is_denver else 1.0
    avg = log[stat_cat].head(10).mean()
    proj = avg * boost
    prob = (1 - poisson.cdf(line - 0.5, proj)) * 100
    
    return {"proj": proj, "prob": prob, "avg": avg, "log": log, "boost": boost}

# --- 4. UI ELEMENTS ---
with st.sidebar:
    st.title("🏀 Sharp Pro v10.0")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)
    st.divider()
    st.info("System Date: Jan 23, 2026")
    st.warning("Injury Lock: Nikola Jokic (OUT)")

# --- 5. RENDER LOGIC ---
if mode == "Single Player Analysis":
    search = st.text_input("Player Search", "Shai Gilgeous-Alexander")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Select Player", matches, format_func=lambda x: x['full_name'])
        is_den = "Nuggets" in str(safe_api_call(commonplayerinfo.CommonPlayerInfo, player_id=sel_p['id']))
        res = get_player_data(sel_p['id'], sel_p['full_name'], is_denver=is_den, stat_cat=stat_cat, line=line)
        
        if res == "OUT":
            st.error(f"🛑 {sel_p['full_name']} is currently OUT (Injury Protocol).")
        elif res:
            st.header(f"{sel_p['full_name']} Analysis")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sharp Projection", round(res['proj'], 1), f"{res['boost']}x Usage")
            c2.metric("Win Prob", f"{round(res['prob'], 1)}%")
            c3.metric("Last 10 Avg", round(res['avg'], 1))
            st.plotly_chart(px.line(res['log'].head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, title="Trend Line"))

elif mode == "Team Scanner":
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Run Team Scan"):
        with st.status("Fetching Data...", expanded=True) as status:
            roster_df = safe_api_call(commonteamroster.CommonTeamRoster, team_id=sel_team['id'])
            
            # BYPASS: If API Roster fails for Denver, use manual core rotation
            if roster_df.empty and "Denver" in sel_team['full_name']:
                st.write("API Throttled. Using Static Roster for Denver...")
                names = ["Jamal Murray", "Michael Porter Jr.", "Aaron Gordon", "Peyton Watson", "Russell Westbrook"]
                p_ids = [players.find_players_by_full_name(n)[0]['id'] for n in names]
                roster_df = pd.DataFrame({"PLAYER": names, "PLAYER_ID": p_ids})

            results = []
            if not roster_df.empty:
                for _, p in roster_df.iterrows():
                    data = get_player_data(p['PLAYER_ID'], p['PLAYER'], 
                                         is_denver=("Nuggets" in sel_team['full_name']),
                                         stat_cat=stat_cat, line=line)
                    if data and data != "OUT":
                        results.append({
                            "Player": p['PLAYER'], "Proj": round(data['proj'], 1),
                            "Prob": f"{round(data['prob'], 1)}%", "Usage": f"{data['boost']}x"
                        })
            status.update(label="Scan Complete!", state="complete", expanded=False)

        # SAFETY GUARD: Fixes the KeyError 'Proj' by checking if results list is empty
        if results:
            final_df = pd.DataFrame(results).sort_values("Proj", ascending=False)
            st.table(final_df)
        else:
            st.warning("No active players found. This is usually due to NBA API throttling. Please wait 60 seconds and try again.")
            
