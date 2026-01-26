import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
import time
import random
import uuid
import unicodedata
from streamlit_gsheets import GSheetsConnection
from nba_api.stats.endpoints import (playergamelog, commonplayerinfo, 
                                     leaguedashteamstats, commonteamroster,
                                     leaguegamefinder)
from nba_api.stats.static import players, teams

# --- 1. UTILITIES & STEALTH ---
def normalize_string(text):
    """Removes accents and converts to lowercase for matching (e.g., Dončić -> doncic)."""
    return "".join(
        c for c in unicodedata.normalize('NFD', text.lower())
        if unicodedata.category(c) != 'Mn'
    )

def get_stealth_headers():
    """Generates unique IDs and random browsers to avoid bot detection."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0 Safari/537.36"
    ]
    return {
        'Host': 'stats.nba.com',
        'User-Agent': random.choice(user_agents),
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
        'x-nba-stats-request-id': str(uuid.uuid4()),
        'Referer': 'https://www.nba.com/',
        'Accept': 'application/json, text/plain, */*',
    }

# --- 2. DATA ENGINES ---
@st.cache_data(ttl=3600)
def get_pace_data():
    try:
        df = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced', 
            headers=get_stealth_headers(),
            timeout=15
        ).get_data_frames()[0]
        return {row['TEAM_ID']: row['PACE'] for _, row in df.iterrows()}, df['PACE'].mean()
    except:
        return {t['id']: 100.0 for t in teams.get_teams()}, 100.0

# --- 3. FULL SIDEBAR RESTORATION ---
st.set_page_config(page_title="Sharp Pro v11.5", layout="wide")

with st.sidebar:
    st.title("🏀 Sharp Pro v11.5")
    st.markdown("---")
    
    # Navigation Mode
    mode = st.radio("Dashboard Navigation", ["Single Player Analysis", "Team Scanner"])
    
    st.markdown("---")
    # Betting Parameters
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)
    
    st.markdown("---")
    # Connection Settings
    st.subheader("System Status")
    use_backup = st.toggle("Enable G-Sheets Fallback", value=False)
    st.info("Bypass: UUID Fingerprinting ✅")

pace_map, avg_pace = get_pace_data()
intel = {"injuries": ["Nikola Jokic", "Kevin Durant", "Joel Embiid"], "ref_bias": {}}

# --- 4. MODE: SINGLE PLAYER ANALYSIS ---
if mode == "Single Player Analysis":
    st.header("👤 Player Matchup Engine")
    
    # STEP 1: Text Search (Accent Insensitive)
    search_query = st.text_input("1. Search Player Name (e.g., 'Doncic' or 'Jokic')", "Luka")
    
    # Filter players using normalized names
    norm_query = normalize_string(search_query)
    all_players = players.get_players()
    matches = [
        p for p in all_players 
        if norm_query in normalize_string(p['full_name']) and p['is_active']
    ]
    
    # STEP 2: Selection Box
    if matches:
        sel_p = st.selectbox("2. Confirm Selection", matches, format_func=lambda x: x['full_name'])
        
        if st.button("🚀 Run Full Analysis"):
            with st.spinner(f"Fetching data for {sel_p['full_name']}..."):
                try:
                    # Fetching Data
                    time.sleep(random.uniform(0.5, 1.2)) # Random delay for stealth
                    log = playergamelog.PlayerGameLog(
                        player_id=sel_p['id'], 
                        season='2025-26', 
                        headers=get_stealth_headers(), 
                        timeout=20
                    ).get_data_frames()[0]
                    
                    if not log.empty:
                        if stat_cat == "PRA": 
                            log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                        
                        raw_avg = log[stat_cat].head(10).mean()
                        final_proj = raw_avg * (1.0) # Simplified for example
                        prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

                        # UI Display
                        c1, c2, c3 = st.columns(3)
                        c1.metric("10-Game Avg", round(raw_avg, 1))
                        c2.metric("Projected", round(final_proj, 1))
                        c3.metric("Over Prob", f"{round(prob_over, 1)}%")
                        
                        st.subheader("Recent Performance Trend")
                        fig = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                        fig.add_hline(y=line, line_color="red", line_dash="dash")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"API Error. Try re-running or wait 30 seconds.")

# --- 5. MODE: TEAM SCANNER ---
elif mode == "Team Scanner":
    st.header("🔍 Full Roster Value Scanner")
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Roster"):
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=sel_team['id'], timeout=20).get_data_frames()[0]
            scan_results = []
            
            with st.status("Scanning roster via stealth tunnel...") as status:
                for i, p in roster.head(8).iterrows(): # Scans top 8 to stay safe
                    status.update(label=f"Analyzing {p['PLAYER']}...")
                    time.sleep(1.0)
                    try:
                        p_log = playergamelog.PlayerGameLog(player_id=p['PLAYER_ID'], timeout=15, headers=get_stealth_headers()).get_data_frames()[0]
                        if not p_log.empty:
                            avg = p_log['PTS'].head(5).mean() # Defaulting to PTS for scan
                            scan_results.append({"Player": p['PLAYER'], "Avg PTS": round(avg, 1)})
                    except: continue
            
            st.table(pd.DataFrame(scan_results))
        except:
            st.error("Scanner Timed Out.")
