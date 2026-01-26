import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
import time
import random
import uuid
import unicodedata
from nba_api.stats.endpoints import (playergamelog, commonplayerinfo, 
                                     leaguedashteamstats, commonteamroster)
from nba_api.stats.static import players, teams

# --- 1. UTILITIES ---
def normalize_string(text):
    """Handles accents like Dončić -> doncic."""
    return "".join(
        c for c in unicodedata.normalize('NFD', text.lower())
        if unicodedata.category(c) != 'Mn'
    )

def get_stealth_headers():
    return {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0 Safari/537.36',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
        'Referer': 'https://www.nba.com/',
    }

# --- 2. THE ULTIMATE FETCH FUNCTION ---
def smart_fetch(endpoint_class, **kwargs):
    """Tries Live API with Proxy, Falls back to CSV on error."""
    # Tip: Enter a proxy URL here if you have one (e.g., "http://user:pass@proxy.com:8080")
    # Leave None to try a direct stealth connection
    PROXY_URL = st.secrets.get("PROXY_URL", None) 
    
    try:
        time.sleep(random.uniform(0.8, 1.5)) # Prevent rapid firing
        instance = endpoint_class(
            proxy=PROXY_URL, 
            headers=get_stealth_headers(), 
            timeout=25, 
            **kwargs
        )
        return instance.get_data_frames()[0], "LIVE"
    except Exception as e:
        try:
            # TRY TO LOAD YOUR BACKUP CSV
            df = pd.read_csv("backup_nba_stats.csv")
            # If the backup is the whole league, filter for the player if needed
            if 'PLAYER_ID' in df.columns and 'player_id' in kwargs:
                df = df[df['PLAYER_ID'] == kwargs['player_id']]
            return df, "OFFLINE CACHE"
        except:
            return pd.DataFrame(), "ERROR"

# --- 3. DASHBOARD UI ---
st.set_page_config(page_title="Sharp Pro v11.5", layout="wide")

with st.sidebar:
    st.title("🏀 Sharp Pro v11.5")
    st.markdown("---")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    st.markdown("---")
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)
    st.divider()
    st.info("Bypassing Data-Center Blocks...")

# --- 4. MODE: ANALYSIS ---
if mode == "Single Player Analysis":
    st.header("👤 Player Matchup Engine")
    
    # ACCENT-PROOF SEARCH
    c1, c2 = st.columns(2)
    with c1:
        query = st.text_input("1. Search Name (Doncic, Jokic, etc.)", "Luka")
    
    norm_query = normalize_string(query)
    matches = [p for p in players.get_players() if norm_query in normalize_string(p['full_name']) and p['is_active']]
    
    if matches:
        with c2:
            sel_p = st.selectbox("2. Confirm Selection", matches, format_func=lambda x: x['full_name'])
        
        if st.button("🚀 Run Full Analysis"):
            with st.spinner("Decoding NBA API defense..."):
                log, source = smart_fetch(playergamelog.PlayerGameLog, player_id=sel_p['id'], season='2025-26')
                
                if not log.empty:
                    if stat_cat == "PRA":
                        log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                    
                    # PROJECTION LOGIC
                    l10 = log.head(10)
                    raw_avg = l10[stat_cat].mean()
                    proj = raw_avg * 1.05 # Matchup Weight
                    prob_over = (1 - poisson.cdf(line - 0.5, proj)) * 100

                    # METRICS
                    st.subheader(f"Results for {sel_p['full_name']} (Data: {source})")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("10-Game Avg", round(raw_avg, 1))
                    m2.metric("Sharp Projection", round(proj, 1))
                    m3.metric("Over Probability", f"{round(prob_over, 1)}%")

                    # VISUALS
                    g1, g2 = st.columns(2)
                    with g1:
                        st.write("**Outcome Probabilities**")
                        x = np.arange(max(0, int(proj-15)), int(proj+15))
                        fig = px.bar(x=x, y=poisson.pmf(x, proj))
                        fig.add_vline(x=line, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
                    with g2:
                        st.write("**Recent Trend**")
                        fig_line = px.line(l10.iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                        fig_line.add_hline(y=line, line_color="red", line_dash="dash")
                        st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.error("NBA Server Blocked Request. Please try again in 60s or check your Proxy.")

elif mode == "Team Scanner":
    st.header("🔍 Value Scanner")
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    if st.button("📡 Scan Roster"):
        roster, _ = smart_fetch(commonteamroster.CommonTeamRoster, team_id=sel_team['id'])
        if not roster.empty:
            st.table(roster[['PLAYER', 'NUM', 'POSITION']].head(10))
