import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import time
import random
import uuid
import unicodedata
import requests
from nba_api.stats.endpoints import (playergamelog, commonplayerinfo, 
                                     leaguedashteamstats, commonteamroster,
                                     leaguegamefinder)
from nba_api.stats.static import players, teams

# --- 1. CORE UTILITIES ---

def normalize_string(text):
    """Removes accents (Dončić -> doncic) for foolproof searching."""
    return "".join(
        c for c in unicodedata.normalize('NFD', text.lower())
        if unicodedata.category(c) != 'Mn'
    )

def get_stealth_session():
    """Creates a requests session with randomized human-like fingerprints."""
    session = requests.Session()
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ]
    session.headers.update({
        'Host': 'stats.nba.com',
        'User-Agent': random.choice(user_agents),
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
        'Referer': 'https://www.nba.com/',
        'Accept': 'application/json, text/plain, */*',
        'x-nba-stats-request-id': str(uuid.uuid4())
    })
    return session

# --- 2. DATA FALLBACK ENGINE ---

def fetch_with_fallback(endpoint_class, **kwargs):
    """Attempt to fetch live data; if blocked, return empty or cached if available."""
    try:
        # Essential delay to prevent rate-limiting
        time.sleep(random.uniform(0.6, 1.4)) 
        session = get_stealth_session()
        # Pass the session directly into the nba_api endpoint
        instance = endpoint_class(proxy=None, headers=session.headers, timeout=20, **kwargs)
        return instance.get_data_frames()[0], "LIVE"
    except Exception as e:
        # This is where you'd point to a local CSV if you have one uploaded to GitHub
        try:
            fallback_df = pd.read_csv("backup_nba_stats.csv")
            return fallback_df, "BACKUP"
        except:
            return pd.DataFrame(), "FAILED"

# --- 3. DASHBOARD CONFIG ---

st.set_page_config(page_title="Sharp Pro v11.5", layout="wide")

with st.sidebar:
    st.title("🏀 Sharp Pro v11.5")
    st.markdown("---")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    st.markdown("---")
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)
    st.divider()
    st.info("System Status: Stealth Session Active ✅")

# --- 4. MODE: SINGLE PLAYER ANALYSIS ---

if mode == "Single Player Analysis":
    st.header("👤 Player Matchup Engine")
    
    # TWO-STEP ACCENT-PROOF SEARCH
    c1, c2 = st.columns(2)
    with c1:
        search_query = st.text_input("1. Search Name (e.g. 'Doncic' or 'Jokic')", "Luka")
    
    norm_query = normalize_string(search_query)
    active_players = [p for p in players.get_players() if p['is_active']]
    matches = [p for p in active_players if norm_query in normalize_string(p['full_name'])]
    
    with c2:
        if matches:
            sel_p = st.selectbox("2. Select Confirmed Player", matches, format_func=lambda x: x['full_name'])
        else:
            st.error("No active players found. Try a different spelling.")
            st.stop()

    if st.button("🚀 Run Full Analysis"):
        with st.spinner(f"Requesting clearance for {sel_p['full_name']}..."):
            
            # GET GAME LOGS
            log, source = fetch_with_fallback(playergamelog.PlayerGameLog, player_id=sel_p['id'], season='2025-26')
            
            if not log.empty:
                if stat_cat == "PRA":
                    log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                
                # METRIC CALCULATIONS
                l10 = log.head(10)
                raw_avg = l10[stat_cat].mean()
                proj = raw_avg * 1.05 # Simple adjustment factor
                prob_over = (1 - poisson.cdf(line - 0.5, proj)) * 100

                # DISPLAY TOP METRICS
                st.subheader(f"Results for {sel_p['full_name']} (Source: {source})")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("L10 Average", round(raw_avg, 1))
                m2.metric("Projected", round(proj, 1))
                m3.metric("Over Probability", f"{round(prob_over, 1)}%")
                m4.metric("Last Game", log[stat_cat].iloc[0])

                # GRAPHS
                g1, g2 = st.columns(2)
                with g1:
                    st.write("**Poisson Distribution (Range of Outcomes)**")
                    x = np.arange(max(0, int(proj-15)), int(proj+20))
                    y = poisson.pmf(x, proj)
                    fig_p = px.bar(x=x, y=y, labels={'x': stat_cat, 'y': 'Probability'})
                    fig_p.add_vline(x=line, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_p, use_container_width=True)
                
                with g2:
                    st.write("**Performance Trend (Last 10)**")
                    fig_l = px.line(l10.iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                    fig_l.add_hline(y=line, line_color="red", line_dash="dash")
                    st.plotly_chart(fig_l, use_container_width=True)

                # BETTING BLUEPRINT
                st.divider()
                st.subheader("🎯 Betting Blueprint")
                b1, b2, b3 = st.columns(3)
                with b1: st.info(f"**Main Play:** {stat_cat} {'OVER' if prob_over > 50 else 'UNDER'} {line}")
                with b2: st.success(f"**Safe Play:** {stat_cat} OVER {line - 4}")
                with b3: st.warning(f"**Ladder Play:** {stat_cat} OVER {line + 5}")

            else:
                st.error("NBA servers blocked the request. Please wait 60 seconds or upload a 'backup_nba_stats.csv' to your GitHub.")

# --- 5. MODE: TEAM SCANNER ---

elif mode == "Team Scanner":
    st.header("🔍 Value Scanner")
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Roster"):
        roster, _ = fetch_with_fallback(commonteamroster.CommonTeamRoster, team_id=sel_team['id'])
        if not roster.empty:
            scan_results = []
            progress = st.progress(0)
            for i, p in roster.head(10).iterrows(): # Scan top 10 for safety
                progress.progress((i+1)/10)
                p_log, _ = fetch_with_fallback(playergamelog.PlayerGameLog, player_id=p['PLAYER_ID'])
                if not p_log.empty:
                    avg = p_log[stat_cat].head(5).mean()
                    scan_results.append({"Player": p['PLAYER'], "L5 Avg": round(avg, 1)})
                time.sleep(1) # Extra gap between players
            st.table(pd.DataFrame(scan_results))
