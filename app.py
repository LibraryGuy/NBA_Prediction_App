import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import time
import random
import uuid
from streamlit_gsheets import GSheetsConnection
from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.stats.static import players

# --- 1. STEALTH & IDENTITY GENERATOR ---
def get_stealth_headers():
    """Generates a unique digital fingerprint for every single request."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ]
    
    # This generates a 'New ID' for the NBA API to see
    random_token = str(uuid.uuid4()) # Unique ID per request
    
    return {
        'Host': 'stats.nba.com',
        'Connection': 'keep-alive',
        'User-Agent': random.choice(user_agents),
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
        'x-nba-stats-request-id': random_token, # Custom unique ID
        'Referer': f'https://www.nba.com/player/{random.randint(100, 10000)}',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
    }

# --- 2. DATA ENGINES (API + SHEETS) ---
def fetch_player_stats(player_id, stat_cat):
    """Try Live API with stealth, fallback to Google Sheets if blocked."""
    
    # 1. Try Live API with Stealth Headers
    try:
        headers = get_stealth_headers()
        log = playergamelog.PlayerGameLog(
            player_id=player_id, 
            season='2025-26', 
            headers=headers, 
            timeout=15
        ).get_data_frames()[0]
        
        if not log.empty:
            st.sidebar.success("🟢 Live Data Active")
            return log, "Live"
    except Exception:
        st.sidebar.warning("🟡 NBA Blocked API. Switching to Sheets...")

    # 2. Fallback to Google Sheets
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        # Replace the URL below with your actual Google Sheet URL
        df = conn.read(spreadsheet=st.secrets["connections"]["gsheets"]["spreadsheet"], ttl="1h")
        player_data = df[df['PLAYER_ID'] == player_id]
        
        if not player_data.empty:
            return player_data, "Cached (Sheets)"
    except Exception as e:
        st.sidebar.error("🔴 All Data Sources Failed")
        return pd.DataFrame(), "Failed"

# --- 3. DASHBOARD UI ---
st.set_page_config(page_title="Sharp Pro v11.0", layout="wide")

# Sidebar Configuration
with st.sidebar:
    st.title("🏀 Sharp Pro v11.0")
    st.info("Bypass: UUID Fingerprinting 🛡️")
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST"])
    line = st.number_input("Sportsbook Line", value=15.5)

# Player Search
search = st.text_input("Search Player", "Nikola Jokic")
matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]

if matches:
    sel_p = st.selectbox("Confirm Player", matches, format_func=lambda x: x['full_name'])
    
    if st.button("🚀 Run Analysis"):
        with st.spinner("Rotating Headers & Fetching..."):
            data, source = fetch_player_stats(sel_p['id'], stat_cat)
            
            if not data.empty:
                avg_val = data[stat_cat].head(10).mean()
                prob_over = (1 - poisson.cdf(line - 0.5, avg_val)) * 100
                
                # Main Display
                st.header(f"{sel_p['full_name']} Analysis")
                st.caption(f"Data Source: {source}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Projected", round(avg_val, 1))
                m2.metric("Over Probability", f"{round(prob_over, 1)}%")
                m3.metric("Request ID", get_stealth_headers()['x-nba-stats-request-id'][:8])
                
                st.dataframe(data.head(10))
            else:
                st.error("Could not retrieve data for this player.")
