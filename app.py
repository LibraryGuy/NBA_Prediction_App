import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson
import requests
import time
from nba_api.stats.endpoints import playergamelog, commonteamroster, leaguegamefinder
from nba_api.stats.static import players, teams

# --- 1. DATA ENGINES ---

@st.cache_data(ttl=1800)
def get_automated_injury_list():
    confirmed_out = []
    try:
        # Added User-Agent to prevent 0-player results due to blocking
        url = "https://www.cbssports.com/nba/injuries/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        # Using 'html5lib' for the best parsing of CBS tables
        tables = pd.read_html(response.text, flavor='html5lib')
        for table in tables:
            if 'Status' in table.columns and 'Player' in table.columns:
                # Target 'Out' or 'Sidelined'
                out_p = table[table['Status'].str.contains('Out|Sidelined|Surgery|Targeting', case=False, na=False)]
                confirmed_out.extend(out_p['Player'].tolist())
        
        # Clean specific string artifacts
        confirmed_out = [name.split('  ')[0].replace(' (C)', '').replace(' (PF)', '').strip() for name in confirmed_out]
    except Exception as e:
        # Hard-coded failsafe for major stars if scraper fails
        confirmed_out = ["Nikola Jokic", "Fred VanVleet", "Ja Morant", "Tyrese Haliburton"]
        
    return list(set(confirmed_out))

@st.cache_data(ttl=3600)
def get_h2h_stats(player_id, opp_team_abbr):
    """Fetches games played by this player specifically against the opponent."""
    try:
        # Get all games for this player
        finder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id)
        df = finder.get_data_frames()[0]
        # Filter for the specific opponent team abbreviation in the MATCHUP column
        h2h_df = df[df['MATCHUP'].str.contains(opp_team_abbr)]
        return h2h_df.head(5) # Return last 5 matchups
    except:
        return pd.DataFrame()

# --- 2. DASHBOARD SETUP ---

st.set_page_config(page_title="Sharp Pro v7.7", layout="wide")
injury_list = get_automated_injury_list()
team_map = {t['abbreviation']: t['id'] for t in teams.get_teams()}

with st.sidebar:
    st.title("🏀 Sharp Pro v7.7")
    # Accurate notification
    if len(injury_list) > 10:
        st.success(f"✅ {len(injury_list)} Players Filtered (Live)")
    else:
        st.warning("⚠️ Injury Sync limited. Using manual list.")
        
    mode = st.radio("App Mode", ["Single Player", "Team Scanner"])
    stat_cat = st.selectbox("Category", ["PTS", "REB", "AST", "STL", "BLK"])
    consensus_line = st.number_input("Sportsbook Line", value=20.5, step=0.5)

if mode == "Single Player":
    search = st.text_input("Search Player", "Jamal Murray")
    opp_abbr = st.selectbox("Tonight's Opponent", sorted(list(team_map.keys())), index=8) # Default to DEN/GSW etc
    p_matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if p_matches:
        sel_p = st.selectbox("Select Player", p_matches, format_func=lambda x: x['full_name'])
        
        if st.button("🚀 Start Player Analysis"):
            if sel_p['full_name'] in injury_list:
                st.error(f"🛑 {sel_p['full_name']} is currently OUT.")
            else:
                with st.spinner("Crunching H2H & Trends..."):
                    # 1. Fetch Data
                    log = playergamelog.PlayerGameLog(player_id=sel_p['id'], season='2024-25').get_data_frames()[0]
                    h2h_df = get_h2h_stats(sel_p['id'], opp_abbr)
                    
                    # 2. Metrics
                    proj_val = log[stat_cat].head(10).mean() * 1.05
                    prob_over = (1 - poisson.cdf(consensus_line - 0.5, proj_val)) * 100
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Projection", round(proj_val, 1))
                    m2.metric(f"L5 H2H Avg vs {opp_abbr}", round(h2h_df[stat_cat].mean(), 1) if not h2h_df.empty else "N/A")
                    m3.metric("Edge", f"{round(proj_val - consensus_line, 1)}")

                    # --- VISUALS ---
                    st.divider()
                    g1, g2 = st.columns(2)
                    with g1:
                        fig_pois = px.bar(x=np.arange(0, 40), y=poisson.pmf(np.arange(0, 40), proj_val), title="Poisson Spread")
                        st.plotly_chart(fig_pois, use_container_width=True)
                    with g2:
                        st.subheader(f"📊 Head-to-Head vs {opp_abbr}")
                        if not h2h_df.empty:
                            st.table(h2h_df[['GAME_DATE', 'MATCHUP', stat_cat, 'WL']])
                        else:
                            st.info("No recent H2H matchups found.")

                    # --- TREND LINE ---
                    st.subheader("📈 Season Trend Line")
                    fig_trend = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                    fig_trend.add_hline(y=consensus_line, line_color="red", line_dash="dash")
                    st.plotly_chart(fig_trend, use_container_width=True)
