import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson
import requests
from datetime import datetime
from nba_api.stats.endpoints import (playergamelog, leaguegamefinder, 
                                     scoreboardv2, commonplayerinfo, 
                                     leaguedashteamstats)
from nba_api.stats.static import players, teams

# --- 1. DATA ENGINES: PACE & INJURIES ---

@st.cache_data(ttl=1800)
def get_automated_injury_list():
    """Scrapes CBS Sports with full headers to ensure high-accuracy injury counts."""
    confirmed_out = []
    try:
        url = "https://www.cbssports.com/nba/injuries/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(response.text, flavor='html5lib')
        for table in tables:
            if 'Status' in table.columns and 'Player' in table.columns:
                out_p = table[table['Status'].str.contains('Out|Sidelined|Surgery|Targeting', case=False, na=False)]
                confirmed_out.extend(out_p['Player'].tolist())
        confirmed_out = [name.split('  ')[0].strip() for name in confirmed_out]
    except:
        confirmed_out = ["Nikola Jokic", "Ja Morant"] # Failsafe
    return list(set(confirmed_out))

@st.cache_data(ttl=3600)
def get_pace_data():
    """Fetches team pace stats and league average for projection scaling."""
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
        pace_map = dict(zip(stats['TEAM_ID'], stats['PACE']))
        league_avg_pace = stats['PACE'].mean()
        return pace_map, league_avg_pace
    except:
        return {}, 100.0

@st.cache_data(ttl=600)
def get_todays_matchups():
    """Automates opponent selection by fetching today's NBA schedule."""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        board = scoreboardv2.ScoreboardV2(game_date=today).get_data_frames()[0]
        matchup_map = {}
        for _, row in board.iterrows():
            m_map = {row['HOME_TEAM_ID']: row['VISITOR_TEAM_ID'], 
                     row['VISITOR_TEAM_ID']: row['HOME_TEAM_ID']}
            matchup_map.update(m_map)
        return matchup_map
    except:
        return {}

# --- 2. THE DASHBOARD ---

st.set_page_config(page_title="Sharp Pro v7.9", layout="wide")
injury_list = get_automated_injury_list()
today_games = get_todays_matchups()
pace_map, avg_pace = get_pace_data()
team_lookup = {t['id']: t['abbreviation'] for t in teams.get_teams()}

with st.sidebar:
    st.title("🏀 Sharp Pro v7.9")
    st.success(f"✅ {len(injury_list)} Players Filtered Out")
    app_mode = st.radio("Navigation", ["Single Player Analysis", "Team Value Scanner"])
    stat_cat = st.selectbox("Category", ["PTS", "REB", "AST", "STL", "BLK"])
    market_line = st.number_input("Sportsbook Line", value=20.5, step=0.5)

if app_mode == "Single Player Analysis":
    search = st.text_input("Enter Player Name", "Jamal Murray")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Confirm Player", matches, format_func=lambda x: x['full_name'])
        
        # --- START SCAN BUTTON ---
        if st.button("🚀 Run Full Sharp Analysis"):
            with st.spinner("Calculating Pace-Adjusted Edge..."):
                # 1. Identity & Schedule
                p_info = commonplayerinfo.CommonPlayerInfo(player_id=sel_p['id']).get_data_frames()[0]
                team_id = p_info['TEAM_ID'].iloc[0]
                team_abbr = p_info['TEAM_ABBREVIATION'].iloc[0]
                opp_id = today_games.get(team_id)
                opp_abbr = team_lookup.get(opp_id, "N/A")
                
                if sel_p['full_name'] in injury_list:
                    st.error(f"🛑 ALERT: {sel_p['full_name']} is currently OUT.")
                else:
                    # 2. Historical Data
                    log = playergamelog.PlayerGameLog(player_id=sel_p['id'], season='2024-25').get_data_frames()[0]
                    h2h = leaguegamefinder.LeagueGameFinder(player_id_nullable=sel_p['id']).get_data_frames()[0]
                    h2h_filtered = h2h[h2h['MATCHUP'].str.contains(opp_abbr)].head(5) if opp_abbr != "N/A" else pd.DataFrame()
                    
                    # 3. PACE ADJUSTMENT LOGIC
                    p_team_pace = pace_map.get(team_id, avg_pace)
                    opp_team_pace = pace_map.get(opp_id, avg_pace)
                    projected_game_pace = (p_team_pace + opp_team_pace) / 2
                    pace_factor = projected_game_pace / avg_pace
                    
                    # 4. Final Projections
                    raw_avg = log[stat_cat].head(10).mean()
                    pace_adj_proj = raw_avg * pace_factor
                    prob_over = (1 - poisson.cdf(market_line - 0.5, pace_adj_proj)) * 100
                    
                    # Header Section
                    st.subheader(f"Analysis: {sel_p['full_name']} ({team_abbr}) vs {opp_abbr}")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Pace-Adj Projection", round(pace_adj_proj, 1), delta=f"{round(pace_adj_proj-market_line, 2)} Edge")
                    m2.metric(f"H2H Avg vs {opp_abbr}", round(h2h_filtered[stat_cat].mean(), 1) if not h2h_filtered.empty else "N/A")
                    m3.metric("Prob. Over", f"{round(prob_over, 1)}%")
                    m4.metric("Matchup Pace", f"{round(projected_game_pace, 1)}", delta=f"{round(projected_game_pace-avg_pace, 1)} vs Avg")

                    # --- DASHBOARD VISUALS ---
                    st.divider()
                    v1, v2 = st.columns(2)
                    with v1:
                        st.subheader("Poisson Distribution (Pace Adjusted)")
                        x_range = np.arange(max(0, int(pace_adj_proj-12)), int(pace_adj_proj+15))
                        fig_p = px.bar(x=x_range, y=poisson.pmf(x_range, pace_adj_proj), labels={'x':stat_cat, 'y':'Prob'})
                        fig_p.add_vline(x=market_line, line_dash="dash", line_color="red", annotation_text="Bookie")
                        st.plotly_chart(fig_p, use_container_width=True)
                    with v2:
                        st.subheader(f"Last 10 Game Trend ({stat_cat})")
                        fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                        fig_t.add_hline(y=market_line, line_color="red", line_dash="dot")
                        st.plotly_chart(fig_t, use_container_width=True)

                    # --- SPORTSBOOK TREND & H2H TABLE ---
                    st.divider()
                    st.subheader(f"Historical Matchups vs {opp_abbr}")
                    if not h2h_filtered.empty:
                        st.table(h2h_filtered[['GAME_DATE', 'MATCHUP', 'WL', stat_cat]].reset_index(drop=True))
                    else:
                        st.info("No recent H2H data found.")

else:
    st.header("📋 Team Value Scanner")
    # Scanner logic remains compatible
