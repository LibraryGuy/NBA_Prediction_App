import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime
import pytz
import time
from nba_api.stats.endpoints import (playergamelog, scoreboardv2, 
                                     commonteamroster, leaguedashteamstats, 
                                     commonplayerinfo, leaguegamefinder)
from nba_api.stats.static import players, teams

# --- 1. SETTINGS & STABLE ENGINES ---
st.set_page_config(page_title="Sharp Pro v10.12", layout="wide", page_icon="🏀")

# Custom Headers to mimic a real browser
HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.nba.com/',
}

@st.cache_data(ttl=1800)
def get_intel():
    """Static intelligence for injuries and referee biases."""
    return {
        "injuries": ["Nikola Jokic", "Kevin Durant", "Joel Embiid", "Ja Morant", "Giannis"],
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96},
            "Marc Davis": {"type": "Over", "impact": 1.05},
            "Jacyn Goble": {"type": "Over", "impact": 1.04},
            "Tony Brothers": {"type": "Under", "impact": 0.97},
            "James Williams": {"type": "Over", "impact": 1.03}
        }
    }

def safe_api_call(endpoint_class, timeout=8, **kwargs):
    """Bypasses timeouts and handles empty data frames gracefully."""
    try:
        time.sleep(0.5) # Gentle rate limiting
        call = endpoint_class(**kwargs, headers=HEADERS, timeout=timeout)
        return call.get_data_frames()
    except:
        return None

# --- 2. SIDEBAR NAVIGATION ---
intel = get_intel()
team_list = sorted(teams.get_teams(), key=lambda x: x['full_name'])

with st.sidebar:
    st.title("🏀 Sharp Pro v10.12")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Value Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)
    sim_runs = st.slider("Monte Carlo Iterations", 1000, 10000, 5000)
    st.divider()
    st.caption("Context: 2025-26 NBA Season")

# --- 3. MODE: SINGLE PLAYER ANALYSIS ---
if mode == "Single Player Analysis":
    search = st.text_input("Search Player", "Peyton Watson")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Select Player", matches, format_func=lambda x: x['full_name'])
        
        if st.button("🚀 Execute Analysis"):
            with st.status("Engine Startup...") as status:
                # A. Fetch Game Logs
                status.update(label="Fetching Player Performance...")
                logs_frames = safe_api_call(playergamelog.PlayerGameLog, player_id=sel_p['id'], season='2025-26')
                
                # B. Fetch Context (Refs & Opponent)
                status.update(label="Syncing Official Ref Assignments...")
                today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
                sb_frames = safe_api_call(scoreboardv2.ScoreboardV2, game_date=today)
                
                if logs_frames:
                    df = logs_frames[0]
                    if stat_cat == "PRA": df['PRA'] = df['PTS'] + df['REB'] + df['AST']
                    
                    # C. Defensive Ref Logic (Fixes KeyError)
                    ref_name = "Unknown/Pending"
                    if sb_frames and len(sb_frames) > 2:
                        off_df = sb_frames[2]
                        if not off_df.empty and 'OFFICIAL_NAME' in off_df.columns:
                            # Grabbing the first official listed as a proxy for the crew
                            ref_name = off_df.iloc[0]['OFFICIAL_NAME']
                    
                    ref_impact = intel['ref_bias'].get(ref_name, {"impact": 1.0})['impact']
                    
                    # D. Analytics
                    status.update(label="Running Projections...")
                    l10 = df[stat_cat].head(10)
                    mean_val = l10.mean()
                    std_dev = l10.std() if len(l10) > 1 else 1.0
                    
                    # Adjust for Ref
                    adj_mean = mean_val * ref_impact
                    
                    # MONTE CARLO SIMULATION
                    status.update(label="Simulating 5,000 Game Realities...")
                    sim_results = np.random.normal(adj_mean, std_dev, sim_runs)
                    sim_results = np.maximum(sim_results, 0) # Stats can't be negative
                    mc_prob_over = (sim_results > line).mean() * 100
                    
                    # POISSON PROBABILITY
                    poi_prob_over = (1 - poisson.cdf(line - 0.5, adj_mean)) * 100

                    status.update(label="Analysis Complete", state="complete")

                    # --- DASHBOARD RENDER ---
                    st.header(f"{sel_p['full_name']} vs. Today's Matchup")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Sharp Projection", round(adj_mean, 1))
                    m2.metric("Official Ref", ref_name)
                    m3.metric("Monte Carlo (Over)", f"{round(mc_prob_over, 1)}%")
                    m4.metric("Poisson (Over)", f"{round(poi_prob_over, 1)}%")

                    # Visuals
                    v1, v2 = st.columns(2)
                    with v1:
                        st.subheader("Monte Carlo Distribution")
                        fig_mc = px.histogram(sim_results, nbins=30, labels={'value': stat_cat}, color_discrete_sequence=['#636EFA'])
                        fig_mc.add_vline(x=line, line_dash="dash", line_color="red", annotation_text="Line")
                        st.plotly_chart(fig_mc, use_container_width=True)
                    
                    with v2:
                        st.subheader("Season Trend (Last 10)")
                        fig_t = px.line(df.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                        fig_t.add_hline(y=line, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_t, use_container_width=True)

                    st.divider()
                    st.subheader("Historical H2H Context")
                    # Fetch H2H specifically
                    h2h_frames = safe_api_call(leaguegamefinder.LeagueGameFinder, player_id_nullable=sel_p['id'])
                    if h2h_frames:
                        st.dataframe(h2h_frames[0][['GAME_DATE', 'MATCHUP', 'WL', stat_cat]].head(5), use_container_width=True)

# --- 4. MODE: TEAM VALUE SCANNER ---
elif mode == "Team Value Scanner":
    sel_team = st.selectbox("Select Team to Scan", team_list, format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Rotation Players"):
        with st.status(f"Scanning {sel_team['full_name']} Rotation...") as status:
            roster = safe_api_call(commonteamroster.CommonTeamRoster, team_id=sel_team['id'])
            
            if roster:
                results = []
                # Scan top 8 players to avoid NBA.com 403 blocks
                for _, player in roster[0].head(8).iterrows():
                    p_log = safe_api_call(playergamelog.PlayerGameLog, player_id=player['PLAYER_ID'], season='2025-26')
                    if p_log:
                        p_df = p_log[0]
                        if stat_cat == "PRA": p_df['PRA'] = p_df['PTS'] + p_df['REB'] + p_df['AST']
                        avg = p_df[stat_cat].head(5).mean()
                        prob = (1 - poisson.cdf(line - 0.5, avg)) * 100
                        results.append({
                            "Player": player['PLAYER'],
                            "L5 Avg": round(avg, 1),
                            "Over %": f"{round(prob, 1)}%",
                            "Signal": "🔥" if prob > 60 else ("❄️" if prob < 40 else "➖")
                        })
                
                status.update(label="Scan Complete", state="complete")
                st.table(pd.DataFrame(results))
