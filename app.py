import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime
import pytz
import time
import requests
from nba_api.stats.endpoints import (playergamelog, scoreboardv2, 
                                     commonteamroster, leaguedashteamstats)
from nba_api.stats.static import players, teams

# --- 1. SETTINGS & STABLE ENGINES ---
st.set_page_config(page_title="Sharp Pro v10.11", layout="wide", page_icon="🏀")

@st.cache_data(ttl=1800)
def get_intel():
    return {
        "injuries": ["Nikola Jokic", "Kevin Durant", "Joel Embiid", "Ja Morant", "Giannis"],
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96, "color": "red"},
            "Marc Davis": {"type": "Over", "impact": 1.05, "color": "green"},
            "Jacyn Goble": {"type": "Over", "impact": 1.04, "color": "green"},
            "Tony Brothers": {"type": "Under", "impact": 0.97, "color": "red"},
            "James Williams": {"type": "Over", "impact": 1.03, "color": "green"}
        }
    }

def safe_fetch(endpoint_class, timeout=7, **kwargs):
    """Silent fail wrapper to prevent app crashes during NBA.com outages."""
    try:
        data = endpoint_class(**kwargs, timeout=timeout).get_data_frames()
        return data if data else None
    except: return None

# --- 2. SIDEBAR & NAVIGATION ---
intel = get_intel()
team_list = sorted(teams.get_teams(), key=lambda x: x['full_name'])

with st.sidebar:
    st.title("🏀 Sharp Pro v10.11")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Value Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)
    st.divider()
    st.caption("Targeting: 2025-26 Season Data")

# --- 3. MODE: SINGLE PLAYER ANALYSIS ---
if mode == "Single Player Analysis":
    search = st.text_input("Find Player", "Peyton Watson")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Select Player", matches, format_func=lambda x: x['full_name'])
        
        if st.button("🚀 Run Analysis"):
            with st.status("Gathering Intelligence...") as status:
                # A. Data Fetching
                logs = safe_fetch(playergamelog.PlayerGameLog, player_id=sel_p['id'], season='2025-26')
                today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
                sb = safe_fetch(scoreboardv2.ScoreboardV2, game_date=today)
                
                if logs:
                    df = logs[0]
                    if stat_cat == "PRA": df['PRA'] = df['PTS'] + df['REB'] + df['AST']
                    
                    # B. Ref Hunting
                    ref_name = "TBD/Blocked"
                    if sb and len(sb) > 2:
                        # Logic to find current player's game ref
                        ref_name = sb[2].iloc[0]['OFFICIAL_NAME'] # Simplified for demo
                    
                    ref_data = intel['ref_bias'].get(ref_name, {"type": "Neutral", "impact": 1.0, "color": "gray"})
                    
                    # C. Calculations
                    avg_10 = df[stat_cat].head(10).mean()
                    final_proj = avg_10 * ref_data['impact']
                    prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

                    status.update(label="Analysis Complete!", state="complete")

                    # --- DASHBOARD UI ---
                    st.header(f"{sel_p['full_name']} Analysis")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Final Projection", round(final_proj, 1))
                    c2.metric("Ref: " + ref_name, f"{int((ref_data['impact']-1)*100)}%", delta_color="normal")
                    c3.metric("L10 Average", round(avg_10, 1))
                    c4.metric("Win Prob (Over)", f"{round(prob_over, 1)}%")

                    # Visuals Section
                    v1, v2 = st.columns(2)
                    with v1:
                        st.subheader("Poisson Probability Curve")
                        x = np.arange(max(0, int(final_proj-15)), int(final_proj+15))
                        y = poisson.pmf(x, final_proj)
                        fig = px.bar(x=x, y=y, labels={'x': stat_cat, 'y': 'Probability'})
                        fig.add_vline(x=line, line_dash="dash", line_color="red", annotation_text="Line")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with v2:
                        st.subheader("Last 10 Performance Trend")
                        fig2 = px.line(df.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                        fig2.add_hline(y=line, line_dash="dash", line_color="red")
                        st.plotly_chart(fig2, use_container_width=True)

# --- 4. MODE: TEAM VALUE SCANNER ---
elif mode == "Team Value Scanner":
    sel_team = st.selectbox("Select Team", team_list, format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Roster for Value"):
        with st.status(f"Scanning {sel_team['full_name']}...") as status:
            roster_data = safe_fetch(commonteamroster.CommonTeamRoster, team_id=sel_team['id'])
            
            if roster_data:
                results = []
                for _, player in roster_data[0].head(10).iterrows(): # Limits to 10 to avoid 403 blocks
                    p_log = safe_fetch(playergamelog.PlayerGameLog, player_id=player['PLAYER_ID'], season='2025-26')
                    if p_log:
                        df_p = p_log[0]
                        if stat_cat == "PRA": df_p['PRA'] = df_p['PTS'] + df_p['REB'] + df_p['AST']
                        p_avg = df_p[stat_cat].head(5).mean()
                        p_prob = (1 - poisson.cdf(line - 0.5, p_avg)) * 100
                        results.append({
                            "Player": player['PLAYER'],
                            "L5 Avg": round(p_avg, 1),
                            "Over Prob": f"{round(p_prob, 1)}%",
                            "Signal": "🔥" if p_prob > 60 else ("❄️" if p_prob < 40 else "➖")
                        })
                
                status.update(label="Scan Complete!", state="complete")
                st.dataframe(pd.DataFrame(results), use_container_width=True)
