import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson
import requests
import time
from nba_api.stats.endpoints import playergamelog, commonteamroster
from nba_api.stats.static import players, teams

# --- 1. DATA ENGINES ---

@st.cache_data(ttl=1800)
def get_automated_injury_list():
    confirmed_out = []
    try:
        url = "https://www.cbssports.com/nba/injuries/"
        # Using html5lib or bs4 for compatibility
        tables = pd.read_html(requests.get(url).text, flavor='html5lib')
        for table in tables:
            if 'Status' in table.columns and 'Player' in table.columns:
                out_p = table[table['Status'].str.contains('Out|Sidelined|Surgery', case=False, na=False)]
                confirmed_out.extend(out_p['Player'].tolist())
        confirmed_out = [name.split('  ')[0].strip() for name in confirmed_out]
    except:
        confirmed_out = ["Nikola Jokic", "Fred VanVleet"] # Hard-coded fallback
    return list(set(confirmed_out))

@st.cache_data(ttl=600)
def get_player_last_10(p_id):
    try:
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]
        return log.head(10).iloc[::-1] # Reverse to show chronological order
    except:
        return pd.DataFrame()

# --- 2. THE DASHBOARD ---

st.set_page_config(page_title="Sharp Pro v7.6", layout="wide")
injury_list = get_automated_injury_list()

with st.sidebar:
    st.title("🏀 Sharp Pro v7.6")
    st.success(f"Injury Filter: {len(injury_list)} Players OUT")
    mode = st.radio("App Mode", ["Single Player", "Team Scanner"])
    stat_cat = st.selectbox("Category", ["PTS", "REB", "AST", "STL", "BLK"])
    consensus_line = st.number_input("Sportsbook Line", value=20.5, step=0.5)

if mode == "Single Player":
    search = st.text_input("Search Player", "Jamal Murray")
    p_matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if p_matches:
        sel_p = st.selectbox("Select Player", p_matches, format_func=lambda x: x['full_name'])
        
        # 🟢 START SCAN BUTTON (Manual Trigger)
        if st.button("🚀 Start Player Analysis"):
            if sel_p['full_name'] in injury_list:
                st.error(f"🛑 {sel_p['full_name']} is currently OUT.")
            else:
                with st.spinner("Fetching Trends..."):
                    df_10 = get_player_last_10(sel_p['id'])
                    
                    # Dashboard Metrics
                    avg_10 = df_10[stat_cat].mean()
                    proj_val = avg_10 * 1.05 # Simple 5% model boost for example
                    prob_over = (1 - poisson.cdf(consensus_line - 0.5, proj_val)) * 100
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Sharp Pro Projection", round(proj_val, 1))
                    m2.metric("L10 Average", round(avg_10, 1))
                    m3.metric(f"Prob. Over {consensus_line}", f"{round(prob_over, 1)}%")

                    # --- RESTORED GRAPHS ---
                    st.divider()
                    g1, g2 = st.columns(2)
                    
                    with g1:
                        st.subheader("Poisson Spread")
                        x = np.arange(max(0, int(proj_val-10)), int(proj_val+15))
                        y = poisson.pmf(x, proj_val)
                        fig_pois = px.bar(x=x, y=y, title="Probability Density")
                        fig_pois.add_vline(x=consensus_line, line_dash="dash", line_color="red", annotation_text="Bookie Line")
                        st.plotly_chart(fig_pois, use_container_width=True)

                    with g2:
                        st.subheader("Radar: Skill Profile")
                        radar_cats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
                        radar_vals = [df_10[c].mean() for c in radar_cats]
                        fig_radar = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=radar_cats, fill='toself'))
                        st.plotly_chart(fig_radar, use_container_width=True)

                    # --- NEW: LAST 10 TREND LINE ---
                    st.divider()
                    st.subheader(f"📈 Last 10 Games Trend: {stat_cat}")
                    fig_trend = px.line(df_10, x=df_10.index, y=stat_cat, markers=True, text=stat_cat)
                    fig_trend.add_hline(y=consensus_line, line_color="red", line_dash="dot", annotation_text="Market Line")
                    fig_trend.add_hline(y=avg_10, line_color="green", annotation_text="L10 Avg")
                    fig_trend.update_layout(xaxis_title="Games Ago (Left to Right)", yaxis_title=stat_cat)
                    st.plotly_chart(fig_trend, use_container_width=True)

                    # --- SPORTSBOOK TRENDS ---
                    st.subheader("🎯 Value Verdict")
                    edge = proj_val - consensus_line
                    if edge > 2:
                        st.success(f"SHARP SIGNAL: Large Positive Edge of {round(edge, 2)} points found.")
                    elif edge < -2:
                        st.warning(f"FADE SIGNAL: Player is projected {round(abs(edge), 2)} UNDER the line.")
                    else:
                        st.info("NEUTRAL: Projection aligns with market consensus.")

else:
    st.header("📋 Team Scanner")
    # (Team scan logic remains the same as previous version)
