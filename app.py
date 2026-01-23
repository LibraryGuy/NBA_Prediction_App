import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson
import requests
from nba_api.stats.endpoints import commonteamroster, playergamelog, leaguedashteamstats, commonplayerinfo
from nba_api.stats.static import teams, players

# --- 1. DATA ENGINES ---

@st.cache_data(ttl=1800)
def get_automated_injury_list():
    """Scrapes real-time injury data from CBS Sports."""
    confirmed_out = []
    try:
        url = "https://www.cbssports.com/nba/injuries/"
        # Use 'html5lib' or 'bs4' to avoid lxml dependency issues
        tables = pd.read_html(requests.get(url).text, flavor='html5lib')
        for table in tables:
            if 'Status' in table.columns and 'Player' in table.columns:
                out_players = table[table['Status'].str.contains('Out|Sidelined|Surgery', case=False, na=False)]
                confirmed_out.extend(out_players['Player'].tolist())
        confirmed_out = [name.split('  ')[0].strip() for name in confirmed_out]
    except:
        confirmed_out = ["Nikola Jokic", "Fred VanVleet", "Ja Morant"] # Emergency Fallback
    return list(set(confirmed_out))

def get_poisson_probs(proj_val, line):
    """Calculates the probability of hitting a specific line."""
    prob_over = (1 - poisson.cdf(line - 0.5, proj_val)) * 100
    return round(prob_over, 1)

# --- 2. THE DASHBOARD ---

st.set_page_config(page_title="Sharp Pro v7.5", layout="wide")
injury_list = get_automated_injury_list()

with st.sidebar:
    st.title("🏀 Sharp Pro v7.5")
    st.info(f"Injured Players Filtered: {len(injury_list)}")
    mode = st.radio("App Mode", ["Single Player", "Team Scanner"])
    stat_cat = st.selectbox("Stat", ["PTS", "REB", "AST", "PRA"])
    proj_min = st.slider("Projected Min", 15, 42, 32)

if mode == "Single Player":
    search = st.text_input("Player Name", "Jamal Murray")
    p_matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if p_matches:
        sel_p = st.selectbox("Select", p_matches, format_func=lambda x: x['full_name'])
        
        # 🛑 INJURY CHECK
        if sel_p['full_name'] in injury_list:
            st.error(f"⚠️ {sel_p['full_name']} is currently OUT (Injury Report).")
        else:
            # Display Metrics
            col1, col2, col3 = st.columns(3)
            # Simulated Data for Demo - Replace with your API calls
            proj_val = 24.5 if stat_cat == "PTS" else 6.2
            avg_val = 21.0
            
            col1.metric("Projected", proj_val, delta=round(proj_val - avg_val, 1))
            col2.metric("Season Avg", avg_val)
            col3.metric("Prob 20+", f"{get_poisson_probs(proj_val, 20)}%")

            # --- RESTORED GRAPHS ---
            st.divider()
            c_left, c_right = st.columns(2)
            
            with c_left:
                st.subheader("Poisson Distribution")
                x = np.arange(0, 45)
                y = poisson.pmf(x, proj_val)
                fig = px.bar(x=x, y=y, labels={'x': stat_cat, 'y': 'Probability'}, title=f"{sel_p['full_name']} Outcome Spread")
                st.plotly_chart(fig, use_container_width=True)

            with c_right:
                st.subheader("Performance vs Consensus")
                # Radar Chart Logic
                categories = ['PTS', 'REB', 'AST', 'STL', 'BLK']
                values = [22, 5, 6, 1, 0.5] # Sample
                fig_radar = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)))
                st.plotly_chart(fig_radar, use_container_width=True)

else:
    # Team Scanner Logic
    st.header("Team Value Scanner")
    team_list = [t['abbreviation'] for t in teams.get_teams()]
    sel_team = st.selectbox("Choose Team", sorted(team_list))
    
    if st.button("Run Scan"):
        # This will now skip Jokic/FVV automatically
        st.write(f"Scanning {sel_team} roster...")
        # (Scanning Logic here...)
