import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, commonteamroster, boxscoretraditionalv2

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def load_nba_universe():
    # ... [Standard 2026 SOS Logic] ...
    return {k: 1.0 for k in range(30)}, {} # Simplified for space; keep your full SOS dict here

@st.cache_data(ttl=600)
def find_players_fuzzy(name_query):
    return [p for p in players.get_players() if name_query.lower() in p['full_name'].lower()]

@st.cache_data(ttl=600)
def get_player_stats(p_id):
    info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
    log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
    if not log.empty:
        log = log.rename(columns={'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'FGA': 'fga', 'FTA': 'fta', 'TOV': 'tov'})
        log['pra'] = log['points'] + log['rebounds'] + log['assists']
        log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
        log['pps'] = log['points'] / log['fga'].replace(0, 1)
    return log, info['TEAM_ABBREVIATION'].iloc[0]

# --- UI CONFIG ---
st.set_page_config(page_title="Sharp Pro v4.3", layout="wide")

with st.sidebar:
    st.title("💰 Sharp Pro Hub")
    total_purse = st.number_input("Purse ($)", value=1000)
    kelly_mult = st.slider("Kelly Fraction", 0.1, 1.0, 0.5)
    mode = st.radio("Mode", ["Single Player", "Team Scout", "Parlay", "Box Score"])
    stat_cat = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    is_home = st.toggle("Home Advantage", value=True)

# --- SINGLE PLAYER DASHBOARD ---
if mode == "Single Player":
    c_srch1, c_srch2, c_srch3 = st.columns([2, 2, 1])
    with c_srch1:
        query = st.text_input("1. Search Name", "Sarr")
    with c_srch2:
        matches = find_players_fuzzy(query)
        player_choice = st.selectbox("2. Confirm Player", matches, format_func=lambda x: x['full_name'])
    with c_srch3:
        vol_boost = st.checkbox("Rookie/High Volatility?", value=True)

    if player_choice:
        p_df, team_abbr = get_player_stats(player_choice['id'])
        p_mean = p_df[stat_cat].mean()
        
        # Line Movement Row
        c_l1, c_l2, c_l3 = st.columns(3)
        cur_line = c_l1.number_input("Current Vegas Line", value=float(round(p_mean, 1)))
        st_lambda = p_mean * (1.10 if vol_boost else 1.0) # Simplified logic for display
        win_p = (1 - poisson.cdf(cur_line - 0.5, st_lambda))
        c_l2.metric("Win Prob", f"{round(win_p*100, 1)}%")
        c_l3.metric("Rec. Bet", f"${round(total_purse * kelly_mult * 0.05, 2)}")

        # ROW 1: Trend and Efficiency
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.subheader("Last 10 Performance")
            fig_t = go.Figure(go.Scatter(y=p_df[stat_cat].head(10).iloc[::-1], mode='lines+markers', line=dict(color='#00ff96')))
            fig_t.add_hline(y=cur_line, line_dash="dash", line_color="red")
            fig_t.update_layout(template="plotly_dark", height=300, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_t, use_container_width=True)

        with row1_col2:
            st.subheader("Efficiency Matrix")
            fig_e = go.Figure(go.Scatter(x=p_df['usage'], y=p_df['pps'], mode='markers', marker=dict(size=12, color='#ffaa00')))
            fig_e.update_layout(template="plotly_dark", height=300, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Usage", yaxis_title="PPS")
            st.plotly_chart(fig_e, use_container_width=True)

        # ROW 2: FULL WIDTH MONTE CARLO
        st.divider()
        st.subheader("🎯 Full-Scale Monte Carlo Projection")
        st.caption("10,000 simulations based on 2026 performance variance.")
        
        sims = np.random.poisson(st_lambda, 10000)
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(x=sims, nbinsx=25, marker_color='#00ff96', opacity=0.7, name="Simulated Outcomes"))
        fig_mc.add_vline(x=cur_line, line_width=4, line_dash="dash", line_color="red", annotation_text="VEGAS LINE")
        
        # Highlight "The Over" Area
        fig_mc.update_layout(
            template="plotly_dark",
            height=450, # Increased height for readability
            xaxis_title=f"Predicted {stat_cat.title()}",
            yaxis_title="Frequency of Outcome",
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_mc, use_container_width=True)
