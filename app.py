import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, commonteamroster, boxscoretraditionalv2

# --- APP CONFIG ---
st.set_page_config(page_title="Sharp Pro Hub v4.5", layout="wide")

# --- DATA ENGINE ---
@st.cache_data(ttl=600)
def find_players_fuzzy(name_query):
    return [p for p in players.get_players() if name_query.lower() in p['full_name'].lower()]

@st.cache_data(ttl=600)
def get_player_stats(p_id):
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if not log.empty:
            log = log.rename(columns={'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'FGA': 'fga', 'FTA': 'fta', 'TOV': 'tov'})
            log['pra'] = log['points'] + log['rebounds'] + log['assists']
            log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
            log['pps'] = log['points'] / log['fga'].replace(0, 1)
        return log, info['TEAM_ABBREVIATION'].iloc[0], info['POSITION'].iloc[0], info['HEIGHT'].iloc[0]
    except: return pd.DataFrame(), None, None, None

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🛡️ Pro Hub v4.5")
    total_purse = st.number_input("Purse ($)", value=1000)
    kelly_mult = st.slider("Kelly Fraction", 0.1, 1.0, 0.5)
    st.divider()
    injury_pos = st.selectbox("Key Injury Impact", ["None", "PG Out", "Center Out", "Wing Out"])
    impact_map = {"None": 1.0, "PG Out": 1.12, "Center Out": 1.08, "Wing Out": 1.05}
    current_impact = impact_map[injury_pos]
    mode = st.radio("Mode", ["Single Player", "Team Scout", "Parlay", "Box Score Scraper"])
    stat_cat = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    is_home = st.toggle("Home Game", value=True)

# --- DASHBOARD LOGIC ---
if mode == "Single Player":
    # 1. SEARCH ROW
    c_s1, c_s2, c_s3 = st.columns([2, 2, 1])
    with c_s1: query = st.text_input("1. Search Name", "Alexandre Sarr")
    with c_s2:
        matches = find_players_fuzzy(query)
        player_choice = st.selectbox("2. Confirm Identity", matches, format_func=lambda x: x['full_name'])
    with c_s3: vol_boost = st.checkbox("Volatility (Rookie) Mode", value=True)

    if player_choice:
        p_df, team_abbr, pos, height = get_player_stats(player_choice['id'])
        p_mean = p_df[stat_cat].mean()
        
        # 2. TOP METRICS BAR (Fixed Missing Info)
        st.divider()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.write(f"**{player_choice['full_name']}**\n{team_abbr} | {pos}")
        m2.metric("Season Avg", round(p_mean, 1))
        m3.metric("Last Game", p_df[stat_cat].iloc[0])
        m4.metric("Height", height)
        m5.metric("Injury Boost", f"{int((current_impact-1)*100)}%")

        # 3. MARKET & BETTING ROW
        st.markdown("### 📊 Market Analysis & Betting Strategy")
        b1, b2, b3, b4 = st.columns(4)
        open_line = b1.number_input("Opening Line", value=float(round(p_mean, 1)))
        curr_line = b2.number_input("Current Vegas Line", value=float(round(p_mean, 1)))
        
        # Projection Logic
        st_lambda = p_mean * (1.10 if vol_boost else 1.0) * current_impact
        win_p = (1 - poisson.cdf(curr_line - 0.5, st_lambda))
        
        b3.metric("Win Prob", f"{round(win_p*100, 1)}%", delta=f"{round(curr_line-open_line, 1)} Move")
        
        # Kelly Stake
        dec_odds = 1.91
        k_f = ((dec_odds - 1) * win_p - (1 - win_p)) / (dec_odds - 1)
        stake = max(0, round(k_f * total_purse * kelly_mult, 2))
        b4.metric("Rec. Stake", f"${stake}", help="Based on Adjusted Kelly Criterion")

        # 4. PERFORMANCE CHARTS (Middle Row)
        st.divider()
        col_left, col_right = st.columns(2)
        with col_left:
            st.write("**Recent Performance Trend (Last 10)**")
            fig_t = go.Figure(go.Scatter(y=p_df[stat_cat].head(10).iloc[::-1], mode='lines+markers', line=dict(color='#00ff96', width=3)))
            fig_t.add_hline(y=curr_line, line_dash="dash", line_color="red", annotation_text="Market")
            fig_t.update_layout(template="plotly_dark", height=280, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_t, use_container_width=True)
            
        with col_right:
            st.write("**Efficiency Matrix (Usage vs PPS)**")
            fig_e = go.Figure(go.Scatter(x=p_df['usage'], y=p_df['pps'], mode='markers', marker=dict(size=12, color='#ffaa00', opacity=0.7)))
            fig_e.update_layout(template="plotly_dark", height=280, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Usage Volume", yaxis_title="Points Per Shot")
            st.plotly_chart(fig_e, use_container_width=True)

        # 5. FULL-WIDTH MONTE CARLO (Bottom Row)
        st.divider()
        st.write("### 🎯 Outcome Probability Distribution")
        sims = np.random.poisson(st_lambda, 10000)
        fig_mc = go.Figure(go.Histogram(x=sims, nbinsx=35, marker_color='#00ff96', opacity=0.6))
        fig_mc.add_vline(x=curr_line, line_width=5, line_dash="dash", line_color="red", annotation_text="VEGAS LINE")
        fig_mc.update_layout(
            template="plotly_dark", 
            height=450, 
            xaxis_title=f"Projected {stat_cat.upper()}", 
            yaxis_title="Frequency",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_mc, use_container_width=True)

elif mode == "Box Score Scraper":
    st.header("📋 Last Game Detailed Box Score")
    # ... (Rest of modules remain available in the background) ...
