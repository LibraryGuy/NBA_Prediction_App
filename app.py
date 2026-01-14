import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, commonteamroster, boxscoretraditionalv2

# --- CORE ENGINE (Same fuzzy search and stats logic) ---
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

# --- UI CONFIG ---
st.set_page_config(page_title="Sharp Pro Hub v4.4", layout="wide")

with st.sidebar:
    st.title("🎯 Sharp Pro Hub")
    total_purse = st.number_input("Purse ($)", value=1000)
    kelly_mult = st.slider("Kelly Fraction", 0.1, 1.0, 0.5)
    st.divider()
    injury_pos = st.selectbox("Key Injury Impact", ["None", "PG Out", "Center Out", "Wing Out"])
    impact_map = {"None": 1.0, "PG Out": 1.12, "Center Out": 1.08, "Wing Out": 1.05}
    current_impact = impact_map[injury_pos]
    mode = st.radio("Mode", ["Single Player", "Team Scout", "Parlay", "Box Score Scraper"])
    stat_cat = st.selectbox("Stat", ["points", "rebounds", "assists", "three_pointers", "pra"])
    selected_opp = st.text_input("Opponent (SOS Logic)", "BOS")
    is_home = st.toggle("Home Game", value=True)

if mode == "Single Player":
    # --- 1. SEARCH & IDENTITY ---
    c_s1, c_s2, c_s3 = st.columns([2, 2, 1])
    with c_s1: query = st.text_input("Search Player", "Alexandre Sarr")
    with c_s2:
        matches = find_players_fuzzy(query)
        player_choice = st.selectbox("Confirm Identity", matches, format_func=lambda x: x['full_name'])
    with c_s3: vol_boost = st.checkbox("Volatility (Rookie) Mode", value=True)

    if player_choice:
        p_df, team_abbr, pos, height = get_player_stats(player_choice['id'])
        
        # --- 2. THE INFO HEADER (Restoring missing data) ---
        st.divider()
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.write(f"### {player_choice['full_name']}")
            st.caption(f"{team_abbr} | {pos} | {height}")
        with h2:
            last_val = p_df[stat_cat].iloc[0]
            st.metric("Last Game", last_val, delta=f"{round(last_val - p_df[stat_cat].mean(), 1)} vs Avg")
        with h3:
            st.metric("Season Average", round(p_df[stat_cat].mean(), 1))
        with h4:
            st.metric("Injury Modifier", f"+{int((current_impact-1)*100)}%")

        # --- 3. BETTING & LINE MOVEMENT ---
        st.write("#### Market Analysis")
        m1, m2, m3, m4 = st.columns(4)
        open_line = m1.number_input("Opening Line", value=float(round(p_df[stat_cat].mean(), 1)))
        curr_line = m2.number_input("Current Line", value=float(round(p_df[stat_cat].mean(), 1)))
        
        st_lambda = p_df[stat_cat].mean() * (1.10 if vol_boost else 1.0) * current_impact
        win_p = (1 - poisson.cdf(curr_line - 0.5, st_lambda))
        
        m3.metric("Win Prob", f"{round(win_p*100, 1)}%", delta=f"{round(curr_line - open_line, 1)} Line Move")
        
        # Kelly Stake
        dec_odds = 1.91
        k_f = ((dec_odds - 1) * win_p - (1 - win_p)) / (dec_odds - 1)
        m4.metric("Rec. Stake", f"${max(0, round(k_f * total_purse * kelly_mult, 2))}")

        # --- 4. VISUALIZATIONS ---
        tab1, tab2 = st.tabs(["Performance Trends", "Monte Carlo Probability"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Last 10 Performance**")
                fig_t = go.Figure(go.Scatter(y=p_df[stat_cat].head(10).iloc[::-1], mode='lines+markers', line=dict(color='#00ff96')))
                fig_t.add_hline(y=curr_line, line_dash="dash", line_color="red")
                fig_t.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig_t, use_container_width=True)
            with col_b:
                st.write("**Efficiency Matrix (Usage vs PPS)**")
                fig_e = go.Figure(go.Scatter(x=p_df['usage'], y=p_df['pps'], mode='markers', marker=dict(size=10, color='#ffaa00')))
                fig_e.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig_e, use_container_width=True)

        with tab2:
            st.write("**Full-Scale Outcome Distribution**")
            sims = np.random.poisson(st_lambda, 10000)
            fig_mc = go.Figure(go.Histogram(x=sims, nbinsx=30, marker_color='#00ff96', opacity=0.7))
            fig_mc.add_vline(x=curr_line, line_width=4, line_dash="dash", line_color="red", annotation_text="VEGAS")
            fig_mc.update_layout(template="plotly_dark", height=450, xaxis_title=stat_cat.upper())
            st.plotly_chart(fig_mc, use_container_width=True)

elif mode == "Box Score Scraper":
    # (Existing Box Score logic remains intact)
    st.write("Box Score Module Active")
