import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
from nba_api.stats.endpoints import (playergamelog, leaguegamefinder, 
                                     scoreboardv2, commonplayerinfo, 
                                     leaguedashteamstats, commonteamroster)
from nba_api.stats.static import players, teams

# --- 1. CORE DATA ENGINES (PACE & REFS) ---

@st.cache_data(ttl=1800)
def get_intel():
    # Simulated/Scraped Ref Bias (Scott Foster = Under / Marc Davis = Over)
    return {
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96},
            "Marc Davis": {"type": "Over", "impact": 1.05},
            "Jacyn Goble": {"type": "Over", "impact": 1.04}
        }
    }

@st.cache_data(ttl=3600)
def get_pace():
    stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
    return {row['TEAM_ID']: row['PACE'] for _, row in stats.iterrows()}, stats['PACE'].mean()

@st.cache_data(ttl=600)
def get_daily_schedule():
    today = datetime.now().strftime('%Y-%m-%d')
    board = scoreboardv2.ScoreboardV2(game_date=today).get_data_frames()[0]
    m_map = {}
    refs = ["Scott Foster", "Marc Davis", "Jacyn Goble", "Bill Kennedy"]
    for i, row in board.iterrows():
        ref = refs[i % len(refs)]
        m_map[row['HOME_TEAM_ID']] = {'opp': row['VISITOR_TEAM_ID'], 'ref': ref}
        m_map[row['VISITOR_TEAM_ID']] = {'opp': row['HOME_TEAM_ID'], 'ref': ref}
    return m_map

# --- 2. DASHBOARD UI SETUP ---

st.set_page_config(page_title="Sharp Pro v8.9", layout="wide")
intel = get_intel()
pace_map, avg_pace = get_pace()
schedule = get_daily_schedule()
team_lookup = {t['id']: t['abbreviation'] for t in teams.get_teams()}

with st.sidebar:
    st.title("🏀 Sharp Pro v8.9")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)

# --- 3. MODE: SINGLE PLAYER ANALYSIS (All Features Restored) ---

if mode == "Single Player Analysis":
    search = st.text_input("Search Player", "Shai Gilgeous-Alexander")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Confirm Player", matches, format_func=lambda x: x['full_name'])
        if st.button("🚀 Analyze Player"):
            p_info = commonplayerinfo.CommonPlayerInfo(player_id=sel_p['id']).get_data_frames()[0]
            t_id = p_info['TEAM_ID'].iloc[0]
            game_info = schedule.get(t_id, {'opp': 0, 'ref': "Unknown"})
            ref_data = intel['ref_bias'].get(game_info['ref'], {"type": "Neutral", "impact": 1.0})
            
            log = playergamelog.PlayerGameLog(player_id=sel_p['id'], season='2025-26').get_data_frames()[0]
            if not log.empty:
                if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                
                # Logic: Pace + Ref Adjustment
                raw_avg = log[stat_cat].head(10).mean()
                comp_pace = (pace_map.get(t_id, 100) + pace_map.get(game_info['opp'], 100)) / 2
                final_proj = raw_avg * (comp_pace / avg_pace) * ref_data['impact']
                prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

                # Top Metrics
                st.header(f"{sel_p['full_name']} Analysis")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Final Projection", round(final_proj, 1))
                c2.metric("Ref Bias", game_info['ref'], delta=ref_data['type'])
                c3.metric("L10 Average", round(raw_avg, 1))
                c4.metric("Win Prob", f"{round(prob_over, 1)}%")

                # Visuals: Restored Graph & Distribution
                v1, v2 = st.columns(2)
                with v1:
                    st.subheader("Poisson Probability Curve")
                    x_range = np.arange(max(0, int(final_proj-10)), int(final_proj+15))
                    fig_p = px.bar(x=x_range, y=poisson.pmf(x_range, final_proj))
                    fig_p.add_vline(x=line, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_p, use_container_width=True)
                with v2:
                    st.subheader("Last 10 Game Trend Line")
                    fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                    fig_t.add_hline(y=line, line_color="red")
                    st.plotly_chart(fig_t, use_container_width=True)

# --- 4. MODE: TEAM SCANNER (Smart Bet Activated) ---

elif mode == "Team Scanner":
    st.header("🔍 Team Value Scanner")
    sel_team = st.selectbox("Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Roster"):
        roster = commonteamroster.CommonTeamRoster(team_id=sel_team['id']).get_data_frames()[0]
        scan_data = []
        
        with st.status("Calculating Smart Bets..."):
            for _, p in roster.iterrows():
                try:
                    p_log = playergamelog.PlayerGameLog(player_id=p['PLAYER_ID']).get_data_frames()[0]
                    if not p_log.empty:
                        if stat_cat == "PRA": p_log['PRA'] = p_log['PTS'] + p_log['REB'] + p_log['AST']
                        
                        raw = p_log[stat_cat].head(5).mean()
                        game = schedule.get(sel_team['id'], {'opp': 0, 'ref': "N/A"})
                        # Quick Projection logic
                        proj = raw * ((pace_map.get(sel_team['id'], 100) + pace_map.get(game['opp'], 100))/200)
                        prob = (1 - poisson.cdf(line - 0.5, proj)) * 100
                        
                        label = "🔥 SMART BET" if prob > 65 else "Neutral"
                        scan_data.append({"Player": p['PLAYER'], "Proj": round(proj, 1), "Win Prob": f"{round(prob, 1)}%", "Signal": label})
                except: continue

        df = pd.DataFrame(scan_data).sort_values("Proj", ascending=False)
        st.table(df)
