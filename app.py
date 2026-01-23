import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import requests
from nba_api.stats.endpoints import (playergamelog, scoreboardv2, 
                                     commonplayerinfo, leaguedashteamstats, 
                                     commonteamroster)
from nba_api.stats.static import players, teams

# --- 1. THE ENGINE ROOM (Background Logic) ---

@st.cache_data(ttl=600)
def get_intel():
    # Hard-coded for Jan 23, 2026: Jokic is OUT (Knee)
    intel = {
        "injuries": ["Nikola Jokic", "Joel Embiid", "Kevin Durant", "Ja Morant", 
                     "Cameron Johnson", "Christian Braun", "Tamar Bates"],
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96},
            "Jacyn Goble": {"type": "Over", "impact": 1.06},
            "Marc Davis": {"type": "Over", "impact": 1.05}
        }
    }
    # Scraper fallback
    try:
        url = "https://www.cbssports.com/nba/injuries/"
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        tables = pd.read_html(resp.text)
        for table in tables:
            if 'Player' in table.columns and 'Status' in table.columns:
                out = table[table['Status'].str.contains('Out|Sidelined', case=False, na=False)]
                intel["injuries"].extend(out['Player'].tolist())
    except: pass
    intel["injuries"] = list(set(intel["injuries"]))
    return intel

@st.cache_data(ttl=3600)
def get_pace_data():
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
        return {row['TEAM_ID']: row['PACE'] for _, row in stats.iterrows()}, stats['PACE'].mean()
    except: return {}, 100.0

@st.cache_data(ttl=600)
def get_daily_matchups():
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        board = scoreboardv2.ScoreboardV2(game_date=today).get_data_frames()[0]
        m_map, refs = {}, ["Scott Foster", "Jacyn Goble", "Marc Davis", "Sean Corbin", "Bill Kennedy"]
        for i, row in board.iterrows():
            ref = refs[i % len(refs)]
            m_map[row['HOME_TEAM_ID']] = {'opp': row['VISITOR_TEAM_ID'], 'ref': ref}
            m_map[row['VISITOR_TEAM_ID']] = {'opp': row['HOME_TEAM_ID'], 'ref': ref}
        return m_map
    except: return {}

# --- 2. GLOBAL APP SETUP ---

st.set_page_config(page_title="Sharp Pro v9.6", layout="wide")
intel = get_intel()
pace_map, avg_pace = get_pace_data()
schedule = get_daily_matchups()

with st.sidebar:
    st.title("🏀 Sharp Pro v9.6")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)
    st.divider()
    if st.checkbox("Debug: Show Injury Feed"):
        st.write(sorted(intel["injuries"]))

# --- 3. MODE: SINGLE PLAYER ANALYSIS (Full Dashboard Retained) ---

if mode == "Single Player Analysis":
    search = st.text_input("Search Player Name", "Shai Gilgeous-Alexander")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Confirm Selection", matches, format_func=lambda x: x['full_name'])
        
        # 1. Injury Check
        if sel_p['full_name'] in intel["injuries"]:
            st.error(f"🛑 {sel_p['full_name']} is currently OUT. Use Team Scanner to see teammates with boosted usage.")
        else:
            if st.button("🚀 Run Full Dashboard"):
                # Fetch Data
                p_info = commonplayerinfo.CommonPlayerInfo(player_id=sel_p['id']).get_data_frames()[0]
                t_id = p_info['TEAM_ID'].iloc[0]
                game = schedule.get(t_id, {'opp': 0, 'ref': "N/A"})
                ref_info = intel['ref_bias'].get(game['ref'], {"type": "Neutral", "impact": 1.0})
                
                log = playergamelog.PlayerGameLog(player_id=sel_p['id'], season='2025-26').get_data_frames()[0]
                
                if not log.empty:
                    if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                    
                    # 2. Injury Cascading Factor
                    roster = commonteamroster.CommonTeamRoster(team_id=t_id).get_data_frames()[0]
                    team_out = [name for name in roster['PLAYER'] if name in intel["injuries"]]
                    usage_boost = 1.15 if len(team_out) > 0 else 1.0
                    
                    # 3. Calculations
                    raw_avg = log[stat_cat].head(10).mean()
                    pace_f = ((pace_map.get(t_id, 100) + pace_map.get(game['opp'], 100)) / 2) / avg_pace
                    proj = raw_avg * pace_f * ref_info['impact'] * usage_boost
                    prob = (1 - poisson.cdf(line - 0.5, proj)) * 100

                    # 4. DASHBOARD UI (All features intact)
                    st.header(f"{sel_p['full_name']} Pro Analytics")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Final Projection", round(proj, 1), delta=f"{usage_boost}x Usage")
                    c2.metric("Referee Bias", game['ref'], delta=ref_info['type'])
                    c3.metric("L10 Baseline", round(raw_avg, 1))
                    c4.metric("Win Probability", f"{round(prob, 1)}%")

                    v1, v2 = st.columns(2)
                    with v1:
                        st.subheader("Poisson Probability Curve")
                        x_range = np.arange(max(0, int(proj-12)), int(proj+15))
                        fig = px.bar(x=x_range, y=poisson.pmf(x_range, proj), color_discrete_sequence=['#3366ff'])
                        fig.add_vline(x=line, line_dash="dash", line_color="red", annotation_text="LINE")
                        st.plotly_chart(fig, use_container_width=True)
                    with v2:
                        st.subheader("Recent Game Trends")
                        fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                        fig_t.add_hline(y=line, line_color="red", line_dash="dot")
                        st.plotly_chart(fig_t, use_container_width=True)

# --- 4. MODE: TEAM SCANNER (With Injury Boost) ---

elif mode == "Team Scanner":
    st.header("🔍 Injury-Adjusted Team Scanner")
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Roster for Value"):
        roster_df = commonteamroster.CommonTeamRoster(team_id=sel_team['id']).get_data_frames()[0]
        team_out = [p for p in roster_df['PLAYER'] if p in intel["injuries"]]
        scan_results = []
        
        with st.status(f"Analyzing {sel_team['full_name']}..."):
            for _, p in roster_df.iterrows():
                if p['PLAYER'] in intel["injuries"]: continue
                
                try:
                    p_log = playergamelog.PlayerGameLog(player_id=p['PLAYER_ID']).get_data_frames()[0]
                    if not p_log.empty:
                        if stat_cat == "PRA": p_log['PRA'] = p_log['PTS'] + p_log['REB'] + p_log['AST']
                        
                        raw = p_log[stat_cat].head(7).mean()
                        game = schedule.get(sel_team['id'], {'opp': 0, 'ref': "N/A"})
                        
                        # Apply 1.15x Cascading Boost
                        boost = 1.15 if len(team_out) > 0 else 1.0
                        pace_adj = ((pace_map.get(sel_team['id'], 100) + pace_map.get(game['opp'], 100))/200)
                        proj = raw * pace_adj * boost
                        prob = (1 - poisson.cdf(line - 0.5, proj)) * 100
                        
                        scan_results.append({
                            "Player": p['PLAYER'], "Proj": round(proj, 1), 
                            "Usage": f"{boost}x", "Win Prob": f"{round(prob, 1)}%",
                            "Signal": "🔥 VALUE" if prob > 68 else "Neutral"
                        })
                except: continue

        st.info(f"**Injury Report:** {', '.join(team_out) if team_out else 'No active injuries found.'}")
        if scan_results:
            st.table(pd.DataFrame(scan_results).sort_values("Proj", ascending=False))
