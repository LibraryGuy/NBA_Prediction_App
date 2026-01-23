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

# --- 1. THE INTEL ENGINE (FIXED INJURY LOGIC) ---

@st.cache_data(ttl=600)
def get_intel():
    # LAYER 1: THE HARD-LOCK (Manual safety net for Jan 2026)
    # If the scraper fails, these players are GUARANTEED to be filtered out.
    intel = {
        "injuries": [
            "Nikola Jokic", "Joel Embiid", "Kevin Durant", "Ja Morant", 
            "Cameron Johnson", "Christian Braun", "Tamar Bates", "Trae Young",
            "Jayson Tatum", "Jimmy Butler", "Anthony Davis", "Kyrie Irving"
        ],
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96},
            "Jacyn Goble": {"type": "Over", "impact": 1.06},
            "Marc Davis": {"type": "Over", "impact": 1.05}
        }
    }
    
    # LAYER 2: THE SCRAPER (Updated for 2026 CSS)
    try:
        url = "https://www.cbssports.com/nba/injuries/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        resp = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(resp.text)
        for table in tables:
            if 'Player' in table.columns and 'Status' in table.columns:
                # Filter for players explicitly listed as Out or Sidelined
                web_out = table[table['Status'].str.contains('Out|Sidelined|Surgery', case=False, na=False)]
                intel["injuries"].extend(web_out['Player'].tolist())
    except Exception:
        pass # Fallback to Hard-Lock list if site is down
        
    intel["injuries"] = list(set(intel["injuries"])) # Remove duplicates
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

# --- 2. UI INITIALIZATION ---

st.set_page_config(page_title="Sharp Pro v9.4", layout="wide")
intel = get_intel()
pace_map, avg_pace = get_pace_data()
schedule = get_daily_matchups()

with st.sidebar:
    st.title("🏀 Sharp Pro v9.4")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Target Line", value=22.5, step=0.5)
    st.divider()
    if st.checkbox("Show Global Injury List"):
        st.write(sorted(intel["injuries"]))

# --- 3. SINGLE PLAYER ANALYSIS ---

if mode == "Single Player Analysis":
    search = st.text_input("Search Player Name", "Shai Gilgeous-Alexander")
    all_p = players.get_players()
    matches = [p for p in all_p if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Select Result", matches, format_func=lambda x: x['full_name'])
        
        # Immediate Injury Check
        if sel_p['full_name'] in intel["injuries"]:
            st.error(f"🛑 {sel_p['full_name']} is currently OUT (Injury Report). No analysis available.")
        else:
            if st.button("📊 Run Deep Analysis"):
                p_info = commonplayerinfo.CommonPlayerInfo(player_id=sel_p['id']).get_data_frames()[0]
                t_id = p_info['TEAM_ID'].iloc[0]
                game = schedule.get(t_id, {'opp': 0, 'ref': "N/A"})
                ref_info = intel['ref_bias'].get(game['ref'], {"type": "Neutral", "impact": 1.0})
                
                log = playergamelog.PlayerGameLog(player_id=sel_p['id'], season='2025-26').get_data_frames()[0]
                if not log.empty:
                    if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                    
                    # CASCADING USAGE LOGIC
                    roster = commonteamroster.CommonTeamRoster(team_id=t_id).get_data_frames()[0]
                    # Fix for Line 55: Pre-calculate missing stars correctly
                    team_injury_list = [name for name in roster['PLAYER'] if name in intel["injuries"]]
                    usage_boost = 1.15 if len(team_injury_list) >= 1 else 1.0
                    
                    raw_avg = log[stat_cat].head(10).mean()
                    pace_f = ((pace_map.get(t_id, 100) + pace_map.get(game['opp'], 100)) / 2) / avg_pace
                    proj = raw_avg * pace_f * ref_info['impact'] * usage_boost
                    prob = (1 - poisson.cdf(line - 0.5, proj)) * 100

                    # UI DISPLAY
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Projection", round(proj, 1), delta=f"{usage_boost}x Usage")
                    c2.metric("Referee", game['ref'], delta=ref_info['type'])
                    c3.metric("L10 Average", round(raw_avg, 1))
                    c4.metric("Win Probability", f"{round(prob, 1)}%")

                    v1, v2 = st.columns(2)
                    with v1:
                        x = np.arange(max(0, int(proj-12)), int(proj+15))
                        fig = px.bar(x=x, y=poisson.pmf(x, proj), title="Poisson Curve")
                        fig.add_vline(x=line, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
                    with v2:
                        fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True, title="Recent Trend")
                        fig_t.add_hline(y=line, line_color="red")
                        st.plotly_chart(fig_t, use_container_width=True)

# --- 4. TEAM SCANNER ---

elif mode == "Team Scanner":
    st.header("🔍 Dynamic Team Scanner")
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Active Roster"):
        roster_df = commonteamroster.CommonTeamRoster(team_id=sel_team['id']).get_data_frames()[0]
        # Identify who is out for this specific team
        team_out = [p for p in roster_df['PLAYER'] if p in intel["injuries"]]
        scan_results = []
        
        progress = st.progress(0)
        for idx, p in roster_df.iterrows():
            # SKIP INJURED PLAYERS
            if p['PLAYER'] in intel["injuries"]: continue
            
            try:
                p_log = playergamelog.PlayerGameLog(player_id=p['PLAYER_ID']).get_data_frames()[0]
                if not p_log.empty:
                    if stat_cat == "PRA": p_log['PRA'] = p_log['PTS'] + p_log['REB'] + p_log['AST']
                    
                    raw = p_log[stat_cat].head(7).mean()
                    game = schedule.get(sel_team['id'], {'opp': 0, 'ref': "N/A"})
                    
                    # Apply Usage Cascade (Star is out = teammates get boost)
                    boost = 1.15 if len(team_out) > 0 else 1.0
                    pace_adj = ((pace_map.get(sel_team['id'], 100) + pace_map.get(game['opp'], 100))/200)
                    proj = raw * pace_adj * boost
                    prob = (1 - poisson.cdf(line - 0.5, proj)) * 100
                    
                    scan_results.append({
                        "Player": p['PLAYER'], "Proj": round(proj, 1), 
                        "Usage": f"{boost}x", "Win Prob": f"{round(prob, 1)}%",
                        "Status": "🔥 VALUE" if prob > 68 else "Neutral"
                    })
            except: continue
            progress.progress((idx + 1) / len(roster_df))

        if scan_results:
            st.subheader(f"Results for {sel_team['full_name']}")
            st.info(f"**Players Ruled OUT:** {', '.join(team_out) if team_out else 'None'}")
            st.dataframe(pd.DataFrame(scan_results).sort_values("Proj", ascending=False), use_container_width=True)
        else:
            st.error("Could not find active player data.")
