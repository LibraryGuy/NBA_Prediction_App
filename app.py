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

# --- 1. THE TRUTH ENGINE (INJURIES & REFS) ---

@st.cache_data(ttl=600)
def get_intel():
    # Primary Hard-Lock for January 2026 (Jokic, Embiid, etc.)
    intel = {
        "injuries": ["Nikola Jokic", "Joel Embiid", "Kevin Durant", "Ja Morant", 
                     "Cameron Johnson", "Christian Braun", "Tamar Bates", "Trae Young"],
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96},
            "Jacyn Goble": {"type": "Over", "impact": 1.06},
            "Marc Davis": {"type": "Over", "impact": 1.05},
            "Sean Corbin": {"type": "Over", "impact": 1.04}
        }
    }
    # Optional Scraper fallback
    try:
        url = "https://www.cbssports.com/nba/injuries/"
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        web_tables = pd.read_html(resp.text)
        for table in web_tables:
            if 'Status' in table.columns and 'Player' in table.columns:
                out = table[table['Status'].str.contains('Out|Sidelined', case=False, na=False)]
                intel["injuries"].extend(out['Player'].tolist())
    except: pass
    intel["injuries"] = list(set(intel["injuries"])) # De-duplicate
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

# --- 2. APP INITIALIZATION ---

st.set_page_config(page_title="Sharp Pro v9.3", layout="wide")
intel = get_intel()
pace_map, avg_pace = get_pace_data()
schedule = get_daily_matchups()

with st.sidebar:
    st.title("🏀 Sharp Pro v9.3")
    st.write(f"📅 **Date:** {datetime.now().strftime('%b %d, 2026')}")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Target Line", value=22.5, step=0.5)
    st.divider()
    if st.checkbox("Show Current Injury List"):
        st.write(intel["injuries"])

# --- 3. MODE: SINGLE PLAYER ANALYSIS ---

if mode == "Single Player Analysis":
    search = st.text_input("Player Search", "Shai Gilgeous-Alexander")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Confirm Player", matches, format_func=lambda x: x['full_name'])
        
        if sel_p['full_name'] in intel["injuries"]:
            st.error(f"⚠️ {sel_p['full_name']} is currently OUT. Analysis unavailable.")
        else:
            if st.button("🚀 Analyze Value"):
                p_info = commonplayerinfo.CommonPlayerInfo(player_id=sel_p['id']).get_data_frames()[0]
                t_id = p_info['TEAM_ID'].iloc[0]
                game = schedule.get(t_id, {'opp': 0, 'ref': "N/A"})
                ref_info = intel['ref_bias'].get(game['ref'], {"type": "Neutral", "impact": 1.0})
                
                log = playergamelog.PlayerGameLog(player_id=sel_p['id'], season='2025-26').get_data_frames()[0]
                if not log.empty:
                    if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                    
                    # Cascading Logic: Check for injured teammates
                    roster = commonteamroster.CommonTeamRoster(team_id=t_id).get_data_frames()[0]
                    missing_stars = [p for p in roster['PLAYER'] if p in intel["injuries"]]
                    usage_boost = 1.12 if len(missing_stars) > 0 else 1.0
                    
                    raw_avg = log[stat_cat].head(10).mean()
                    pace_f = ((pace_map.get(t_id, 100) + pace_map.get(game['opp'], 100)) / 2) / avg_pace
                    proj = raw_avg * pace_f * ref_info['impact'] * usage_boost
                    prob = (1 - poisson.cdf(line - 0.5, proj)) * 100

                    # UI DASHBOARD
                    st.header(f"{sel_p['full_name']} Prop Analysis")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Projection", round(proj, 1), delta=f"{round(usage_boost,2)}x Usage")
                    c2.metric("Referee", game['ref'], delta=ref_info['type'])
                    c3.metric("L10 Avg", round(raw_avg, 1))
                    c4.metric("Win Prob", f"{round(prob, 1)}%")

                    v1, v2 = st.columns(2)
                    with v1:
                        st.subheader("Poisson Distribution")
                        x = np.arange(max(0, int(proj-10)), int(proj+15))
                        fig = px.bar(x=x, y=poisson.pmf(x, proj), labels={'x':'Outcome', 'y':'Prob'})
                        fig.add_vline(x=line, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
                    with v2:
                        st.subheader("Last 10 Game Trend")
                        fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                        fig_t.add_hline(y=line, line_color="red")
                        st.plotly_chart(fig_t, use_container_width=True)

# --- 4. MODE: TEAM SCANNER ---

elif mode == "Team Scanner":
    st.header("🔍 Team Value Scanner")
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Run Full Roster Scan"):
        roster = commonteamroster.CommonTeamRoster(team_id=sel_team['id']).get_data_frames()[0]
        missing = [p for p in roster['PLAYER'] if p in intel["injuries"]]
        scan_results = []
        
        with st.status(f"Scanning {sel_team['full_name']}..."):
            for _, p in roster.iterrows():
                # RULE 1: If injured, skip completely
                if p['PLAYER'] in intel["injuries"]: continue
                
                try:
                    p_log = playergamelog.PlayerGameLog(player_id=p['PLAYER_ID']).get_data_frames()[0]
                    if not p_log.empty:
                        if stat_cat == "PRA": p_log['PRA'] = p_log['PTS'] + p_log['REB'] + p_log['AST']
                        
                        raw = p_log[stat_cat].head(5).mean()
                        game = schedule.get(sel_team['id'], {'opp': 0, 'ref': "N/A"})
                        
                        # Calculation Logic
                        cascade = 1.12 if len(missing) > 0 else 1.0
                        proj = raw * ((pace_map.get(sel_team['id'], 100) + pace_map.get(game['opp'], 100))/200) * cascade
                        prob = (1 - poisson.cdf(line - 0.5, proj)) * 100
                        
                        signal = "🔥 SMART BET" if prob > 65 else "Neutral"
                        scan_results.append({
                            "Player": p['PLAYER'], "Proj": round(proj, 1), 
                            "Usage Boost": f"{cascade}x", "Win Prob": f"{round(prob, 1)}%", "Signal": signal
                        })
                except: continue

        if scan_results:
            df = pd.DataFrame(scan_results).sort_values("Proj", ascending=False)
            st.write(f"**Injured/Out:** {', '.join(missing) if missing else 'None'}")
            st.dataframe(df.style.highlight_max(axis=0, subset=['Proj']), use_container_width=True)
        else:
            st.warning("No active players with data found.")
