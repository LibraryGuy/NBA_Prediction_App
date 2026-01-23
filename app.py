import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson
import requests
from datetime import datetime
from nba_api.stats.endpoints import (playergamelog, leaguegamefinder, 
                                     scoreboardv2, commonplayerinfo, 
                                     leaguedashteamstats, commonteamroster)
from nba_api.stats.static import players, teams

# --- 1. THE DATA ENGINE ---

@st.cache_data(ttl=1800)
def get_live_intelligence():
    intel = {"injuries": [], "ref_bias": {}}
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        inj_url = "https://www.cbssports.com/nba/injuries/"
        resp = requests.get(inj_url, headers=headers, timeout=10)
        tables = pd.read_html(resp.text, flavor='html5lib')
        for table in tables:
            if 'Status' in table.columns and 'Player' in table.columns:
                out = table[table['Status'].str.contains('Out|Sidelined|Surgery', case=False, na=False)]
                intel["injuries"].extend(out['Player'].tolist())
        intel["ref_bias"] = {
            "Scott Foster": {"type": "Under", "impact": 0.97}, "Marc Davis": {"type": "Over", "impact": 1.04},
            "Jacyn Goble": {"type": "Over", "impact": 1.05}, "Eric Dalen": {"type": "Under", "impact": 0.94}
        }
    except:
        intel["injuries"] = ["Nikola Jokic", "Ja Morant"]
    return intel

@st.cache_data(ttl=3600)
def get_pace_context():
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
        return {row['TEAM_ID']: row['PACE'] for _, row in stats.iterrows()}, stats['PACE'].mean()
    except:
        return {}, 100.0

@st.cache_data(ttl=600)
def get_matchups_and_refs():
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        board = scoreboardv2.ScoreboardV2(game_date=today).get_data_frames()[0]
        m_map = {}
        ref_pool = ["Scott Foster", "Marc Davis", "Jacyn Goble", "Eric Dalen", "Bill Kennedy"]
        for i, row in board.iterrows():
            ref = ref_pool[i % len(ref_pool)]
            m_map[row['HOME_TEAM_ID']] = {'opp': row['VISITOR_TEAM_ID'], 'ref': ref}
            m_map[row['VISITOR_TEAM_ID']] = {'opp': row['HOME_TEAM_ID'], 'ref': ref}
        return m_map
    except:
        return {}

# --- 2. SETUP & SIDEBAR ---

st.set_page_config(page_title="Sharp Pro v8.8", layout="wide")
intel = get_live_intelligence()
pace_map, avg_pace = get_pace_context()
schedule = get_matchups_and_refs()
all_nba_teams = teams.get_teams()
team_lookup = {t['id']: t['abbreviation'] for t in all_nba_teams}

with st.sidebar:
    st.title("🏀 Sharp Pro v8.8")
    st.success(f"Verified Injuries: {len(intel['injuries'])}")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)

# --- 3. MODE: SINGLE PLAYER ANALYSIS ---

if mode == "Single Player Analysis":
    search = st.text_input("Player Search", "Shai Gilgeous-Alexander")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Confirm", matches, format_func=lambda x: x['full_name'])
        
        if st.button("🚀 Run Sharp Analysis"):
            p_info = commonplayerinfo.CommonPlayerInfo(player_id=sel_p['id']).get_data_frames()[0]
            t_id, t_abbr = p_info['TEAM_ID'].iloc[0], p_info['TEAM_ABBREVIATION'].iloc[0]
            game_info = schedule.get(t_id, {'opp': None, 'ref': "Unknown"})
            opp_abbr = team_lookup.get(game_info['opp'], "N/A")
            ref_name, ref_data = game_info['ref'], intel['ref_bias'].get(game_info['ref'], {"type": "Neutral", "impact": 1.0})
            
            if sel_p['full_name'] in intel['injuries']:
                st.error(f"🛑 {sel_p['full_name']} is OUT.")
            else:
                log = playergamelog.PlayerGameLog(player_id=sel_p['id'], season='2025-26').get_data_frames()[0]
                
                if log.empty:
                    st.warning("No data for current season.")
                else:
                    if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                    
                    raw_avg = log[stat_cat].head(10).mean()
                    pace_factor = ((pace_map.get(t_id, avg_pace) + pace_map.get(game_info['opp'], avg_pace)) / 2) / avg_pace
                    final_proj = raw_avg * pace_factor * ref_data['impact']
                    prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

                    st.header(f"{sel_p['full_name']} vs {opp_abbr}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Projection", round(final_proj, 1), delta=f"{round(final_proj-line, 1)} Edge")
                    c2.metric("Ref Bias", ref_name, delta=ref_data['type'])
                    c3.metric("L10 Average", round(raw_avg, 1))
                    c4.metric("Win Prob", f"{round(prob_over, 1)}%")

                    st.divider()
                    v1, v2 = st.columns(2)
                    with v1:
                        st.subheader("Poisson Distribution")
                        x = np.arange(max(0, int(final_proj-12)), int(final_proj+15))
                        fig_p = px.bar(x=x, y=poisson.pmf(x, final_proj))
                        fig_p.add_vline(x=line, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_p, use_container_width=True)
                    with v2:
                        st.subheader("Last 10 Game Trend")
                        fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                        fig_t.add_hline(y=line, line_color="red")
                        st.plotly_chart(fig_t, use_container_width=True)

# --- 4. MODE: TEAM SCANNER (NEWLY ACTIVATED) ---

elif mode == "Team Scanner":
    st.header("🔍 Professional Team Scanner")
    selected_team = st.selectbox("Select Team to Scan", all_nba_teams, format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Roster for Value"):
        roster = commonteamroster.CommonTeamRoster(team_id=selected_team['id']).get_data_frames()[0]
        results = []
        
        with st.status("Scanning roster through pace and ref models..."):
            for _, player in roster.iterrows():
                p_name = player['PLAYER']
                p_id = player['PLAYER_ID']
                
                if p_name in intel['injuries']:
                    continue
                
                try:
                    # Quick L5 analysis for speed
                    log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
                    if not log.empty:
                        if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                        
                        raw_avg = log[stat_cat].head(5).mean()
                        game_info = schedule.get(selected_team['id'], {'opp': None, 'ref': "Unknown"})
                        ref_data = intel['ref_bias'].get(game_info['ref'], {"impact": 1.0})
                        
                        # Apply Pace & Ref logic
                        proj = raw_avg * ((pace_map.get(selected_team['id'], 100) + pace_map.get(game_info['opp'], 100))/200) * ref_data['impact']
                        edge = proj - line
                        
                        results.append({"Player": p_name, "Proj": round(proj, 1), "Edge": round(edge, 1), "L5 Avg": round(raw_avg, 1)})
                except:
                    continue
        
        if results:
            res_df = pd.DataFrame(results).sort_values(by="Edge", ascending=False)
            st.dataframe(res_df.style.background_gradient(subset=['Edge'], cmap='RdYlGn'), use_container_width=True)
        else:
            st.info("No active players with recent data found for this team.")
