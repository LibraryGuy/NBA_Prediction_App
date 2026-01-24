import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import pytz
import time
from requests.exceptions import ReadTimeout
from nba_api.stats.endpoints import (playergamelog, leaguegamefinder, 
                                     scoreboardv2, commonplayerinfo, 
                                     leaguedashteamstats, commonteamroster)
from nba_api.stats.static import players, teams

# --- BROWSER EMULATION HEADERS ---
HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Connection': 'keep-alive',
}

# --- 1. UTILITY: API WRAPPER WITH RETRY ---
def safe_api_call(endpoint_class, **kwargs):
    """Retries API calls to bypass NBA.com throttling/timeouts."""
    for attempt in range(3):
        try:
            call = endpoint_class(**kwargs, headers=HEADERS, timeout=30)
            return call.get_data_frames()
        except (ReadTimeout, Exception):
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
    return None

# --- 2. CORE DATA ENGINES ---

@st.cache_data(ttl=1800)
def get_intel():
    return {
        "injuries": ["Nikola Jokic", "Kevin Durant", "Joel Embiid", "Ja Morant", "Giannis Antetokounmpo"],
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96},
            "Marc Davis": {"type": "Over", "impact": 1.05},
            "Jacyn Goble": {"type": "Over", "impact": 1.04},
            "Tony Brothers": {"type": "Under", "impact": 0.97},
            "James Williams": {"type": "Over", "impact": 1.03}
        }
    }

@st.cache_data(ttl=3600)
def get_daily_context():
    """Pulls schedule and OFFICIAL referee assignments from ScoreboardV2."""
    tz = pytz.timezone('US/Eastern')
    today = datetime.now(tz).strftime('%Y-%m-%d')
    
    frames = safe_api_call(scoreboardv2.ScoreboardV2, game_date=today)
    m_map = {}
    
    if frames:
        # frames[0] is the GameHeader (Matchups)
        # frames[2] is often the Officials (Referees)
        header = frames[0]
        officials = frames[2] if len(frames) > 2 else pd.DataFrame()
        
        for _, row in header.iterrows():
            g_id = row['GAME_ID']
            home_id = row['HOME_TEAM_ID']
            away_id = row['VISITOR_TEAM_ID']
            
            # Find refs assigned to this specific Game ID
            ref_name = "Unknown"
            if not officials.empty and 'GAME_ID' in officials.columns:
                game_refs = officials[officials['GAME_ID'] == g_id]
                if not game_refs.empty:
                    # We take the Lead Ref (Crew Chief) usually listed first
                    ref_name = game_refs.iloc[0]['OFFICIAL_NAME']
            
            m_map[home_id] = {'opp_id': away_id, 'ref': ref_name}
            m_map[away_id] = {'opp_id': home_id, 'ref': ref_name}
            
    return m_map

@st.cache_data(ttl=3600)
def get_pace_data():
    frames = safe_api_call(leaguedashteamstats.LeagueDashTeamStats, measure_type_detailed_defense='Advanced')
    if frames:
        df = frames[0]
        return {row['TEAM_ID']: row['PACE'] for _, row in df.iterrows()}, df['PACE'].mean()
    return {}, 100.0

# --- 3. DASHBOARD SETUP ---

st.set_page_config(page_title="Sharp Pro v10.7", layout="wide")
intel = get_intel()
pace_map, avg_pace = get_pace_data()
schedule = get_daily_context()
team_lookup = {t['id']: t['full_name'] for t in teams.get_teams()}

with st.sidebar:
    st.title("🏀 Sharp Pro v10.7")
    st.info("Status: Official Ref Sync Active ✅")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)

# --- 4. MODE: SINGLE PLAYER ANALYSIS ---

if mode == "Single Player Analysis":
    search = st.text_input("Search Player", "Peyton Watson")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Confirm Player", matches, format_func=lambda x: x['full_name'])
        if st.button("🚀 Run Full Analysis"):
            with st.spinner(f"Analyzing {sel_p['full_name']}..."):
                # A. Info & Context
                p_frames = safe_api_call(commonplayerinfo.CommonPlayerInfo, player_id=sel_p['id'])
                if not p_frames:
                    st.error("NBA.com connection failed. Please try again.")
                else:
                    t_id = p_frames[0]['TEAM_ID'].iloc[0]
                    game_context = schedule.get(t_id, {'opp_id': 0, 'ref': "TBD/Not Assigned"})
                    opp_name = team_lookup.get(game_context['opp_id'], "Opponent")
                    ref_data = intel['ref_bias'].get(game_context['ref'], {"type": "Neutral", "impact": 1.0})

                    # B. Fetch Game Logs
                    log_frames = safe_api_call(playergamelog.PlayerGameLog, player_id=sel_p['id'], season='2025-26')
                    h2h_frames = safe_api_call(leaguegamefinder.LeagueGameFinder, player_id_nullable=sel_p['id'], vs_team_id_nullable=game_context['opp_id'])
                    
                    if log_frames:
                        log = log_frames[0]
                        h2h = h2h_frames[0] if h2h_frames else pd.DataFrame()
                        
                        if stat_cat == "PRA": 
                            log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                            if not h2h.empty: h2h['PRA'] = h2h['PTS'] + h2h['REB'] + h2h['AST']
                        
                        raw_avg = log[stat_cat].head(10).mean()
                        comp_pace = (pace_map.get(t_id, 100) + pace_map.get(game_context['opp_id'], 100)) / 2
                        
                        # Injury Check
                        roster_frames = safe_api_call(commonteamroster.CommonTeamRoster, team_id=t_id)
                        usage_boost = 1.0
                        if roster_frames:
                            injured_here = [p for p in roster_frames[0]['PLAYER'] if p in intel['injuries']]
                            if injured_here: usage_boost = 1.12

                        # Final Projections
                        final_proj = raw_avg * (comp_pace / avg_pace) * ref_data['impact'] * usage_boost
                        prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

                        # --- UI RENDER ---
                        st.header(f"{sel_p['full_name']} vs {opp_name}")
                        st.caption(f"Ref: {game_context['ref']} ({ref_data['type']})")
                        
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Final Projection", round(final_proj, 1), delta=f"{round(usage_boost,2)}x Usage")
                        c2.metric("Ref Impact", game_context['ref'], delta=f"{int((ref_data['impact']-1)*100)}%")
                        c3.metric("L10 Average", round(raw_avg, 1))
                        c4.metric("Win Prob (Over)", f"{round(prob_over, 1)}%")

                        st.divider()
                        st.subheader("🎯 Sharp Pro Betting Blueprint")
                        b1, b2, b3 = st.columns(3)
                        direction = "OVER" if prob_over > 55 else "UNDER"
                        with b1: st.info(f"**Primary:** {stat_cat} {direction} {line}")
                        with b2: st.success(f"**Safety:** {stat_cat} {direction} {line - 4 if direction == 'OVER' else line + 4}")
                        with b3: st.warning(f"**Ladder:** {stat_cat} {direction} {line + 5 if direction == 'OVER' else line - 5}")

                        st.divider()
                        st.subheader(f"📅 Last 5 Games vs {opp_name}")
                        if not h2h.empty:
                            h2h_display = h2h[['GAME_DATE', 'MATCHUP', 'WL', stat_cat]].head(5)
                            h2h_display['Result'] = h2h_display[stat_cat].apply(lambda x: "✅ Over" if x > line else "❌ Under")
                            st.table(h2h_display)
                        else: st.info("No historical H2H data found.")

                        v1, v2 = st.columns(2)
                        with v1:
                            st.subheader("Poisson Probability")
                            x_range = np.arange(max(0, int(final_proj-12)), int(final_proj+15))
                            fig_p = px.bar(x=x_range, y=poisson.pmf(x_range, final_proj))
                            fig_p.add_vline(x=line, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_p, use_container_width=True)
                        with v2:
                            st.subheader("Last 10 Game Trend")
                            fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                            fig_t.add_hline(y=line, line_color="red", line_dash="dash")
                            st.plotly_chart(fig_t, use_container_width=True)

# --- 5. MODE: TEAM SCANNER ---
elif mode == "Team Scanner":
    st.header("🔍 Value Scanner")
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Roster"):
        roster_frames = safe_api_call(commonteamroster.CommonTeamRoster, team_id=sel_team['id'])
        if roster_frames:
            scan_data = []
            with st.status("Processing..."):
                for _, p in roster_frames[0].iterrows():
                    p_logs = safe_api_call(playergamelog.PlayerGameLog, player_id=p['PLAYER_ID'])
                    if p_logs:
                        log = p_logs[0]
                        if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                        avg = log[stat_cat].head(5).mean()
                        # Simplified projection for scanner
                        prob = (1 - poisson.cdf(line - 0.5, avg)) * 100
                        scan_data.append({"Player": p['PLAYER'], "L5 Avg": round(avg,1), "Prob": f"{round(prob,1)}%"})
            st.table(pd.DataFrame(scan_data))
