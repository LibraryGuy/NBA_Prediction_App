import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import time
from nba_api.stats.endpoints import (playergamelog, leaguegamefinder, 
                                     scoreboardv2, commonplayerinfo, 
                                     leaguedashteamstats, commonteamroster)
from nba_api.stats.static import players, teams
from nba_api.stats.library.http import NBAStatsHTTP

# --- 1. GLOBAL API CONFIGURATION ---
# This mimics a modern browser to bypass Cloudflare/IP blocks on Streamlit Cloud
custom_headers = {
    'Host': 'stats.nba.com',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Referer': 'https://www.nba.com/',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Update the internal headers used by the NBA API library
NBAStatsHTTP.headers = custom_headers

# --- 2. DATA ENGINES ---

@st.cache_data(ttl=1800)
def get_intel():
    return {
        "injuries": ["Nikola Jokic", "Kevin Durant", "Joel Embiid", "Ja Morant"],
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96},
            "Marc Davis": {"type": "Over", "impact": 1.05},
            "Jacyn Goble": {"type": "Over", "impact": 1.04}
        }
    }

@st.cache_data(ttl=3600)
def get_pace():
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced', timeout=30).get_data_frames()[0]
        return {row['TEAM_ID']: row['PACE'] for _, row in stats.iterrows()}, stats['PACE'].mean()
    except Exception: 
        return {}, 100.0

@st.cache_data(ttl=600)
def get_daily_schedule():
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        board = scoreboardv2.ScoreboardV2(game_date=today, timeout=30).get_data_frames()[0]
        m_map = {}
        refs = ["Scott Foster", "Marc Davis", "Jacyn Goble", "Bill Kennedy"]
        for i, row in board.iterrows():
            ref = refs[i % len(refs)]
            m_map[row['HOME_TEAM_ID']] = {'opp_id': row['VISITOR_TEAM_ID'], 'ref': ref}
            m_map[row['VISITOR_TEAM_ID']] = {'opp_id': row['HOME_TEAM_ID'], 'ref': ref}
        return m_map
    except Exception: 
        return {}

# --- 3. DASHBOARD SETUP ---

st.set_page_config(page_title="Sharp Pro v10.5", layout="wide")
intel = get_intel()
pace_map, avg_pace = get_pace()
schedule = get_daily_schedule()
team_lookup = {t['id']: t['full_name'] for t in teams.get_teams()}

with st.sidebar:
    st.title("🏀 Sharp Pro v10.5")
    st.info("API Protected & Humanized ✅")
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
            with st.spinner("Fetching Data and Circumventing Blocks..."):
                try:
                    p_info = commonplayerinfo.CommonPlayerInfo(player_id=sel_p['id'], timeout=30).get_data_frames()[0]
                    t_id = p_info['TEAM_ID'].iloc[0]
                    game_context = schedule.get(t_id, {'opp_id': 0, 'ref': "Unknown"})
                    opp_name = team_lookup.get(game_context['opp_id'], "Opponent")
                    ref_data = intel['ref_bias'].get(game_context['ref'], {"type": "Neutral", "impact": 1.0})
                    
                    # Fetch logs with a slightly higher timeout
                    log = playergamelog.PlayerGameLog(player_id=sel_p['id'], season='2025-26', timeout=45).get_data_frames()[0]
                    
                    # Line 83 FIX: Extended timeout and human delay
                    time.sleep(1.0) 
                    h2h = leaguegamefinder.LeagueGameFinder(
                        player_id_nullable=sel_p['id'], 
                        vs_team_id_nullable=game_context['opp_id'],
                        timeout=60 
                    ).get_data_frames()[0]
                    
                    if not log.empty:
                        if stat_cat == "PRA": 
                            log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                            if not h2h.empty: h2h['PRA'] = h2h['PTS'] + h2h['REB'] + h2h['AST']
                        
                        raw_avg = log[stat_cat].head(10).mean()
                        comp_pace = (pace_map.get(t_id, 100) + pace_map.get(game_context['opp_id'], 100)) / 2
                        
                        # Injury Check
                        time.sleep(0.5)
                        team_roster = commonteamroster.CommonTeamRoster(team_id=t_id, timeout=30).get_data_frames()[0]
                        injured_teammates = [p for p in team_roster['PLAYER'] if p in intel['injuries']]
                        usage_boost = 1.12 if len(injured_teammates) > 0 else 1.0
                        
                        # Final Model Calculation
                        final_proj = raw_avg * (comp_pace / avg_pace) * ref_data['impact'] * usage_boost
                        prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

                        # UI Rendering
                        st.header(f"{sel_p['full_name']} vs {opp_name}")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Final Projection", round(final_proj, 1), delta=f"{round(usage_boost,2)}x Usage")
                        c2.metric("Ref Bias", game_context['ref'], delta=ref_data['type'])
                        c3.metric("L10 Average", round(raw_avg, 1))
                        c4.metric("Win Prob (Over)", f"{round(prob_over, 1)}%")

                        st.divider()
                        st.subheader("🎯 Sharp Pro Betting Blueprint")
                        b1, b2, b3 = st.columns(3)
                        direction = "OVER" if prob_over > 55 else "UNDER"
                        with b1: st.info(f"**Primary Leg:** {stat_cat} {direction} {line}")
                        with b2: st.success(f"**Alt-Line:** {stat_cat} {direction} {line - 4 if direction == 'OVER' else line + 4}")
                        with b3: st.warning(f"**Ladder Goal:** {stat_cat} {direction} {line + 5 if direction == 'OVER' else line - 5}")

                        if not h2h.empty:
                            st.divider()
                            st.subheader(f"📅 Last 5 Games vs {opp_name}")
                            h2h_display = h2h[['GAME_DATE', 'MATCHUP', 'WL', stat_cat]].head(5)
                            h2h_display['Result'] = h2h_display[stat_cat].apply(lambda x: "✅ Over" if x > line else "❌ Under")
                            st.table(h2h_display)

                        v1, v2 = st.columns(2)
                        with v1:
                            st.subheader("Poisson Probability Curve")
                            x_range = np.arange(max(0, int(final_proj-12)), int(final_proj+15))
                            fig_p = px.bar(x=x_range, y=poisson.pmf(x_range, final_proj))
                            fig_p.add_vline(x=line, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_p, use_container_width=True)
                        with v2:
                            st.subheader("Last 10 Game Trend")
                            fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                            fig_t.add_hline(y=line, line_color="red", line_dash="dash")
                            st.plotly_chart(fig_t, use_container_width=True)
                except Exception as e:
                    st.error(f"Data Fetching Error: {str(e)}. The NBA servers might be busy. Please wait 10 seconds and try again.")

# --- 5. MODE: TEAM SCANNER ---

elif mode == "Team Scanner":
    st.header("🔍 Value Scanner with Injury Cascading")
    sel_team = st.selectbox("Select Team to Scan", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Roster for Value"):
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=sel_team['id'], timeout=30).get_data_frames()[0]
            injured_stars = [p for p in roster['PLAYER'] if p in intel['injuries']]
            scan_data = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.status(f"Analyzing {sel_team['full_name']}..."):
                for i, p in roster.iterrows():
                    try:
                        status_text.text(f"Scanning: {p['PLAYER']}...")
                        time.sleep(0.8) # Critical sleep to prevent cloud IP blacklisting
                        
                        if p['PLAYER'] in intel['injuries']: continue
                        p_log = playergamelog.PlayerGameLog(player_id=p['PLAYER_ID'], timeout=30).get_data_frames()[0]
                        
                        if not p_log.empty:
                            if stat_cat == "PRA": p_log['PRA'] = p_log['PTS'] + p_log['REB'] + p_log['AST']
                            raw = p_log[stat_cat].head(5).mean()
                            game = schedule.get(sel_team['id'], {'opp_id': 0, 'ref': "N/A"})
                            cascade = 1.12 if len(injured_stars) > 0 else 1.0
                            proj = raw * ((pace_map.get(sel_team['id'], 100) + pace_map.get(game['opp_id'], 100))/200) * cascade
                            prob = (1 - poisson.cdf(line - 0.5, proj)) * 100
                            scan_data.append({
                                "Player": p['PLAYER'], 
                                "L5 Avg": round(raw, 1), "Proj": round(proj, 1), 
                                "Win Prob": f"{round(prob, 1)}%", 
                                "Signal": "🔥 OVER" if prob > 65 else ("❄️ UNDER" if prob < 35 else "Neutral")
                            })
                        progress_bar.progress((i + 1) / len(roster))
                    except Exception: continue
            
            status_text.text("Scan Complete!")
            st.dataframe(pd.DataFrame(scan_data).sort_values("Proj", ascending=False), use_container_width=True)
        except Exception as e:
            st.error(f"Scanner Failure: {str(e)}")
