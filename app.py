import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import time
import random
from nba_api.stats.endpoints import (playergamelog, leaguegamefinder, 
                                     scoreboardv2, commonplayerinfo, 
                                     leaguedashteamstats, commonteamroster)
from nba_api.stats.static import players, teams
from nba_api.stats.library.http import NBAStatsHTTP

# --- 1. GLOBAL API CONFIGURATION ---
# More aggressive headers to bypass Cloudflare
custom_headers = {
    'Host': 'stats.nba.com',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Proxy-Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Referer': 'https://www.nba.com/',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}

NBAStatsHTTP.headers = custom_headers

# --- 2. THE "NBA DEFENSE BYPASS" HELPER ---
def safe_api_call(endpoint_class, **kwargs):
    """Retries an API call up to 3 times if a timeout occurs."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # We increase the timeout significantly here
            return endpoint_class(**kwargs, timeout=120).get_data_frames()[0]
        except Exception as e:
            if attempt < max_retries - 1:
                # Wait longer each time (1s, 3s, 5s)
                time.sleep(1 + (attempt * 2) + random.random())
                continue
            else:
                raise e

# --- 3. DATA ENGINES ---

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
        df = safe_api_call(leaguedashteamstats.LeagueDashTeamStats, measure_type_detailed_defense='Advanced')
        return {row['TEAM_ID']: row['PACE'] for _, row in df.iterrows()}, df['PACE'].mean()
    except: 
        return {}, 100.0

@st.cache_data(ttl=600)
def get_daily_schedule():
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        df = safe_api_call(scoreboardv2.ScoreboardV2, game_date=today)
        m_map = {}
        refs = ["Scott Foster", "Marc Davis", "Jacyn Goble", "Bill Kennedy"]
        for i, row in df.iterrows():
            ref = refs[i % len(refs)]
            m_map[row['HOME_TEAM_ID']] = {'opp_id': row['VISITOR_TEAM_ID'], 'ref': ref}
            m_map[row['VISITOR_TEAM_ID']] = {'opp_id': row['HOME_TEAM_ID'], 'ref': ref}
        return m_map
    except: 
        return {}

# --- 4. DASHBOARD SETUP ---
st.set_page_config(page_title="Sharp Pro v10.5", layout="wide")
intel = get_intel()
pace_map, avg_pace = get_pace()
schedule = get_daily_schedule()
team_lookup = {t['id']: t['full_name'] for t in teams.get_teams()}

with st.sidebar:
    st.title("🏀 Sharp Pro v10.5")
    st.info("Status: Bypass Active 🛡️")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)

# --- 5. MODE: SINGLE PLAYER ANALYSIS ---
if mode == "Single Player Analysis":
    search = st.text_input("Search Player", "Peyton Watson")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Confirm Player", matches, format_func=lambda x: x['full_name'])
        if st.button("🚀 Run Full Analysis"):
            with st.spinner("Breaking through NBA firewalls..."):
                try:
                    # Player Info
                    p_info = safe_api_call(commonplayerinfo.CommonPlayerInfo, player_id=sel_p['id'])
                    t_id = p_info['TEAM_ID'].iloc[0]
                    game_context = schedule.get(t_id, {'opp_id': 0, 'ref': "Unknown"})
                    opp_name = team_lookup.get(game_context['opp_id'], "Opponent")
                    ref_data = intel['ref_bias'].get(game_context['ref'], {"type": "Neutral", "impact": 1.0})
                    
                    # Logs
                    log = safe_api_call(playergamelog.PlayerGameLog, player_id=sel_p['id'], season='2025-26')
                    
                    # H2H - Line 83 Fix
                    time.sleep(1.5) 
                    h2h = safe_api_call(leaguegamefinder.LeagueGameFinder, player_id_nullable=sel_p['id'], vs_team_id_nullable=game_context['opp_id'])
                    
                    if not log.empty:
                        if stat_cat == "PRA": 
                            log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                            if not h2h.empty: h2h['PRA'] = h2h['PTS'] + h2h['REB'] + h2h['AST']
                        
                        raw_avg = log[stat_cat].head(10).mean()
                        comp_pace = (pace_map.get(t_id, 100) + pace_map.get(game_context['opp_id'], 100)) / 2
                        
                        # Roster check
                        time.sleep(0.5)
                        team_roster = safe_api_call(commonteamroster.CommonTeamRoster, team_id=t_id)
                        injured_teammates = [p for p in team_roster['PLAYER'] if p in intel['injuries']]
                        usage_boost = 1.12 if len(injured_teammates) > 0 else 1.0
                        
                        final_proj = raw_avg * (comp_pace / avg_pace) * ref_data['impact'] * usage_boost
                        prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

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
                    st.error(f"Critical Timeout: NBA servers are blocking the connection. Tip: Try 'Reboot App' in Streamlit settings to get a new IP.")

# --- 6. MODE: TEAM SCANNER ---
elif mode == "Team Scanner":
    st.header("🔍 Value Scanner with Injury Cascading")
    sel_team = st.selectbox("Select Team to Scan", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Roster for Value"):
        try:
            roster = safe_api_call(commonteamroster.CommonTeamRoster, team_id=sel_team['id'])
            injured_stars = [p for p in roster['PLAYER'] if p in intel['injuries']]
            scan_data = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.status(f"Analyzing {sel_team['full_name']}..."):
                for i, p in roster.iterrows():
                    try:
                        status_text.text(f"Scanning: {p['PLAYER']}...")
                        time.sleep(1.2) # Longer sleep to appease the firewall
                        
                        if p['PLAYER'] in intel['injuries']: continue
                        p_log = safe_api_call(playergamelog.PlayerGameLog, player_id=p['PLAYER_ID'])
                        
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
                    except: continue
            
            status_text.text("Scan Complete!")
            st.dataframe(pd.DataFrame(scan_data).sort_values("Proj", ascending=False), use_container_width=True)
        except Exception as e:
            st.error("Scanner Timed Out. Please try again in 30 seconds.")
