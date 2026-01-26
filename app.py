import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import time
import random
import uuid
import unicodedata
from nba_api.stats.endpoints import (playergamelog, leaguegamefinder, 
                                     scoreboardv2, commonplayerinfo, 
                                     leaguedashteamstats, commonteamroster)
from nba_api.stats.static import players, teams
from nba_api.stats.library.http import NBAStatsHTTP

# --- 1. UTILITIES & STEALTH FINGERPRINTING ---

def normalize_string(text):
    """Removes accents (Dončić -> doncic) for bulletproof searching."""
    return "".join(
        c for c in unicodedata.normalize('NFD', text.lower())
        if unicodedata.category(c) != 'Mn'
    )

def get_stealth_headers():
    """Generates unique browser fingerprints and IDs for every request."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) Chrome/121.0.0.0 Safari/537.36"
    ]
    return {
        'Host': 'stats.nba.com',
        'Connection': 'keep-alive',
        'User-Agent': random.choice(user_agents),
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
        'x-nba-stats-request-id': str(uuid.uuid4()),
        'Referer': 'https://www.nba.com/',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
    }

# Apply global stealth headers
NBAStatsHTTP.headers = get_stealth_headers()

# --- 2. DATA ENGINES (Cached) ---

@st.cache_data(ttl=1800)
def get_intel():
    """Returns manual injury data and referee bias weights."""
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
        stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced', 
            timeout=30, 
            headers=get_stealth_headers()
        ).get_data_frames()[0]
        return {row['TEAM_ID']: row['PACE'] for _, row in stats.iterrows()}, stats['PACE'].mean()
    except: 
        return {}, 100.0

@st.cache_data(ttl=600)
def get_daily_schedule():
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        board = scoreboardv2.ScoreboardV2(game_date=today, timeout=30, headers=get_stealth_headers()).get_data_frames()[0]
        m_map = {}
        # Simulated refs for visualization
        refs = ["Scott Foster", "Marc Davis", "Jacyn Goble", "Bill Kennedy"]
        for i, row in board.iterrows():
            ref = refs[i % len(refs)]
            m_map[row['HOME_TEAM_ID']] = {'opp_id': row['VISITOR_TEAM_ID'], 'ref': ref}
            m_map[row['VISITOR_TEAM_ID']] = {'opp_id': row['HOME_TEAM_ID'], 'ref': ref}
        return m_map
    except: 
        return {}

# --- 3. DASHBOARD SETUP & SIDEBAR ---

st.set_page_config(page_title="Sharp Pro v11.5", layout="wide")
intel = get_intel()
pace_map, avg_pace = get_pace()
schedule = get_daily_schedule()
team_lookup = {t['id']: t['full_name'] for t in teams.get_teams()}

with st.sidebar:
    st.title("🏀 Sharp Pro v11.5")
    st.info(f"Stealth Active: UUID Enabled ✅")
    
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    st.divider()
    
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)
    st.divider()
    
    if st.button("Clear Cache & Refresh"):
        st.cache_data.clear()
        st.rerun()

# --- 4. MODE: SINGLE PLAYER ANALYSIS ---

if mode == "Single Player Analysis":
    st.header("👤 Player Matchup Engine")
    
    # ACCENT-INSENSITIVE SEARCH
    c1, c2 = st.columns(2)
    with c1:
        search_query = st.text_input("1. Search Name (Doncic, Jokic, etc.)", "Peyton Watson")
    
    norm_query = normalize_string(search_query)
    all_players = players.get_players()
    matches = [p for p in all_players if norm_query in normalize_string(p['full_name']) and p['is_active']]
    
    with c2:
        if matches:
            sel_p = st.selectbox("2. Confirm Selection", matches, format_func=lambda x: x['full_name'])
        else:
            st.warning("No players found. Check spelling.")
            st.stop()
    
    if st.button("🚀 Run Full Analysis"):
        with st.spinner("Decoding NBA API defense..."):
            try:
                # Basic Context
                p_info = commonplayerinfo.CommonPlayerInfo(player_id=sel_p['id'], timeout=30, headers=get_stealth_headers()).get_data_frames()[0]
                t_id = p_info['TEAM_ID'].iloc[0]
                game_context = schedule.get(t_id, {'opp_id': 0, 'ref': "Unknown"})
                opp_name = team_lookup.get(game_context['opp_id'], "Opponent")
                ref_data = intel['ref_bias'].get(game_context['ref'], {"type": "Neutral", "impact": 1.0})
                
                # Fetch Logs
                time.sleep(0.5)
                log = playergamelog.PlayerGameLog(player_id=sel_p['id'], season='2025-26', timeout=30, headers=get_stealth_headers()).get_data_frames()[0]
                
                # H2H Matchup (The Line 83 Fix)
                time.sleep(1.0)
                h2h = leaguegamefinder.LeagueGameFinder(
                    player_id_nullable=sel_p['id'], 
                    vs_team_id_nullable=game_context['opp_id'],
                    timeout=60,
                    headers=get_stealth_headers()
                ).get_data_frames()[0]
                
                if not log.empty:
                    if stat_cat == "PRA": 
                        log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                        if not h2h.empty: h2h['PRA'] = h2h['PTS'] + h2h['REB'] + h2h['AST']
                    
                    raw_avg = log[stat_cat].head(10).mean()
                    
                    # Pace Adjustment
                    p_pace = pace_map.get(t_id, avg_pace)
                    o_pace = pace_map.get(game_context['opp_id'], avg_pace)
                    comp_pace = (p_pace + o_pace) / 2
                    
                    # Injury Check
                    time.sleep(0.4)
                    roster = commonteamroster.CommonTeamRoster(team_id=t_id, timeout=30, headers=get_stealth_headers()).get_data_frames()[0]
                    injured_count = sum(1 for p in roster['PLAYER'] if p in intel['injuries'])
                    usage_boost = 1.12 if injured_count > 0 else 1.0
                    
                    # Final Projection Calculation
                    final_proj = raw_avg * (comp_pace / avg_pace) * ref_data['impact'] * usage_boost
                    prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

                    # UI: TOP METRICS
                    st.divider()
                    st.subheader(f"Analysis: {sel_p['full_name']} vs {opp_name}")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Sharp Projection", round(final_proj, 1), delta=f"{round(usage_boost,2)}x Usage")
                    m2.metric("Ref: " + game_context['ref'], ref_data['type'])
                    m3.metric("L10 Avg", round(raw_avg, 1))
                    m4.metric("Win Prob (Over)", f"{round(prob_over, 1)}%")

                    # UI: BETTING BLUEPRINT
                    st.divider()
                    st.subheader("🎯 Betting Blueprint")
                    b1, b2, b3 = st.columns(3)
                    direction = "OVER" if prob_over > 55 else "UNDER"
                    with b1: st.info(f"**Main Play:** {stat_cat} {direction} {line}")
                    with b2: st.success(f"**Alt Safety:** {stat_cat} {direction} {line - 3 if direction == 'OVER' else line + 3}")
                    with b3: st.warning(f"**Ladder Goal:** {stat_cat} {direction} {line + 6 if direction == 'OVER' else line - 6}")

                    # UI: DATA VISUALS
                    v1, v2 = st.columns(2)
                    with v1:
                        st.write("**Poisson Outcome Distribution**")
                        x_range = np.arange(max(0, int(final_proj-15)), int(final_proj+20))
                        fig_p = px.bar(x=x_range, y=poisson.pmf(x_range, final_proj), labels={'x': stat_cat, 'y': 'Prob'})
                        fig_p.add_vline(x=line, line_dash="dash", line_color="red", annotation_text="Line")
                        st.plotly_chart(fig_p, use_container_width=True)
                    with v2:
                        st.write("**Last 10 Games Trend**")
                        fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                        fig_t.add_hline(y=line, line_color="red", line_dash="dash")
                        st.plotly_chart(fig_t, use_container_width=True)

                    # UI: H2H TABLE
                    if not h2h.empty:
                        st.divider()
                        st.subheader(f"📅 Historical vs {opp_name}")
                        h2h_display = h2h[['GAME_DATE', 'MATCHUP', 'WL', stat_cat]].head(5)
                        h2h_display['Result'] = h2h_display[stat_cat].apply(lambda x: "✅ OVER" if x > line else "❌ UNDER")
                        st.table(h2h_display)
                else:
                    st.error("Player has no game logs for this season.")
            except Exception as e:
                st.error(f"Data Fetching Error: {str(e)}. Tip: The NBA servers are sensitive. Try again in 10 seconds.")

# --- 5. MODE: TEAM SCANNER ---

elif mode == "Team Scanner":
    st.header("🔍 Value Scanner")
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Roster"):
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=sel_team['id'], timeout=30, headers=get_stealth_headers()).get_data_frames()[0]
            scan_data = []
            
            progress_bar = st.progress(0)
            with st.status("Scanning roster for betting value...") as status:
                for i, p in roster.iterrows():
                    status.update(label=f"Analyzing {p['PLAYER']}...")
                    time.sleep(1.2) # Essential stealth delay
                    try:
                        p_log = playergamelog.PlayerGameLog(player_id=p['PLAYER_ID'], timeout=20, headers=get_stealth_headers()).get_data_frames()[0]
                        if not p_log.empty:
                            if stat_cat == "PRA": p_log['PRA'] = p_log['PTS'] + p_log['REB'] + p_log['AST']
                            avg = p_log[stat_cat].head(5).mean()
                            prob = (1 - poisson.cdf(line - 0.5, avg)) * 100
                            scan_data.append({
                                "Player": p['PLAYER'], 
                                "L5 Avg": round(avg, 1), 
                                "Prob (Over)": f"{round(prob, 1)}%",
                                "Signal": "🔥 HIGH" if prob > 70 else ("❄️ LOW" if prob < 30 else "Neutral")
                            })
                    except: continue
                    progress_bar.progress((i + 1) / len(roster))
            
            st.dataframe(pd.DataFrame(scan_data).sort_values("L5 Avg", ascending=False), use_container_width=True)
        except:
            st.error("Connection lost. NBA server is currently throttling this IP.")
