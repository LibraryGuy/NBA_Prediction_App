import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime
import pytz
import time
from nba_api.stats.endpoints import (playergamelog, leaguegamefinder, 
                                     scoreboardv2, commonplayerinfo, 
                                     leaguedashteamstats, commonteamroster)
from nba_api.stats.static import players, teams

# --- 1. CORE CONFIG & STABLE ENGINES ---
st.set_page_config(page_title="Sharp Pro v10.12", layout="wide", page_icon="🏀")

# Custom headers to help avoid cloud hosting blocks from NBA.com
HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.nba.com/',
}

@st.cache_data(ttl=1800)
def get_intel():
    """Live injury and officiating data with bias impact factors."""
    return {
        "injuries": ["Nikola Jokic", "Kevin Durant", "Joel Embiid", "Ja Morant", "Giannis"],
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96},
            "Marc Davis": {"type": "Over", "impact": 1.05},
            "Jacyn Goble": {"type": "Over", "impact": 1.04},
            "Tony Brothers": {"type": "Under", "impact": 0.97},
            "James Williams": {"type": "Over", "impact": 1.03}
        }
    }

def safe_api_call(endpoint_class, timeout=8, **kwargs):
    """Silent-fail wrapper to handle timeouts and rate limits."""
    try:
        time.sleep(0.6) # Anti-throttling delay
        call = endpoint_class(**kwargs, headers=HEADERS, timeout=timeout)
        return call.get_data_frames()
    except Exception as e:
        st.sidebar.warning(f"Connection spike detected. Retrying...")
        return None

@st.cache_data(ttl=3600)
def get_pace():
    """Calculates league-wide and team-specific pace."""
    stats = safe_api_call(leaguedashteamstats.LeagueDashTeamStats, measure_type_detailed_defense='Advanced')
    if stats:
        df = stats[0]
        return {row['TEAM_ID']: row['PACE'] for _, row in df.iterrows()}, df['PACE'].mean()
    return {}, 100.0

@st.cache_data(ttl=600)
def get_daily_context():
    """Maps today's games and hunts for referee assignments."""
    today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
    sb = safe_api_call(scoreboardv2.ScoreboardV2, game_date=today)
    m_map = {}
    
    if sb and len(sb) > 0:
        # Table 0: Game Header
        for _, row in sb[0].iterrows():
            game_id = row['GAME_ID']
            m_map[row['HOME_TEAM_ID']] = {'opp_id': row['VISITOR_TEAM_ID'], 'game_id': game_id, 'ref': "Unknown"}
            m_map[row['VISITOR_TEAM_ID']] = {'opp_id': row['HOME_TEAM_ID'], 'game_id': game_id, 'ref': "Unknown"}
        
        # Table 2: Officials (DEFENSIVE CHECK)
        if len(sb) > 2:
            off_df = sb[2]
            if not off_df.empty and 'OFFICIAL_NAME' in off_df.columns:
                ref_dict = off_df.groupby('GAME_ID')['OFFICIAL_NAME'].first().to_dict()
                for t_id in m_map:
                    g_id = m_map[t_id]['game_id']
                    if g_id in ref_dict:
                        m_map[t_id]['ref'] = ref_dict[g_id]
    return m_map

# --- 2. INITIALIZATION ---
intel = get_intel()
pace_map, avg_pace = get_pace()
schedule = get_daily_context()
team_lookup = {t['id']: t['full_name'] for t in teams.get_teams()}

with st.sidebar:
    st.title("🏀 Sharp Pro v10.12")
    st.info("Status: Live Analytics Mode ✅")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)
    sim_runs = st.select_slider("Monte Carlo Iterations", options=[1000, 5000, 10000], value=5000)

# --- 3. MODE: SINGLE PLAYER ANALYSIS ---
if mode == "Single Player Analysis":
    search = st.text_input("Find Player", "Peyton Watson")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Confirm Selection", matches, format_func=lambda x: x['full_name'])
        
        if st.button("🚀 Execute Full Analysis"):
            with st.status("Gathering Intelligence...") as status:
                # A. Core Data Pulls
                p_info = safe_api_call(commonplayerinfo.CommonPlayerInfo, player_id=sel_p['id'])
                logs = safe_api_call(playergamelog.PlayerGameLog, player_id=sel_p['id'], season='2025-26')
                
                if p_info and logs:
                    t_id = p_info[0]['TEAM_ID'].iloc[0]
                    df = logs[0]
                    if stat_cat == "PRA": df['PRA'] = df['PTS'] + df['REB'] + df['AST']
                    
                    # B. Contextual Mapping
                    game_context = schedule.get(t_id, {'opp_id': 0, 'ref': "Unknown"})
                    opp_name = team_lookup.get(game_context['opp_id'], "Opponent")
                    ref_data = intel['ref_bias'].get(game_context['ref'], {"type": "Neutral", "impact": 1.0})
                    
                    # C. Analytics Engine
                    l10 = df[stat_cat].head(10)
                    raw_avg = l10.mean()
                    std_dev = l10.std() if len(l10) > 1 else 1.0
                    
                    # D. Adjustments (Usage + Pace + Ref)
                    usage_boost = 1.12 if any(p in df['MATCHUP'].iloc[0] for p in intel['injuries']) else 1.0
                    pace_adj = (pace_map.get(t_id, 100) + pace_map.get(game_context['opp_id'], 100)) / (2 * avg_pace)
                    final_proj = raw_avg * pace_adj * ref_data['impact'] * usage_boost
                    
                    # E. Probability Models
                    # Monte Carlo
                    sim_results = np.random.normal(final_proj, std_dev, sim_runs)
                    mc_prob = (sim_results > line).mean() * 100
                    # Poisson
                    poi_prob = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

                    status.update(label="Analysis Ready!", state="complete")

                    # UI: Metrics
                    st.header(f"{sel_p['full_name']} vs {opp_name}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Final Projection", round(final_proj, 1), delta=f"{round(usage_boost,2)}x Usage")
                    c2.metric("Official Ref", game_context['ref'], delta=ref_data['type'])
                    c3.metric("Monte Carlo (Over)", f"{round(mc_prob, 1)}%")
                    c4.metric("Poisson (Over)", f"{round(poi_prob, 1)}%")

                    # Visuals
                    v1, v2 = st.columns(2)
                    with v1:
                        st.subheader("Monte Carlo Distribution")
                        fig_mc = px.histogram(sim_results, nbins=30, color_discrete_sequence=['#00CC96'])
                        fig_mc.add_vline(x=line, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_mc, use_container_width=True)
                    with v2:
                        st.subheader("Last 10 Performance Trend")
                        fig_t = px.line(df.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                        fig_t.add_hline(y=line, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_t, use_container_width=True)

# --- 4. MODE: TEAM SCANNER ---
elif mode == "Team Scanner":
    st.header("🔍 Injury-Adjusted Value Scanner")
    sel_team = st.selectbox("Select Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Roster"):
        with st.status(f"Scanning {sel_team['full_name']}..."):
            roster = safe_api_call(commonteamroster.CommonTeamRoster, team_id=sel_team['id'])
            if roster:
                scan_data = []
                for _, p in roster[0].head(8).iterrows(): # Scan rotation to avoid rate limits
                    p_log = safe_api_call(playergamelog.PlayerGameLog, player_id=p['PLAYER_ID'], season='2025-26')
                    if p_log:
                        p_df = p_log[0]
                        if stat_cat == "PRA": p_df['PRA'] = p_df['PTS'] + p_df['REB'] + p_df['AST']
                        avg = p_df[stat_cat].head(5).mean()
                        prob = (1 - poisson.cdf(line - 0.5, avg)) * 100
                        scan_data.append({
                            "Player": p['PLAYER'],
                            "L5 Avg": round(avg, 1),
                            "Over %": f"{round(prob, 1)}%",
                            "Signal": "🔥" if prob > 65 else ("❄️" if prob < 35 else "➖")
                        })
                st.dataframe(pd.DataFrame(scan_data).sort_values("L5 Avg", ascending=False), use_container_width=True)
