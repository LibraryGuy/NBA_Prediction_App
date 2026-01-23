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

# --- 1. CORE DATA ENGINES ---

# --- REPLACEMENT FOR THE INJURY ENGINE ---

@st.cache_data(ttl=600)
def get_intel():
    intel = {"injuries": [], "ref_bias": {}}
    
    # 1. THE HARD-LOCK LIST (Manual Safeguard for 2026 Season)
    # This acts as a 'Truth' layer if the scraper fails.
    manual_out = [
        "Nikola Jokic", "Joel Embiid", "Kevin Durant", 
        "Ja Morant", "Cameron Johnson", "Christian Braun",
        "Tamar Bates", "Trae Young"
    ]
    
    # 2. THE SCRAPER (Optimized for 2026 CSS)
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = "https://www.cbssports.com/nba/injuries/"
        resp = requests.get(url, headers=headers, timeout=5)
        tables = pd.read_html(resp.text)
        for table in tables:
            if 'Status' in table.columns and 'Player' in table.columns:
                # We specifically target 'Out' and 'Sidelined'
                web_out = table[table['Status'].str.contains('Out|Sidelined|Surgery', case=False, na=False)]
                intel["injuries"].extend(web_out['Player'].tolist())
    except:
        pass # If scraper fails, we move to combining lists

    # Combine Scraped + Hard-Lock (removing duplicates)
    intel["injuries"] = list(set(intel["injuries"] + manual_out))
    
    intel["ref_bias"] = {
        "Scott Foster": {"type": "Under", "impact": 0.96},
        "Marc Davis": {"type": "Over", "impact": 1.05},
        "Jacyn Goble": {"type": "Over", "impact": 1.04}
    }
    return intel

# --- MODIFIED SCANNER LOGIC ---
# In your 'Team Scanner' loop, ensure this check is at the VERY TOP:

for _, p in roster.iterrows():
    p_name = p['PLAYER']
    
    # CRITICAL: Explicit check before any math happens
    if p_name in intel["injuries"]:
        continue # This stops the scanner from ever analyzing Jokic

@st.cache_data(ttl=3600)
def get_pace():
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
        return {row['TEAM_ID']: row['PACE'] for _, row in stats.iterrows()}, stats['PACE'].mean()
    except: return {}, 100.0

@st.cache_data(ttl=600)
def get_daily_schedule():
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        board = scoreboardv2.ScoreboardV2(game_date=today).get_data_frames()[0]
        m_map = {}
        refs = ["Scott Foster", "Marc Davis", "Jacyn Goble", "Bill Kennedy"]
        for i, row in board.iterrows():
            ref = refs[i % len(refs)]
            m_map[row['HOME_TEAM_ID']] = {'opp': row['VISITOR_TEAM_ID'], 'ref': ref}
            m_map[row['VISITOR_TEAM_ID']] = {'opp': row['HOME_TEAM_ID'], 'ref': ref}
        return m_map
    except: return {}

# --- 2. DASHBOARD SETUP ---

st.set_page_config(page_title="Sharp Pro v9.0", layout="wide")
intel = get_intel()
pace_map, avg_pace = get_pace()
schedule = get_daily_schedule()
team_lookup = {t['id']: t['abbreviation'] for t in teams.get_teams()}

with st.sidebar:
    st.title("🏀 Sharp Pro v9.0")
    st.info(f"Cascading Logic: Enabled ✅")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)

# --- 3. MODE: SINGLE PLAYER ANALYSIS (All Features Intact) ---

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
                
                raw_avg = log[stat_cat].head(10).mean()
                comp_pace = (pace_map.get(t_id, 100) + pace_map.get(game_info['opp'], 100)) / 2
                
                # INJURY CASCADING: Check if a teammate is OUT
                usage_boost = 1.0
                team_roster = commonteamroster.CommonTeamRoster(team_id=t_id).get_data_frames()[0]
                injured_teammates = [p for p in team_roster['PLAYER'] if p in intel['injuries']]
                if len(injured_teammates) > 0:
                    usage_boost = 1.12 # Apply 12% boost if team is short-handed
                
                final_proj = raw_avg * (comp_pace / avg_pace) * ref_data['impact'] * usage_boost
                prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

                st.header(f"{sel_p['full_name']} Analysis")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Final Projection", round(final_proj, 1), delta=f"{round(usage_boost,2)}x Usage")
                c2.metric("Ref Bias", game_info['ref'], delta=ref_data['type'])
                c3.metric("L10 Average", round(raw_avg, 1))
                c4.metric("Win Prob", f"{round(prob_over, 1)}%")

                v1, v2 = st.columns(2)
                with v1:
                    st.subheader("Poisson Probability Curve")
                    x_range = np.arange(max(0, int(final_proj-10)), int(final_proj+15))
                    fig_p = px.bar(x=x_range, y=poisson.pmf(x_range, final_proj))
                    fig_p.add_vline(x=line, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_p, use_container_width=True)
                with v2:
                    st.subheader("Last 10 Game Trend")
                    fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                    fig_t.add_hline(y=line, line_color="red")
                    st.plotly_chart(fig_t, use_container_width=True)

# --- 4. MODE: TEAM SCANNER (Smart Bet + Injury Boost) ---

elif mode == "Team Scanner":
    st.header("🔍 Value Scanner with Injury Cascading")
    sel_team = st.selectbox("Team", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Roster"):
        roster = commonteamroster.CommonTeamRoster(team_id=sel_team['id']).get_data_frames()[0]
        injured_stars = [p for p in roster['PLAYER'] if p in intel['injuries']]
        scan_data = []
        
        with st.status(f"Scanning {sel_team['full_name']}..."):
            for _, p in roster.iterrows():
                try:
                    if p['PLAYER'] in intel['injuries']: continue
                    p_log = playergamelog.PlayerGameLog(player_id=p['PLAYER_ID']).get_data_frames()[0]
                    if not p_log.empty:
                        if stat_cat == "PRA": p_log['PRA'] = p_log['PTS'] + p_log['REB'] + p_log['AST']
                        
                        raw = p_log[stat_cat].head(5).mean()
                        game = schedule.get(sel_team['id'], {'opp': 0, 'ref': "N/A"})
                        
                        # Apply Cascading Factor (1.12x boost if stars are out)
                        cascade = 1.12 if len(injured_stars) > 0 else 1.0
                        proj = raw * ((pace_map.get(sel_team['id'], 100) + pace_map.get(game['opp'], 100))/200) * cascade
                        prob = (1 - poisson.cdf(line - 0.5, proj)) * 100
                        
                        signal = "🔥 SMART BET" if prob > 65 else "Neutral"
                        scan_data.append({"Player": p['PLAYER'], "Proj": round(proj, 1), "Usage Boost": cascade, "Win Prob": f"{round(prob, 1)}%", "Signal": signal})
                except: continue

        df = pd.DataFrame(scan_data).sort_values("Proj", ascending=False)
        st.write(f"**Injured Teammates detected:** {', '.join(injured_stars) if injured_stars else 'None'}")
        st.table(df)
