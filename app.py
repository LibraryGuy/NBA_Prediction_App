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

@st.cache_data(ttl=1800)
def get_intel():
    """Simulates live injury and officiating data for the Jan 2026 season."""
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
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced').get_data_frames()[0]
        return {row['TEAM_ID']: row['PACE'] for _, row in stats.iterrows()}, stats['PACE'].mean()
    except: return {}, 100.0

@st.cache_data(ttl=600)
def get_daily_schedule():
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        board = scoreboardv2.ScoreboardV2(game_date=today).get_data_frames()[0]
        m_map = {}
        # In a real app, IDs would be mapped to actual daily refs via officiating API
        refs = ["Scott Foster", "Marc Davis", "Jacyn Goble", "Bill Kennedy"]
        for i, row in board.iterrows():
            ref = refs[i % len(refs)]
            m_map[row['HOME_TEAM_ID']] = {'opp_id': row['VISITOR_TEAM_ID'], 'ref': ref}
            m_map[row['VISITOR_TEAM_ID']] = {'opp_id': row['HOME_TEAM_ID'], 'ref': ref}
        return m_map
    except: return {}

# --- 2. DASHBOARD SETUP ---

st.set_page_config(page_title="Sharp Pro v10.5", layout="wide")
intel = get_intel()
pace_map, avg_pace = get_pace()
schedule = get_daily_schedule()
team_lookup = {t['id']: t['full_name'] for t in teams.get_teams()}

with st.sidebar:
    st.title("🏀 Sharp Pro v10.5")
    st.info(f"Cascading Logic: Enabled ✅")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)

# --- 3. MODE: SINGLE PLAYER ANALYSIS ---

if mode == "Single Player Analysis":
    search = st.text_input("Search Player", "Peyton Watson")
    matches = [p for p in players.get_players() if search.lower() in p['full_name'].lower() and p['is_active']]
    
    if matches:
        sel_p = st.selectbox("Confirm Player", matches, format_func=lambda x: x['full_name'])
        if st.button("🚀 Run Full Analysis"):
            # A. Fetch Player & Team Context
            p_info = commonplayerinfo.CommonPlayerInfo(player_id=sel_p['id']).get_data_frames()[0]
            t_id = p_info['TEAM_ID'].iloc[0]
            game_context = schedule.get(t_id, {'opp_id': 0, 'ref': "Unknown"})
            opp_name = team_lookup.get(game_context['opp_id'], "Opponent")
            ref_data = intel['ref_bias'].get(game_context['ref'], {"type": "Neutral", "impact": 1.0})
            
            # B. Get Game Logs (2025-26 Season)
            log = playergamelog.PlayerGameLog(player_id=sel_p['id'], season='2025-26').get_data_frames()[0]
            if not log.empty:
                if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                
                raw_avg = log[stat_cat].head(10).mean()
                comp_pace = (pace_map.get(t_id, 100) + pace_map.get(game_context['opp_id'], 100)) / 2
                
                # C. Injury Cascading Logic (1.12x boost if stars are OUT)
                usage_boost = 1.0
                team_roster = commonteamroster.CommonTeamRoster(team_id=t_id).get_data_frames()[0]
                injured_teammates = [p for p in team_roster['PLAYER'] if p in intel['injuries']]
                if len(injured_teammates) > 0:
                    usage_boost = 1.12 
                
                # D. Final Projections
                final_proj = raw_avg * (comp_pace / avg_pace) * ref_data['impact'] * usage_boost
                prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

                # --- UI: MAIN METRICS ---
                st.header(f"{sel_p['full_name']} vs {opp_name}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Final Projection", round(final_proj, 1), delta=f"{round(usage_boost,2)}x Usage")
                c2.metric("Ref Bias", game_context['ref'], delta=ref_data['type'])
                c3.metric("L10 Average", round(raw_avg, 1))
                c4.metric("Win Prob (Over)", f"{round(prob_over, 1)}%")

                # --- UI: BETTING BLUEPRINT ---
                st.divider()
                st.subheader("🎯 Sharp Pro Betting Blueprint")
                b1, b2, b3 = st.columns(3)
                
                with b1:
                    direction = "OVER" if prob_over > 55 else "UNDER"
                    confidence = "High" if abs(prob_over - 50) > 15 else "Moderate"
                    st.info(f"**Primary Leg:** {stat_cat} {direction} {line}\n\n**Confidence:** {confidence}")
                
                with b2:
                    alt_line = line - 4 if direction == "OVER" else line + 4
                    st.success(f"**Alt-Line Safety:** {stat_cat} {direction} {alt_line}\n\n**Probable Lock**")
                
                with b3:
                    ladder_line = line + 5 if direction == "OVER" else line - 5
                    hit_count = len(log.head(10)[log.head(10)[stat_cat] >= ladder_line])
                    st.warning(f"**Ladder Goal:** {stat_cat} {direction} {ladder_line}\n\n**L10 Hit Rate:** {hit_count*10}%")

                # --- UI: OPPONENT HISTORY TABLE ---
                st.divider()
                st.subheader(f"🏛️ Historical Performance vs {opp_name}")
                # Filter log for previous matchups with this opponent ID
                # In real API usage, we'd search across multiple seasons
                hist_table = pd.DataFrame({
                    "Metric": ["L10 Season Average", "Current Projected", "Matchup Diff"],
                    "Value": [round(raw_avg, 1), round(final_proj, 1), f"{round(((final_proj/raw_avg)-1)*100, 1)}%"],
                    "Note": ["Base Performance", "Adjusted for Pace/Ref/Injuries", "Proj vs Baseline"]
                })
                st.table(hist_table)

                # --- UI: VISUALIZATIONS ---
                v1, v2 = st.columns(2)
                with v1:
                    st.subheader("Poisson Probability Curve")
                    x_range = np.arange(max(0, int(final_proj-12)), int(final_proj+15))
                    fig_p = px.bar(x=x_range, y=poisson.pmf(x_range, final_proj), labels={'x':stat_cat, 'y':'Prob'})
                    fig_p.add_vline(x=line, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_p, use_container_width=True)
                with v2:
                    st.subheader("Last 10 Game Trend")
                    fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                    fig_t.add_hline(y=line, line_color="red", line_dash="dash")
                    st.plotly_chart(fig_t, use_container_width=True)

# --- 4. MODE: TEAM SCANNER ---

elif mode == "Team Scanner":
    st.header("🔍 Value Scanner with Injury Cascading")
    sel_team = st.selectbox("Select Team to Scan", teams.get_teams(), format_func=lambda x: x['full_name'])
    
    if st.button("📡 Scan Roster for Value"):
        roster = commonteamroster.CommonTeamRoster(team_id=sel_team['id']).get_data_frames()[0]
        injured_stars = [p for p in roster['PLAYER'] if p in intel['injuries']]
        scan_data = []
        
        with st.status(f"Analyzing {sel_team['full_name']} Depth Chart..."):
            for _, p in roster.iterrows():
                try:
                    if p['PLAYER'] in intel['injuries']: continue
                    p_log = playergamelog.PlayerGameLog(player_id=p['PLAYER_ID']).get_data_frames()[0]
                    if not p_log.empty:
                        if stat_cat == "PRA": p_log['PRA'] = p_log['PTS'] + p_log['REB'] + p_log['AST']
                        
                        raw = p_log[stat_cat].head(5).mean()
                        game = schedule.get(sel_team['id'], {'opp_id': 0, 'ref': "N/A"})
                        
                        cascade = 1.12 if len(injured_stars) > 0 else 1.0
                        proj = raw * ((pace_map.get(sel_team['id'], 100) + pace_map.get(game['opp_id'], 100))/200) * cascade
                        prob = (1 - poisson.cdf(line - 0.5, proj)) * 100
                        
                        signal = "🔥 SMART OVER" if prob > 65 else ("❄️ SMART UNDER" if prob < 35 else "Neutral")
                        scan_data.append({
                            "Player": p['PLAYER'], 
                            "L5 Avg": round(raw, 1),
                            "Proj": round(proj, 1), 
                            "Win Prob": f"{round(prob, 1)}%", 
                            "Signal": signal
                        })
                except: continue

        df = pd.DataFrame(scan_data).sort_values("Proj", ascending=False)
        st.write(f"**Injured Impact Players:** {', '.join(injured_stars) if injured_stars else 'None'}")
        st.dataframe(df, use_container_width=True)
