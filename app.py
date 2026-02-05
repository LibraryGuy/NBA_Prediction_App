import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import time
from balldontlie import BalldontlieAPI

# --- 1. INITIALIZATION ---
# Replace with your actual API key from balldontlie.io
BDL_API_KEY = "ea294f0e-31cd-4b1b-aedb-e8fd246b907f"
api = BalldontlieAPI(api_key=BDL_API_KEY)

# --- 2. DATA ENGINES (Balldontlie Refactor) ---

@st.cache_data(ttl=3600)
def get_intel():
    """Manual overlay for injuries and ref bias."""
    return {
        "injuries": ["Nikola Jokic", "Kevin Durant", "Joel Embiid", "Ja Morant"],
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96},
            "Marc Davis": {"type": "Over", "impact": 1.05},
            "Jacyn Goble": {"type": "Over", "impact": 1.04}
        }
    }

@st.cache_data(ttl=86400)
def get_all_teams():
    """Fetches static team list."""
    try:
        teams_data = api.nba.teams.list()
        return {t.id: {"full_name": t.full_name, "abbreviation": t.abbreviation} for t in teams_data}
    except:
        return {}

@st.cache_data(ttl=3600)
def get_player_data(search_query):
    """Searches for a player and returns their BDL ID and info."""
    try:
        results = api.nba.players.list(search=search_query)
        if results.data:
            p = results.data[0]
            return {"id": p.id, "full_name": f"{p.first_name} {p.last_name}", "team_id": p.team.id}
        return None
    except:
        return None

@st.cache_data(ttl=600)
def get_player_stats(player_id, season=2024):
    """Fetches recent game logs. Balldontlie uses 'stats' for box scores."""
    try:
        # Fetch last 15 games to ensure we have enough for L10 analysis
        stats = api.nba.stats.list(
            player_ids=[player_id], 
            seasons=[season], 
            per_page=15
        )
        data = []
        for s in stats.data:
            data.append({
                "GAME_DATE": s.game.date,
                "PTS": s.pts or 0,
                "REB": s.reb or 0,
                "AST": s.ast or 0,
                "MATCHUP": f"vs {s.game.visitor_team_id if s.team.id == s.game.home_team_id else s.game.home_team_id}"
            })
        df = pd.DataFrame(data)
        if not df.empty:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date
            df = df.sort_values('GAME_DATE', ascending=False)
        return df
    except Exception as e:
        st.error(f"API Error: {e}")
        return pd.DataFrame()

# --- 3. DASHBOARD SETUP ---

st.set_page_config(page_title="Sharp Pro v12.0 (BDL Edition)", layout="wide")
intel = get_intel()
team_map = get_all_teams()

with st.sidebar:
    st.title("🏀 Sharp Pro v12.0")
    st.success("API: Balldontlie Free Tier ✅")
    
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
    
    search_query = st.text_input("1. Enter Player Name (e.g. Luka Doncic)", "Peyton Watson")
    
    if search_query:
        player_info = get_player_data(search_query)
        
        if player_info:
            st.info(f"Selected: **{player_info['full_name']}**")
            
            if st.button("🚀 Run Full Analysis"):
                with st.spinner("Analyzing stats via Balldontlie..."):
                    log = get_player_stats(player_info['id'])
                    
                    if not log.empty:
                        # Logic for PRA
                        if stat_cat == "PRA":
                            log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                        
                        raw_avg = log[stat_cat].head(10).mean()
                        
                        # Injury/Usage Logic (Manual from Intel)
                        usage_boost = 1.12 if any(p in intel['injuries'] for p in [player_info['full_name']]) else 1.0
                        
                        # Balldontlie doesn't provide live Pace/Refs in the free tier easily,
                        # so we use a neutral multiplier (1.0) or manual adjustment.
                        final_proj = raw_avg * usage_boost
                        prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

                        # UI: TOP METRICS
                        st.divider()
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Sharp Projection", round(final_proj, 1), delta=f"{usage_boost}x Boost")
                        m2.metric("Stat Category", stat_cat)
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
                            fig_p.add_vline(x=line, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_p, use_container_width=True)
                        with v2:
                            st.write("**Last 10 Games Trend**")
                            fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                            fig_t.add_hline(y=line, line_color="red", line_dash="dash")
                            st.plotly_chart(fig_t, use_container_width=True)
                    else:
                        st.error("No recent game logs found for this player in the 2024-25 season.")
        else:
            st.warning("Player not found.")

# --- 5. MODE: TEAM SCANNER ---

elif mode == "Team Scanner":
    st.header("🔍 Value Scanner")
    
    if team_map:
        team_options = sorted([{"id": k, "name": v['full_name']} for k, v in team_map.items()], key=lambda x: x['name'])
        sel_team = st.selectbox("Select Team", team_options, format_func=lambda x: x['name'])
        
        if st.button("📡 Scan Roster"):
            try:
                # Get players for the specific team
                roster = api.nba.players.list(team_ids=[sel_team['id']])
                scan_data = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, p in enumerate(roster.data):
                    status_text.text(f"Analyzing {p.first_name} {p.last_name}...")
                    
                    # Essential delay for Free Tier (30 requests/min)
                    time.sleep(2.0) 
                    
                    p_log = get_player_stats(p.id)
                    if not p_log.empty:
                        if stat_cat == "PRA": 
                            p_log['PRA'] = p_log['PTS'] + p_log['REB'] + p_log['AST']
                        
                        avg = p_log[stat_cat].head(5).mean()
                        prob = (1 - poisson.cdf(line - 0.5, avg)) * 100
                        
                        scan_data.append({
                            "Player": f"{p.first_name} {p.last_name}", 
                            "L5 Avg": round(avg, 1), 
                            "Prob (Over)": f"{round(prob, 1)}%",
                            "Signal": "🔥 HIGH" if prob > 70 else ("❄️ LOW" if prob < 30 else "Neutral")
                        })
                    
                    progress_bar.progress((i + 1) / len(roster.data))
                
                status_text.text("Scan Complete!")
                if scan_data:
                    st.dataframe(pd.DataFrame(scan_data).sort_values("L5 Avg", ascending=False), use_container_width=True)
                else:
                    st.warning("No active stats found for this roster.")
                    
            except Exception as e:
                st.error(f"Scanner Error: {e}")
    else:
        st.error("Could not load team list. Check your API key.")
