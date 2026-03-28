import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
import time
from balldontlie import BalldontlieAPI

# --- 1. INITIALIZATION ---
# Hardcoded API Key as requested
BDL_API_KEY = "ea294f0e-31cd-4b1b-aedb-e8fd246b907f"
api = BalldontlieAPI(api_key=BDL_API_KEY)

# --- 2. DATA ENGINES (Rate-Limit Proof) ---

@st.cache_data(ttl=3600)
def get_intel():
    return {
        "injuries": ["Nikola Jokic", "Kevin Durant", "Joel Embiid", "Ja Morant"],
        "ref_bias": {"Scott Foster": 0.96, "Marc Davis": 1.05}
    }

@st.cache_data(ttl=86400) # Teams don't change daily
def get_all_teams():
    try:
        return {t.id: {"full_name": t.full_name, "abbreviation": t.abbreviation} for t in api.nba.teams.list()}
    except: return {}

@st.cache_data(ttl=86400)
def search_players(query):
    if not query or len(query) < 3: return []
    try:
        results = api.nba.players.list(search=query)
        return results.data if results.data else []
    except Exception as e:
        if "429" in str(e): st.error("Rate limit hit. Wait 60s.")
        return []

@st.cache_data(ttl=86400) # Extreme caching to save requests
def get_player_stats(player_id):
    """Fetches stats with auto-retry logic for the 5 req/min free limit."""
    for season in [2025, 2024]:
        retries = 2
        while retries > 0:
            try:
                stats = api.nba.stats.list(player_ids=[player_id], seasons=[season], per_page=20)
                if not stats.data: break # Try next season
                
                data = [{"DATE": s.game.date, "PTS": s.pts or 0, "REB": s.reb or 0, "AST": s.ast or 0} 
                        for s in stats.data if s.pts is not None]
                
                if data:
                    df = pd.DataFrame(data)
                    df['DATE'] = pd.to_datetime(df['DATE']).dt.date
                    return df.sort_values('DATE', ascending=False)
                break
            except Exception as e:
                if "429" in str(e):
                    st.toast("⏳ API Limit Hit. Self-healing in 15s...", icon="⚠️")
                    time.sleep(15) # Pause to let the rate limit window clear
                    retries -= 1
                else: return pd.DataFrame()
    return pd.DataFrame()

# --- 3. DASHBOARD UI ---

st.set_page_config(page_title="Sharp Pro v12.3", layout="wide")
intel = get_intel()
team_map = get_all_teams()

with st.sidebar:
    st.title("🏀 Sharp Pro v12.3")
    st.info("Limit: 5 Requests/Min (Free Tier)")
    mode = st.radio("Navigation", ["Single Player", "Team Scanner"])
    stat_cat = st.selectbox("Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Line", value=22.5, step=0.5)
    if st.button("Flush Cache"):
        st.cache_data.clear()
        st.rerun()

# --- 4. SINGLE PLAYER ANALYSIS (REVISED) ---
if mode == "Single Player":
    st.header("👤 Player Analyst")
    search_q = st.text_input("Search Name (e.g., 'LeBron')", "Luka Doncic")
    
    matches = search_players(search_q)
    
    if matches:
        # format_func ensures the UI looks good while 'p' remains the object
        p = st.selectbox("Confirm Player", matches, 
                         format_func=lambda x: f"{x.first_name} {x.last_name} ({x.team.abbreviation if x.team else 'N/A'})")
        
        if st.button("🚀 Analyze"):
            with st.spinner(f"Fetching data for {p.first_name}..."):
                log = get_player_stats(p.id)
                
                if not log.empty:
                    # Calculate PRA if selected
                    if stat_cat == "PRA": 
                        log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                    
                    # Statistical Logic
                    recent_games = log.head(10)
                    avg = recent_games[stat_cat].mean()
                    
                    # Poisson Projection
                    proj = avg * (1.12 if f"{p.first_name} {p.last_name}" in intel['injuries'] else 1.0)
                    # Simple Poisson: Prob of getting MORE than 'line'
                    prob = (1 - poisson.cdf(line - 0.5, proj)) * 100

                    st.divider()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current Projection", round(proj, 1))
                    m2.metric("L10 Average", round(avg, 1))
                    m3.metric("Probability Over", f"{round(prob, 1)}%")

                    v1, v2 = st.columns(2)
                    with v1:
                        # Probability Distribution Chart
                        x_axis = np.arange(max(0, int(proj-15)), int(proj+15))
                        y_axis = [poisson.pmf(i, proj) for i in x_axis]
                        fig_dist = px.bar(x=x_axis, y=y_axis, title=f"{stat_cat} Probability Dist.", labels={'x':stat_cat, 'y':'Prob'})
                        st.plotly_chart(fig_dist, use_container_width=True)
                    with v2:
                        # Trend Chart
                        fig_trend = px.line(recent_games.iloc[::-1], x='DATE', y=stat_cat, title="Last 10 Games Trend", markers=True)
                        st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.warning("No recent stats found for this player in 2024-2025.")
    else:
        st.info("Type at least 3 characters to search for a player.")

# --- 5. TEAM SCANNER (REVISED) ---
elif mode == "Team Scanner":
    st.header("🔍 Team Value Scanner")
    # Clean list creation
    t_options = {t.full_name: t.id for t in api.nba.teams.list() if t.id <= 30}
    sel_team_name = st.selectbox("Select Team", sorted(t_options.keys()))
    sel_team_id = t_options[sel_team_name]
    
    if st.button("📡 Scan Top Rotation"):
        # We fetch players and just take a slice to avoid hitting rate limits instantly
        roster_resp = api.nba.players.list(team_ids=[sel_team_id])
        roster = roster_resp.data[:10] # Top 10 results from the search
        
        scan_data = []
        prog_bar = st.progress(0)
        status_text = st.empty()
        
        for i, player in enumerate(roster):
            status_text.text(f"Analyzing {player.first_name} {player.last_name}...")
            
            # This is the bottleneck (5 req/min). 
            # 12 seconds per player = 5 players per minute.
            p_log = get_player_stats(player.id)
            
            if not p_log.empty:
                if stat_cat == "PRA": 
                    p_log['PRA'] = p_log['PTS'] + p_log['REB'] + p_log['AST']
                val = p_log[stat_cat].head(5).mean()
                scan_data.append({
                    "Player": f"{player.first_name} {player.last_name}", 
                    f"L5 {stat_cat} Avg": round(val, 1),
                    "Status": "Active"
                })
            
            prog_bar.progress((i + 1) / len(roster))
            if i < len(roster) - 1:
                time.sleep(12) # Crucial for Free Tier
        
        status_text.success("Scan Complete!")
        if scan_data:
            st.table(pd.DataFrame(scan_data))
        else:
            st.error("No active stat data found for this team's roster.")
