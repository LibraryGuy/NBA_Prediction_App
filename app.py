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

# --- 4. SINGLE PLAYER ANALYSIS ---

if mode == "Single Player":
    st.header("👤 Player Analyst")
    c1, c2 = st.columns(2)
    with c1:
        search_q = st.text_input("Search Name", "Luka Doncic")
    
    matches = search_players(search_q)
    with c2:
        if matches:
            p = st.selectbox("Confirm Player", matches, format_func=lambda x: f"{x.first_name} {x.last_name}")
        else:
            st.stop()

    if st.button("🚀 Analyze"):
        log = get_player_stats(p.id)
        if not log.empty:
            if stat_cat == "PRA": log['PRA'] = log['PTS'] + log['REB'] + log['AST']
            
            avg = log[stat_cat].head(10).mean()
            proj = avg * (1.12 if f"{p.first_name} {p.last_name}" in intel['injuries'] else 1.0)
            prob = (1 - poisson.cdf(line - 0.5, proj)) * 100

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Projection", round(proj, 1))
            m2.metric("L10 Avg", round(avg, 1))
            m3.metric("Over Prob", f"{round(prob, 1)}%")

            v1, v2 = st.columns(2)
            with v1:
                st.plotly_chart(px.bar(x=np.arange(max(0, int(proj-10)), int(proj+15)), 
                                       y=poisson.pmf(np.arange(max(0, int(proj-10)), int(proj+15)), proj), 
                                       title="Probability Curve"), use_container_width=True)
            with v2:
                st.plotly_chart(px.line(log.head(10).iloc[::-1], x='DATE', y=stat_cat, title="Trend"), use_container_width=True)

# --- 5. TEAM SCANNER ---

elif mode == "Team Scanner":
    st.header("🔍 Value Scanner")
    t_list = sorted([{"id": k, "name": v['full_name']} for k, v in team_map.items()], key=lambda x: x['name'])
    sel_t = st.selectbox("Select Team", t_list, format_func=lambda x: x['name'])
    
    if st.button("📡 Scan Roster"):
        roster = api.nba.players.list(team_ids=[sel_t['id']])
        scan_data = []
        prog = st.progress(0)
        
        for i, p in enumerate(roster.data[:8]): # Limited to top 8 players to stay safe
            time.sleep(12) # Manual delay to stay under 5 req/min
            p_log = get_player_stats(p.id)
            if not p_log.empty:
                if stat_cat == "PRA": p_log['PRA'] = p_log['PTS'] + p_log['REB'] + p_log['AST']
                val = p_log[stat_cat].head(5).mean()
                scan_data.append({"Player": f"{p.first_name} {p.last_name}", "Avg": round(val, 1)})
            prog.progress((i + 1) / 8)
        
        st.table(pd.DataFrame(scan_data))
