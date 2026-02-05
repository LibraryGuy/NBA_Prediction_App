import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import time
from balldontlie import BalldontlieAPI

# --- 1. INITIALIZATION ---
# Use Streamlit secrets or paste your key here
BDL_API_KEY = st.sidebar.text_input("Balldontlie API Key", type="password")
if not BDL_API_KEY:
    st.warning("Please enter your Balldontlie API key in the sidebar to begin.")
    st.stop()

api = BalldontlieAPI(api_key=BDL_API_KEY)

# --- 2. DATA ENGINES ---

@st.cache_data(ttl=3600)
def get_intel():
    return {
        "injuries": ["Nikola Jokic", "Kevin Durant", "Joel Embiid", "Ja Morant"],
        "ref_bias": {
            "Scott Foster": {"type": "Under", "impact": 0.96},
            "Marc Davis": {"type": "Over", "impact": 1.05}
        }
    }

@st.cache_data(ttl=86400)
def get_all_teams():
    try:
        teams_data = api.nba.teams.list()
        return {t.id: {"full_name": t.full_name, "abbreviation": t.abbreviation} for t in teams_data}
    except:
        return {}

@st.cache_data(ttl=3600)
def search_players(query):
    """Returns a list of player objects matching the query."""
    try:
        results = api.nba.players.list(search=query)
        return results.data if results.data else []
    except:
        return []

@st.cache_data(ttl=600)
def get_player_stats(player_id, season=2024):
    try:
        stats = api.nba.stats.list(player_ids=[player_id], seasons=[season], per_page=20)
        data = []
        for s in stats.data:
            data.append({
                "GAME_DATE": s.game.date,
                "PTS": s.pts if s.pts is not None else 0,
                "REB": s.reb if s.reb is not None else 0,
                "AST": s.ast if s.ast is not None else 0,
            })
        df = pd.DataFrame(data)
        if not df.empty:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date
            df = df.sort_values('GAME_DATE', ascending=False)
        return df
    except:
        return pd.DataFrame()

# --- 3. DASHBOARD SETUP ---

st.set_page_config(page_title="Sharp Pro v12.1", layout="wide")
intel = get_intel()
team_map = get_all_teams()

with st.sidebar:
    st.title("🏀 Sharp Pro v12.1")
    mode = st.radio("Navigation", ["Single Player Analysis", "Team Scanner"])
    st.divider()
    stat_cat = st.selectbox("Stat Category", ["PTS", "REB", "AST", "PRA"])
    line = st.number_input("Sportsbook Line", value=22.5, step=0.5)
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.rerun()

# --- 4. MODE: SINGLE PLAYER ANALYSIS ---

if mode == "Single Player Analysis":
    st.header("👤 Player Matchup Engine")
    
    c1, c2 = st.columns(2)
    with c1:
        search_query = st.text_input("1. Search Name", "Peyton Watson")
    
    # SEARCH LOGIC
    matches = search_players(search_query) if search_query else []
    
    with c2:
        if matches:
            selected_player_obj = st.selectbox(
                "2. Confirm Selection", 
                matches, 
                format_func=lambda x: f"{x.first_name} {x.last_name} ({x.team.abbreviation if x.team else 'N/A'})"
            )
        else:
            st.warning("No players found.")
            st.stop()

    if st.button("🚀 Run Full Analysis"):
        with st.spinner("Decoding stats..."):
            log = get_player_stats(selected_player_obj.id)
            full_name = f"{selected_player_obj.first_name} {selected_player_obj.last_name}"
            
            if not log.empty:
                if stat_cat == "PRA":
                    log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                
                raw_avg = log[stat_cat].head(10).mean()
                usage_boost = 1.12 if full_name in intel['injuries'] else 1.0
                final_proj = raw_avg * usage_boost
                prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

                st.divider()
                st.subheader(f"Analysis: {full_name}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sharp Projection", round(final_proj, 1), delta=f"{usage_boost}x Boost")
                m2.metric("Stat Category", stat_cat)
                m3.metric("L10 Avg", round(raw_avg, 1))
                m4.metric("Win Prob (Over)", f"{round(prob_over, 1)}%")

                # Visuals
                v1, v2 = st.columns(2)
                with v1:
                    x_range = np.arange(max(0, int(final_proj-15)), int(final_proj+20))
                    fig_p = px.bar(x=x_range, y=poisson.pmf(x_range, final_proj), title="Outcome Probability")
                    fig_p.add_vline(x=line, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_p, use_container_width=True)
                with v2:
                    fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True, title="Recent Trend")
                    fig_t.add_hline(y=line, line_color="red", line_dash="dash")
                    st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.error("No stats found for this season.")

# --- 5. MODE: TEAM SCANNER ---

elif mode == "Team Scanner":
    st.header("🔍 Value Scanner")
    team_options = sorted([{"id": k, "name": v['full_name']} for k, v in team_map.items()], key=lambda x: x['name'])
    sel_team = st.selectbox("Select Team", team_options, format_func=lambda x: x['name'])
    
    if st.button("📡 Scan Roster"):
        try:
            roster = api.nba.players.list(team_ids=[sel_team['id']])
            scan_data = []
            prog = st.progress(0)
            
            for i, p in enumerate(roster.data):
                time.sleep(2.0) # Rate limit respect
                p_log = get_player_stats(p.id)
                if not p_log.empty:
                    if stat_cat == "PRA": p_log['PRA'] = p_log['PTS'] + p_log['REB'] + p_log['AST']
                    avg = p_log[stat_cat].head(5).mean()
                    prob = (1 - poisson.cdf(line - 0.5, avg)) * 100
                    scan_data.append({
                        "Player": f"{p.first_name} {p.last_name}", 
                        "L5 Avg": round(avg, 1), 
                        "Prob (Over)": f"{round(prob, 1)}%",
                        "Signal": "🔥 HIGH" if prob > 70 else ("❄️ LOW" if prob < 30 else "Neutral")
                    })
                prog.progress((i + 1) / len(roster.data))
            
            st.dataframe(pd.DataFrame(scan_data).sort_values("L5 Avg", ascending=False), use_container_width=True)
        except Exception as e:
            st.error(f"Scanner Error: {e}")
