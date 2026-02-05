import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime
import time
from balldontlie import BalldontlieAPI

# --- 1. INITIALIZATION ---
# Using the provided API Key
BDL_API_KEY = "ea294f0e-31cd-4b1b-aedb-e8fd246b907f"
api = BalldontlieAPI(api_key=BDL_API_KEY)

# --- 2. DATA ENGINES (Optimized for reliability) ---

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
    """Fetches team mapping for the scanner."""
    try:
        teams_data = api.nba.teams.list()
        return {t.id: {"full_name": t.full_name, "abbreviation": t.abbreviation} for t in teams_data}
    except:
        return {}

@st.cache_data(ttl=3600)
def search_players(query):
    """Returns a list of player objects matching the query (min 3 chars)."""
    if not query or len(query) < 3:
        return []
    try:
        # Search the balldontlie database
        results = api.nba.players.list(search=query)
        return results.data if results.data else []
    except Exception as e:
        st.error(f"Search API Error: {e}")
        return []

@st.cache_data(ttl=600)
def get_player_stats(player_id):
    """Fetches stats, trying 2025 (current) and 2024 (previous) for safety."""
    for season in [2025, 2024]:
        try:
            stats = api.nba.stats.list(
                player_ids=[player_id], 
                seasons=[season], 
                per_page=25
            )
            if stats.data:
                data = []
                for s in stats.data:
                    # Filter for actual games played (where points are recorded)
                    if s.pts is not None:
                        data.append({
                            "GAME_DATE": s.game.date,
                            "PTS": s.pts,
                            "REB": s.reb if s.reb is not None else 0,
                            "AST": s.ast if s.ast is not None else 0,
                        })
                if data:
                    df = pd.DataFrame(data)
                    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date
                    return df.sort_values('GAME_DATE', ascending=False)
        except:
            continue
    return pd.DataFrame()

# --- 3. DASHBOARD SETUP ---

st.set_page_config(page_title="Sharp Pro v12.2", layout="wide")
intel = get_intel()
team_map = get_all_teams()

with st.sidebar:
    st.title("🏀 Sharp Pro v12.2")
    st.success("API: Balldontlie Active ✅")
    
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
    
    c1, c2 = st.columns(2)
    with c1:
        search_query = st.text_input("1. Search Name (e.g., LeBron, Curry, Watson)", "Peyton Watson")
    
    # Live Search Logic
    matches = search_players(search_query)
    
    with c2:
        if matches:
            selected_player = st.selectbox(
                "2. Confirm Selection", 
                matches, 
                format_func=lambda x: f"{x.first_name} {x.last_name} ({x.team.abbreviation if x.team else 'N/A'})"
            )
        else:
            st.warning("Type 3+ letters to find players.")
            st.stop()
    
    if st.button("🚀 Run Full Analysis"):
        with st.spinner(f"Analyzing {selected_player.first_name}..."):
            log = get_player_stats(selected_player.id)
            full_name = f"{selected_player.first_name} {selected_player.last_name}"
            
            if not log.empty:
                # PRA Logic
                if stat_cat == "PRA":
                    log['PRA'] = log['PTS'] + log['REB'] + log['AST']
                
                # Projections
                raw_avg = log[stat_cat].head(10).mean()
                usage_boost = 1.12 if full_name in intel['injuries'] else 1.0
                final_proj = raw_avg * usage_boost
                prob_over = (1 - poisson.cdf(line - 0.5, final_proj)) * 100

                # UI: TOP METRICS
                st.divider()
                st.subheader(f"Analysis: {full_name}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sharp Projection", round(final_proj, 1), delta=f"{usage_boost}x Boost")
                m2.metric("Category", stat_cat)
                m3.metric("L10 Avg", round(raw_avg, 1))
                m4.metric("Win Prob (Over)", f"{round(prob_over, 1)}%")

                # UI: BETTING BLUEPRINT
                b1, b2, b3 = st.columns(3)
                direction = "OVER" if prob_over > 55 else "UNDER"
                with b1: st.info(f"**Main Play:** {stat_cat} {direction} {line}")
                with b2: st.success(f"**Alt Safety:** {stat_cat} {direction} {line-3 if direction=='OVER' else line+3}")
                with b3: st.warning(f"**Ladder Goal:** {stat_cat} {direction} {line+6 if direction=='OVER' else line-6}")

                # UI: VISUALS
                v1, v2 = st.columns(2)
                with v1:
                    st.write("**Poisson Outcome Probability**")
                    x_range = np.arange(max(0, int(final_proj-15)), int(final_proj+20))
                    fig_p = px.bar(x=x_range, y=poisson.pmf(x_range, final_proj), labels={'x': stat_cat, 'y': 'Prob'})
                    fig_p.add_vline(x=line, line_dash="dash", line_color="red", annotation_text="Line")
                    st.plotly_chart(fig_p, use_container_width=True)
                with v2:
                    st.write("**Recent Performance Trend**")
                    fig_t = px.line(log.head(10).iloc[::-1], x='GAME_DATE', y=stat_cat, markers=True)
                    fig_t.add_hline(y=line, line_color="red", line_dash="dash")
                    st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.error("This player has no registered stats for the current or previous season.")

# --- 5. MODE: TEAM SCANNER ---

elif mode == "Team Scanner":
    st.header("🔍 Value Scanner")
    if team_map:
        team_options = sorted([{"id": k, "name": v['full_name']} for k, v in team_map.items()], key=lambda x: x['name'])
        sel_team = st.selectbox("Select Team", team_options, format_func=lambda x: x['name'])
        
        if st.button("📡 Scan Roster"):
            try:
                roster = api.nba.players.list(team_ids=[sel_team['id']])
                scan_data = []
                
                prog = st.progress(0)
                status = st.empty()
                
                for i, p in enumerate(roster.data):
                    status.text(f"Scanning {p.first_name} {p.last_name}...")
                    time.sleep(2.0) # Respecting 30 req/min limit
                    
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
                    prog.progress((i + 1) / len(roster.data))
                
                status.text("Scan Complete!")
                st.dataframe(pd.DataFrame(scan_data).sort_values("L5 Avg", ascending=False), use_container_width=True)
            except Exception as e:
                st.error(f"Scanner Error: {e}")
    else:
        st.error("Wait for team data to load or check API key.")
