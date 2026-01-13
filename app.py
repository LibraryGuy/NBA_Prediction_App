import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# --- 1. SETTINGS & CONFIG ---
st.set_page_config(page_title="NBA Sharp Pro Hub", layout="wide", page_icon="🏀")

@st.cache_data(ttl=3600)
def load_nba_base_data():
    try:
        team_stats_raw = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced', season='2025-26'
        ).get_data_frames()[0]
        nba_teams = teams.get_teams()
        id_to_abbr = {t['id']: t['abbreviation'] for t in nba_teams}
        avg_drtg = team_stats_raw['DEF_RATING'].mean()
        sos_map = {id_to_abbr[row['TEAM_ID']]: (row['DEF_RATING'] * 0.8 + avg_drtg * 0.2) / avg_drtg 
                   for _, row in team_stats_raw.iterrows() if id_to_abbr.get(row['TEAM_ID'])}
        return sos_map, avg_drtg
    except:
        return {"BOS": 1.0, "GSW": 1.0, "LAL": 1.0, "OKC": 1.0}, 115.0

@st.cache_data(ttl=600)
def get_player_data(player_full_name):
    nba_players = players.get_players()
    match = [p for p in nba_players if p['full_name'].lower() == player_full_name.lower()]
    if not match:
        match = [p for p in nba_players if player_full_name.lower() in p['full_name'].lower()]
    if not match: return pd.DataFrame()
    p_id = match[0]['id']
    try:
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if log.empty:
            log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]
    except: return pd.DataFrame()

    log = log.rename(columns={
        'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers',
        'FGA': 'fga', 'FG_PCT': 'fg_pct', 'FTA': 'fta', 'TOV': 'tov'
    })
    log['pra'] = log['points'] + log['rebounds'] + log['assists']
    # Usage Proxy Calculation
    log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
    # Points Per Shot (Efficiency Metric)
    log['pps'] = log['points'] / log['fga'].replace(0, 1)
    return log

# --- 2. ADVANCED LOGIC ENGINES ---
def calculate_poisson_prob(lambda_val, line):
    return round((1 - poisson.cdf(line, lambda_val)) * 100, 1)

def run_monte_carlo(lambda_val, user_line, iterations=10000):
    simulated_games = np.random.poisson(lambda_val, iterations)
    levels = [max(0, round(l, 1)) for l in [user_line - 1, user_line, user_line + 1]]
    results = [{"Stat Level": f"{l}+", "Hit Frequency": f"{(np.sum(simulated_games >= l)/iterations)*100:.1f}%"} 
               for l in sorted(list(set(levels)))]
    return pd.DataFrame(results), simulated_games

# --- 3. UI RENDERING ---
st.title("🏀 NBA Sharp Pro Hub (v2.2)")
sos_data, league_avg_drtg = load_nba_base_data()

with st.sidebar:
    st.header("🎯 Target Selection")
    search_query = st.text_input("Search Player", "Shai Gilgeous-Alexander")
    all_names = [p['full_name'] for p in players.get_players()]
    filtered = [p for p in all_names if search_query.lower() in p.lower()]
    selected_p = st.selectbox("Confirm Player", filtered if filtered else ["No Player Found"])
    
    st.divider()
    st.subheader("🎲 Manual Context Entry")
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    user_line = st.number_input(f"Sportsbook Line", value=25.5 if stat_category=="points" else 5.5, step=0.5)
    selected_opp = st.selectbox("Opponent", sorted(list(sos_data.keys())))
    is_home = st.toggle("Home Game", value=True)
    is_b2b = st.toggle("Back-to-Back (Fatigue)")
    star_out = st.toggle("Star Teammate Out?")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

# --- 4. DATA PROCESSING ---
p_df = get_player_data(selected_p)

if not p_df.empty:
    p_mean = p_df[stat_category].mean()
    sos_mult = sos_data.get(selected_opp, 1.0)
    pace_mult = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]
    sharp_lambda = p_mean * pace_mult * sos_mult * (1.15 if star_out else 1.0) * (1.03 if is_home else 0.97) * (0.95 if is_b2b else 1.0)
    over_prob = calculate_poisson_prob(sharp_lambda, user_line)

    col_main, col_side = st.columns([2, 1])

    with col_main:
        # Efficiency suggestions logic
        recent_pps = p_df['pps'].head(5).mean()
        season_pps = p_df['pps'].mean()
        
        st.subheader("📊 Volume vs. Efficiency Matrix")
        eff_fig = go.Figure()
        # Adding game-by-game markers
        eff_fig.add_trace(go.Scatter(
            x=p_df['usage'].head(15), 
            y=p_df['pps'].head(15),
            mode='markers+text',
            text=p_df['points'].head(15),
            textposition="top center",
            marker=dict(size=12, color=p_df['points'], colorscale='Viridis', showscale=True),
            name="Recent Games"
        ))
        eff_fig.update_layout(
            template="plotly_dark", height=350,
            xaxis_title="Usage Volume (Shots + TOV + FT)",
            yaxis_title="Efficiency (Points Per Shot)",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(eff_fig, use_container_width=True)

        # Dashboard Insights
        c_ins1, c_ins2 = st.columns(2)
        with c_ins1:
            if recent_pps > season_pps * 1.1:
                st.warning("⚠️ **Efficiency Warning**: Player is scoring at a much higher rate than usual. Regression to the mean (Lower Points) is likely.")
            elif recent_pps < season_pps * 0.9:
                st.success("✅ **Bounce Back Candidate**: Player is shooting poorly compared to season average. Expect an efficiency spike (Higher Points) soon.")
            else:
                st.info("ℹ️ **Stable Efficiency**: Player is performing exactly at their expected career levels.")

        with c_ins2:
            avg_usage = p_df['usage'].mean()
            current_usage = p_df['usage'].head(5).mean()
            if current_usage > avg_usage:
                st.write(f"📈 **Volume Trend**: Usage is UP (+{round(current_usage-avg_usage, 1)} possessions)")
            else:
                st.write(f"📉 **Volume Trend**: Usage is DOWN ({round(current_usage-avg_usage, 1)} possessions)")

        st.divider()
        st.subheader("📈 Last 10 Games Performance")
        last_10 = p_df.head(10).iloc[::-1]
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=list(range(1, 11)), y=last_10[stat_category], mode='lines+markers', name='Actual', line=dict(color='#00ff96', width=3)))
        trend_fig.add_hline(y=user_line, line_dash="dash", line_color="#ff4b4b", annotation_text="Vegas Line")
        trend_fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(trend_fig, use_container_width=True)

        st.subheader("🎲 10,000 Game Monte Carlo Simulation")
        sim_df, sim_raw = run_monte_carlo(sharp_lambda, user_line)
        mc_fig = go.Figure()
        mc_fig.add_trace(go.Histogram(x=sim_raw, nbinsx=30, marker_color='#00ff96', opacity=0.7, histnorm='probability'))
        mc_fig.add_vline(x=user_line, line_width=3, line_dash="dash", line_color="#ff4b4b", annotation_text="LINE")
        mc_fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title=f"Projected {stat_category.capitalize()}", showlegend=False)
        st.plotly_chart(mc_fig, use_container_width=True)

    with col_side:
        st.subheader("📊 Model Output")
        st.metric("Sharp Projection", round(sharp_lambda, 1))
        st.metric("Win Prob (Over)", f"{over_prob}%")
        st.divider()
        if over_prob > 60: st.success("🔥 **STRONG VALUE DETECTED: OVER**")
        elif over_prob < 40: st.error("❄️ **STRONG VALUE DETECTED: UNDER**")
        else: st.info("⚖️ **NEUTRAL: NO EDGE DETECTED**")

    st.caption(f"v2.2 efficiency map | Dataset: {len(p_df)} games | SOS: {round(sos_mult, 2)}")
else:
    st.warning("Player data not found. Please confirm the name in the sidebar search.")
