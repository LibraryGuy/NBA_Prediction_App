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
    log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
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

def american_to_implied(odds):
    if odds > 0: return 100 / (odds + 100)
    else: return abs(odds) / (abs(odds) + 100)

def american_to_decimal(odds):
    if odds > 0: return (odds / 100) + 1
    else: return (100 / abs(odds)) + 1

# --- 3. UI RENDERING ---
st.title("🏀 NBA Sharp Pro Hub (v2.3)")
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
    market_odds = st.number_input("Market Odds (e.g. -110)", value=-110, step=5)
    selected_opp = st.selectbox("Opponent", sorted(list(sos_data.keys())))
    is_home = st.toggle("Home Game", value=True)
    is_b2b = st.toggle("Back-to-Back (Fatigue)")
    star_out = st.toggle("Star Teammate Out?")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

    # --- NEW: BANKROLL MANAGEMENT ---
    st.divider()
    st.subheader("🏦 Bankroll Management")
    bankroll = st.number_input("Total Bankroll ($)", value=1000, step=100)
    kelly_mode = st.select_slider("Kelly Fraction", options=["Quarter", "Half", "Full"], value="Half")
    kelly_mult = {"Quarter": 0.25, "Half": 0.5, "Full": 1.0}[kelly_mode]

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
        eff_fig.add_trace(go.Scatter(
            x=p_df['usage'].head(15), 
            y=p_df['pps'].head(15),
            mode='markers+text',
            text=p_df['points'].head(15),
            textposition="top center",
            marker=dict(size=12, color=p_df['points'], colorscale='Viridis', showscale=True),
            name="Recent Games"
        ))
        eff_fig.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Usage Volume", yaxis_title="Efficiency (PPS)")
        st.plotly_chart(eff_fig, use_container_width=True)

        c_ins1, c_ins2 = st.columns(2)
        with c_ins1:
            if recent_pps > season_pps * 1.1: st.warning("⚠️ **Efficiency Warning**: Regression likely.")
            elif recent_pps < season_pps * 0.9: st.success("✅ **Bounce Back Candidate**: Efficiency spike expected.")
            else: st.info("ℹ️ **Stable Efficiency**: Performing at career levels.")
        with c_ins2:
            avg_usage = p_df['usage'].mean()
            current_usage = p_df['usage'].head(5).mean()
            st.write(f"📈 **Volume Trend**: {'UP' if current_usage > avg_usage else 'DOWN'} ({round(current_usage-avg_usage, 1)} poss)")

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
        st.metric("Model Win Prob", f"{over_prob}%")
        
        implied_prob = american_to_implied(market_odds) * 100
        edge = over_prob - implied_prob
        
        st.divider()
        st.subheader("💰 Market Edge")
        st.metric("Market Implied", f"{round(implied_prob, 1)}%", delta=f"{round(edge, 1)}% Edge")
        
        # --- KELLY CALCULATION ---
        dec_odds = american_to_decimal(market_odds)
        b = dec_odds - 1
        p = over_prob / 100
        q = 1 - p
        
        # Kelly Formula: f* = (bp - q) / b
        if b > 0:
            kelly_f = (b * p - q) / b
        else:
            kelly_f = 0
            
        suggested_stake = max(0, kelly_f * bankroll * kelly_mult)
        
        st.divider()
        st.subheader("🎯 Kelly Recommendation")
        if edge > 0 and suggested_stake > 0:
            st.write(f"Suggested Stake ({kelly_mode} Kelly):")
            st.header(f"${round(suggested_stake, 2)}")
            st.caption(f"Bet {round(kelly_f * kelly_mult * 100, 2)}% of your ${bankroll} bankroll.")
            st.success("🔥 **VALUE DETECTED**")
        else:
            st.header("$0.00")
            st.error("❌ **NO MATHEMATICAL EDGE**")
            st.write("The model suggests the risk outweighs the potential reward at these odds.")

    st.caption(f"v2.3 bankroll edition | Dataset: {len(p_df)} games | SOS: {round(sos_mult, 2)}")
else:
    st.warning("Player data not found. Please confirm the name in the sidebar search.")
