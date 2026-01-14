import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime, timedelta
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, playernextngames, commonplayerinfo

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
    if not match: return pd.DataFrame(), None, None
    
    p_id = match[0]['id']
    try:
        # Get Player Info for Team Abbr
        info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
        team_abbr = info['TEAM_ABBREVIATION'].iloc[0]
        
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if log.empty:
            log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]
    except: return pd.DataFrame(), None, None

    log = log.rename(columns={
        'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers',
        'FGA': 'fga', 'FG_PCT': 'fg_pct', 'FTA': 'fta', 'TOV': 'tov'
    })
    log['pra'] = log['points'] + log['rebounds'] + log['assists']
    log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
    log['pps'] = log['points'] / log['fga'].replace(0, 1)
    return log, p_id, team_abbr

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
st.title("🏀 NBA Sharp Pro Hub (v2.4)")
sos_data, league_avg_drtg = load_nba_base_data()

# Initialize session state for auto-fill features
if 'auto_opp' not in st.session_state: st.session_state.auto_opp = "BOS"
if 'auto_home' not in st.session_state: st.session_state.auto_home = True
if 'auto_b2b' not in st.session_state: st.session_state.auto_b2b = False

with st.sidebar:
    st.header("🎯 Target Selection")
    search_query = st.text_input("Search Player", "Shai Gilgeous-Alexander")
    all_names = [p['full_name'] for p in players.get_players()]
    filtered = [p for p in all_names if search_query.lower() in p.lower()]
    selected_p = st.selectbox("Confirm Player", filtered if filtered else ["No Player Found"])
    
    # DATA FETCHING
    p_df, p_id, team_abbr = get_player_data(selected_p)
    
    if st.button("🚀 Auto-Fill Game Context"):
        try:
            next_g = playernextngames.PlayerNextNGames(player_id=p_id, number_of_games=1).get_data_frames()[0]
            if not next_g.empty:
                home_team = next_g['HOME_TEAM_ABBREVIATION'].iloc[0]
                visitor_team = next_g['VISITOR_TEAM_ABBREVIATION'].iloc[0]
                st.session_state.auto_home = (home_team == team_abbr)
                st.session_state.auto_opp = visitor_team if st.session_state.auto_home else home_team
                
                # B2B Check
                last_game_str = p_df['GAME_DATE'].iloc[0]
                last_date = datetime.strptime(last_game_str, '%b %d, %Y')
                next_date = datetime.strptime(next_g['GAME_DATE'].iloc[0], '%b %d, %Y')
                st.session_state.auto_b2b = ((next_date - last_date).days == 1)
                st.toast("Schedule context updated!")
        except:
            st.error("Could not fetch next game. Manual entry required.")

    st.divider()
    st.subheader("🎲 Manual Context Entry")
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    
    # Smart default for line
    default_line = p_df[stat_category].mean() if not p_df.empty else 20.5
    user_line = st.number_input(f"Sportsbook Line", value=float(round(default_line, 1)), step=0.5)
    
    market_odds = st.number_input("Market Odds (e.g. -110)", value=-110, step=5)
    selected_opp = st.selectbox("Opponent", sorted(list(sos_data.keys())), index=sorted(list(sos_data.keys())).index(st.session_state.auto_opp))
    is_home = st.toggle("Home Game", value=st.session_state.auto_home)
    is_b2b = st.toggle("Back-to-Back (Fatigue)", value=st.session_state.auto_b2b)
    star_out = st.toggle("Star Teammate Out?")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

    st.divider()
    st.subheader("🏦 Bankroll Management")
    bankroll = st.number_input("Total Bankroll ($)", value=1000, step=100)
    kelly_mode = st.select_slider("Kelly Fraction", options=["Quarter", "Half", "Full"], value="Half")
    kelly_mult = {"Quarter": 0.25, "Half": 0.5, "Full": 1.0}[kelly_mode]

# --- 4. DATA PROCESSING & MAIN DASHBOARD ---
if not p_df.empty:
    p_mean = p_df[stat_category].mean()
    sos_mult = sos_data.get(selected_opp, 1.0)
    pace_mult = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]
    sharp_lambda = p_mean * pace_mult * sos_mult * (1.15 if star_out else 1.0) * (1.03 if is_home else 0.97) * (0.95 if is_b2b else 1.0)
    over_prob = calculate_poisson_prob(sharp_lambda, user_line)

    col_main, col_side = st.columns([2, 1])

    with col_main:
        # EXTERNAL INTEL BAR
        st.info(f"🔗 **Live Intel for {selected_p}:** [Injury Report](https://www.rotowire.com/basketball/nba-lineups.php) | [Line Movement](https://www.vegasinsider.com/nba/odds/player-props/)")
        
        st.subheader("📊 Volume vs. Efficiency Matrix")
        eff_fig = go.Figure()
        eff_fig.add_trace(go.Scatter(x=p_df['usage'].head(15), y=p_df['pps'].head(15), mode='markers+text', text=p_df['points'].head(15), textposition="top center", marker=dict(size=12, color=p_df['points'], colorscale='Viridis', showscale=True), name="Recent Games"))
        eff_fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Usage Volume", yaxis_title="Efficiency (PPS)")
        st.plotly_chart(eff_fig, use_container_width=True)

        c_ins1, c_ins2 = st.columns(2)
        with c_ins1:
            recent_pps = p_df['pps'].head(5).mean()
            season_pps = p_df['pps'].mean()
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
        trend_fig.update_layout(template="plotly_dark", height=250, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(trend_fig, use_container_width=True)

        st.subheader("🎲 10,000 Game Monte Carlo Simulation")
        sim_df, sim_raw = run_monte_carlo(sharp_lambda, user_line)
        mc_fig = go.Figure()
        mc_fig.add_trace(go.Histogram(x=sim_raw, nbinsx=30, marker_color='#00ff96', opacity=0.7, histnorm='probability'))
        mc_fig.add_vline(x=user_line, line_width=3, line_dash="dash", line_color="#ff4b4b", annotation_text="LINE")
        mc_fig.update_layout(template="plotly_dark", height=250, margin=dict(l=20, r=20, t=20, b=20), xaxis_title=f"Projected {stat_category.capitalize()}", showlegend=False)
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
        
        dec_odds = american_to_decimal(market_odds)
        b = dec_odds - 1
        p = over_prob / 100
        q = 1 - p
        kelly_f = (b * p - q) / b if b > 0 else 0
        suggested_stake = max(0, kelly_f * bankroll * kelly_mult)
        
        st.divider()
        st.subheader("🎯 Kelly Recommendation")
        if edge > 0 and suggested_stake > 0:
            st.header(f"${round(suggested_stake, 2)}")
            st.caption(f"Bet {round(kelly_f * kelly_mult * 100, 2)}% of bankroll ({kelly_mode} Kelly).")
            st.success("🔥 **VALUE DETECTED**")
        else:
            st.header("$0.00")
            st.error("❌ **NO EDGE**")

    st.caption(f"v2.4 auto-context | Dataset: {len(p_df)} games | Player Team: {team_abbr}")
else:
    st.warning("Player data not found. Please confirm the name in the sidebar search.")
