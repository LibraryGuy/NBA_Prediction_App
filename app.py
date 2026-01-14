import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from scipy.stats import poisson
from datetime import datetime, timedelta
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, playernextngames, commonplayerinfo, commonteamroster

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
        abbr_to_id = {t['abbreviation']: t['id'] for t in nba_teams}
        avg_drtg = team_stats_raw['DEF_RATING'].mean()
        sos_map = {id_to_abbr[row['TEAM_ID']]: (row['DEF_RATING'] * 0.8 + avg_drtg * 0.2) / avg_drtg 
                   for _, row in team_stats_raw.iterrows() if id_to_abbr.get(row['TEAM_ID'])}
        return sos_map, abbr_to_id
    except:
        return {"BOS": 1.0, "GSW": 1.0, "LAL": 1.0, "OKC": 1.0}, {"BOS": 1610612738, "GSW": 1610612744}

@st.cache_data(ttl=600)
def get_player_data(player_input, is_id=False):
    """Flexible fetcher: supports name search or direct ID fetch for roster scans."""
    if not is_id:
        nba_players = players.get_players()
        match = [p for p in nba_players if p['full_name'].lower() == player_input.lower()]
        if not match:
            match = [p for p in nba_players if player_input.lower() in p['full_name'].lower()]
        if not match: return pd.DataFrame(), None, None
        p_id = match[0]['id']
    else:
        p_id = player_input
    
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
        team_abbr = info['TEAM_ABBREVIATION'].iloc[0]
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if log.empty:
            log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]
    except: return pd.DataFrame(), None, None

    log = log.rename(columns={'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'FGA': 'fga', 'FG_PCT': 'fg_pct', 'FTA': 'fta', 'TOV': 'tov'})
    log['pra'] = log['points'] + log['rebounds'] + log['assists']
    log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
    log['pps'] = log['points'] / log['fga'].replace(0, 1)
    return log, p_id, team_abbr

# --- 2. ADVANCED LOGIC ENGINES ---
def calculate_sharp_lambda(p_mean, pace_mult, sos_mult, star_out, is_home, is_b2b):
    return p_mean * pace_mult * sos_mult * (1.15 if star_out else 1.0) * (1.03 if is_home else 0.97) * (0.95 if is_b2b else 1.0)

def calculate_poisson_prob(lambda_val, line):
    return round((1 - poisson.cdf(line, lambda_val)) * 100, 1)

def run_monte_carlo(lambda_val, user_line, iterations=10000):
    simulated_games = np.random.poisson(lambda_val, iterations)
    levels = [max(0, round(l, 1)) for l in [user_line - 1, user_line, user_line + 1]]
    results = [{"Stat Level": f"{l}+", "Hit Frequency": f"{(np.sum(simulated_games >= l)/iterations)*100:.1f}%"} 
               for l in sorted(list(set(levels)))]
    return pd.DataFrame(results), simulated_games

def american_to_implied(odds):
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

def american_to_decimal(odds):
    return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1

# --- 3. UI RENDERING & SIDEBAR ---
st.title("🏀 NBA Sharp Pro Hub (v2.5)")
sos_data, abbr_to_id = load_nba_base_data()

# Initialize session state for auto-fill features
if 'auto_opp' not in st.session_state: st.session_state.auto_opp = "BOS"
if 'auto_home' not in st.session_state: st.session_state.auto_home = True
if 'auto_b2b' not in st.session_state: st.session_state.auto_b2b = False

with st.sidebar:
    st.header("🎮 Analysis Mode")
    mode = st.radio("Switch View", ["Single Player", "Team Scout Radar"])
    
    st.divider()
    if mode == "Single Player":
        search_query = st.text_input("Search Player", "Shai Gilgeous-Alexander")
        all_names = [p['full_name'] for p in players.get_players()]
        filtered = [p for p in all_names if search_query.lower() in p.lower()]
        selected_p = st.selectbox("Confirm Player", filtered if filtered else ["No Player Found"])
        p_df, p_id, team_abbr = get_player_data(selected_p)
    else:
        selected_team_abbr = st.selectbox("Select Team to Scout", sorted(list(abbr_to_id.keys())))
        p_df = pd.DataFrame() # Reset for team mode
    
    # Global Context
    st.subheader("🎲 Game Context")
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    selected_opp = st.selectbox("Opponent", sorted(list(sos_data.keys())), index=sorted(list(sos_data.keys())).index(st.session_state.auto_opp))
    is_home = st.toggle("Home Game", value=st.session_state.auto_home)
    is_b2b = st.toggle("Back-to-Back", value=st.session_state.auto_b2b)
    star_out = st.toggle("Star Teammate Out?")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

    if mode == "Single Player" and not p_df.empty:
        if st.button("🚀 Auto-Fill Game Context"):
            try:
                next_g = playernextngames.PlayerNextNGames(player_id=p_id, number_of_games=1).get_data_frames()[0]
                if not next_g.empty:
                    st.session_state.auto_home = (next_g['HOME_TEAM_ABBREVIATION'].iloc[0] == team_abbr)
                    st.session_state.auto_opp = next_g['VISITOR_TEAM_ABBREVIATION'].iloc[0] if st.session_state.auto_home else next_g['HOME_TEAM_ABBREVIATION'].iloc[0]
                    last_date = datetime.strptime(p_df['GAME_DATE'].iloc[0], '%b %d, %Y')
                    next_date = datetime.strptime(next_g['GAME_DATE'].iloc[0], '%b %d, %Y')
                    st.session_state.auto_b2b = ((next_date - last_date).days == 1)
                    st.rerun()
            except: st.error("Schedule fetch failed.")

# --- 4. MAIN LOGIC BRANCHING ---

# SHARED MULTIPLIERS
sos_mult = sos_data.get(selected_opp, 1.0)
pace_mult = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]

if mode == "Single Player":
    if not p_df.empty:
        p_mean = p_df[stat_category].mean()
        sharp_lambda = calculate_sharp_lambda(p_mean, pace_mult, sos_mult, star_out, is_home, is_b2b)
        
        # Sidebar Line Entry (Specific to Single Player)
        with st.sidebar:
            st.divider()
            default_line = p_mean
            user_line = st.number_input(f"Sportsbook Line", value=float(round(default_line, 1)), step=0.5)
            market_odds = st.number_input("Market Odds", value=-110, step=5)
            bankroll = st.number_input("Bankroll ($)", value=1000)
            kelly_mult = {"Quarter": 0.25, "Half": 0.5, "Full": 1.0}[st.select_slider("Kelly", ["Quarter", "Half", "Full"], "Half")]

        # Layout
        col_main, col_side = st.columns([2, 1])
        with col_main:
            st.info(f"🔗 **Intel:** [Injuries](https://www.rotowire.com/basketball/nba-lineups.php) | [Line Movement](https://www.vegasinsider.com/nba/odds/player-props/)")
            
            # Efficiency/Volume Matrix
            st.subheader("📊 Performance Matrix")
            eff_fig = go.Figure()
            eff_fig.add_trace(go.Scatter(x=p_df['usage'].head(15), y=p_df['pps'].head(15), mode='markers+text', text=p_df['points'].head(15), textposition="top center", marker=dict(size=12, color=p_df['points'], colorscale='Viridis')))
            eff_fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Usage", yaxis_title="Efficiency")
            st.plotly_chart(eff_fig, use_container_width=True)

            # Last 10 Trend
            last_10 = p_df.head(10).iloc[::-1]
            trend_fig = go.Figure()
            trend_fig.add_trace(go.Scatter(x=list(range(1, 11)), y=last_10[stat_category], mode='lines+markers', line=dict(color='#00ff96', width=3)))
            trend_fig.add_hline(y=user_line, line_dash="dash", line_color="#ff4b4b")
            trend_fig.update_layout(template="plotly_dark", height=250, title="Last 10 Games Trend")
            st.plotly_chart(trend_fig, use_container_width=True)

            # Monte Carlo
            sim_df, sim_raw = run_monte_carlo(sharp_lambda, user_line)
            mc_fig = go.Figure(go.Histogram(x=sim_raw, marker_color='#00ff96', opacity=0.7, histnorm='probability'))
            mc_fig.add_vline(x=user_line, line_dash="dash", line_color="#ff4b4b")
            mc_fig.update_layout(template="plotly_dark", height=250, title="10,000 Game Monte Carlo")
            st.plotly_chart(mc_fig, use_container_width=True)

        with col_side:
            st.subheader("🎯 Sharp Output")
            st.metric("Projection", round(sharp_lambda, 1))
            over_prob = calculate_poisson_prob(sharp_lambda, user_line)
            st.metric("Win Prob (Over)", f"{over_prob}%")
            
            implied_prob = american_to_implied(market_odds) * 100
            edge = over_prob - implied_prob
            st.metric("Market Implied", f"{round(implied_prob, 1)}%", delta=f"{round(edge, 1)}% Edge")
            
            # Kelly Logic
            dec_odds = american_to_decimal(market_odds)
            kelly_f = ((dec_odds-1)*(over_prob/100) - (1-(over_prob/100))) / (dec_odds-1) if dec_odds > 1 else 0
            stake = max(0, kelly_f * bankroll * kelly_mult)
            st.divider()
            if edge > 0 and stake > 0:
                st.header(f"${round(stake, 2)}")
                st.success("🔥 VALUE DETECTED")
            else:
                st.header("$0.00")
                st.error("❌ NO EDGE")

else:
    # --- TEAM SCOUT RADAR MODE ---
    st.subheader(f"📡 {selected_team_abbr} Roster Radar")
    st.write("Analyzing current context against season averages to find the best value targets.")
    
    if st.button("🚀 Run Team Scan"):
        t_id = abbr_to_id[selected_team_abbr]
        # Fetching roster (Top 8 for speed/relevance)
        with st.status("Fetching Roster Data...", expanded=True) as status:
            roster = commonteamroster.CommonTeamRoster(team_id=t_id).get_data_frames()[0].head(8)
            results = []
            
            for idx, row in roster.iterrows():
                p_name = row['PLAYER']
                p_id = row['PLAYER_ID']
                st.write(f"Analyzing {p_name}...")
                
                # Fetch individual data (is_id=True)
                t_df, _, _ = get_player_data(p_id, is_id=True)
                if not t_df.empty:
                    t_mean = t_df[stat_category].mean()
                    t_proj = calculate_sharp_lambda(t_mean, pace_mult, sos_mult, star_out, is_home, is_b2b)
                    diff = t_proj - t_mean
                    
                    results.append({
                        "Player": p_name,
                        "Season Avg": round(t_mean, 1),
                        "Sharp Proj": round(t_proj, 1),
                        "Bump": round(diff, 1),
                        "Status": "📈 High" if diff > 1.5 else "📉 Low" if diff < -1.5 else "⚖️ Neutral"
                    })
                time.sleep(0.2) # API Rate Limit protection
            status.update(label="Scan Complete!", state="complete", expanded=False)

        # Display Results
        res_df = pd.DataFrame(results).sort_values(by="Bump", ascending=False)
        st.dataframe(res_df.style.background_gradient(subset=['Bump'], cmap='RdYlGn'), use_container_width=True)
        
        # Visualization
        radar_fig = go.Figure(go.Bar(x=res_df['Player'], y=res_df['Bump'], marker_color='#00ff96'))
        radar_fig.update_layout(template="plotly_dark", title=f"Contextual Impact on {stat_category.upper()}", yaxis_title="Proj vs Avg Diff")
        st.plotly_chart(radar_fig, use_container_width=True)
        st.balloons()
