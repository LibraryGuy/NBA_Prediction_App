import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from scipy.stats import poisson
from datetime import datetime
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, playernextngames, commonplayerinfo, commonteamroster

# --- 1. SETTINGS & CONFIG ---
st.set_page_config(page_title="NBA Sharp Pro Hub", layout="wide", page_icon="🏀")

@st.cache_data(ttl=3600)
def load_nba_base_data():
    # Complete list of all 30 NBA Team IDs as a ultimate safety net
    all_nba_teams = {
        'ATL': 1610612737, 'BOS': 1610612738, 'CLE': 1610612739, 'NOP': 1610612740,
        'CHI': 1610612741, 'DAL': 1610612742, 'DEN': 1610612743, 'GSW': 1610612744,
        'HOU': 1610612745, 'LAC': 1610612746, 'LAL': 1610612747, 'MIA': 1610612748,
        'MIL': 1610612749, 'MIN': 1610612750, 'BKN': 1610612751, 'NYK': 1610612752,
        'ORL': 1610612753, 'IND': 1610612754, 'PHI': 1610612755, 'PHX': 1610612756,
        'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'OKC': 1610612760,
        'TOR': 1610612761, 'UTA': 1610612762, 'MEM': 1610612763, 'WAS': 1610612764,
        'DET': 1610612765, 'CHA': 1610612766
    }
    
    try:
        # Attempt to get data for the most recent completed/active season (2024-25)
        # 2025-26 is often too early for the stats API to respond with valid JSON
        team_stats_raw = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced', season='2024-25'
        ).get_data_frames()[0]
        
        nba_teams = teams.get_teams()
        id_to_abbr = {t['id']: t['abbreviation'] for t in nba_teams}
        avg_drtg = team_stats_raw['DEF_RATING'].mean()
        
        sos_map = {id_to_abbr[row['TEAM_ID']]: (row['DEF_RATING'] * 0.8 + avg_drtg * 0.2) / avg_drtg 
                   for _, row in team_stats_raw.iterrows() if id_to_abbr.get(row['TEAM_ID'])}
        
        # If SOS map is empty, trigger fallback
        if not sos_map: raise ValueError("Empty Stats")
        return sos_map, all_nba_teams
        
    except Exception as e:
        # Fallback logic: Use all 30 teams with a neutral 1.0 SOS multiplier
        sos_map = {abbr: 1.0 for abbr in all_nba_teams.keys()}
        return sos_map, all_nba_teams

@st.cache_data(ttl=600)
def get_player_data(player_input, is_id=False):
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
        # Check current season, fall back to previous if empty
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if log.empty:
            log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]
    except: return pd.DataFrame(), None, None

    if not log.empty:
        log = log.rename(columns={'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'FGA': 'fga', 'FG_PCT': 'fg_pct', 'FTA': 'fta', 'TOV': 'tov'})
        log['pra'] = log['points'] + log['rebounds'] + log['assists']
        log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
        log['pps'] = log['points'] / log['fga'].replace(0, 1)
    return log, p_id, team_abbr

# --- 2. ENGINES ---
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

# --- 3. UI RENDERING ---
st.title("🏀 NBA Sharp Pro Hub (v2.8)")
sos_data, abbr_to_id = load_nba_base_data()

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
        p_df = pd.DataFrame() 

    st.subheader("🎲 Game Context")
    stat_category = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    
    opp_options = sorted(list(sos_data.keys()))
    # Improved index lookup safety
    try:
        opp_idx = opp_options.index(st.session_state.auto_opp)
    except:
        opp_idx = 0
        
    selected_opp = st.selectbox("Opponent", opp_options, index=opp_idx)
    is_home = st.toggle("Home Game", value=st.session_state.auto_home)
    is_b2b = st.toggle("Back-to-Back", value=st.session_state.auto_b2b)
    star_out = st.toggle("Star Teammate Out?")
    pace_script = st.select_slider("Expected Pace", options=["Snail", "Balanced", "Track Meet"], value="Balanced")

    if mode == "Single Player" and not p_df.empty:
        st.divider()
        st.subheader("🏦 Bankroll Management")
        user_line = st.number_input(f"Sportsbook Line", value=float(round(p_df[stat_category].mean(), 1)), step=0.5)
        market_odds = st.number_input("Market Odds", value=-110, step=5)
        bankroll = st.number_input("Total Bankroll ($)", value=1000, step=100)
        kelly_mode = st.select_slider("Kelly Fraction", options=["Quarter", "Half", "Full"], value="Half")
        kelly_mult = {"Quarter": 0.25, "Half": 0.5, "Full": 1.0}[kelly_mode]

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

# --- 4. MAIN DASHBOARD ---
sos_mult = sos_data.get(selected_opp, 1.0)
pace_mult = {"Snail": 0.92, "Balanced": 1.0, "Track Meet": 1.08}[pace_script]

if mode == "Single Player" and not p_df.empty:
    p_mean = p_df[stat_category].mean()
    sharp_lambda = calculate_sharp_lambda(p_mean, pace_mult, sos_mult, star_out, is_home, is_b2b)
    over_prob = calculate_poisson_prob(sharp_lambda, user_line)

    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.info(f"🔗 **Live Intel for {selected_p}:** [Injury Report](https://www.rotowire.com/basketball/nba-lineups.php) | [Line Movement](https://www.vegasinsider.com/nba/odds/player-props/)")
        
        # Performance Viz
        st.subheader("📊 Volume vs. Efficiency Matrix")
        eff_fig = go.Figure(go.Scatter(x=p_df['usage'].head(15), y=p_df['pps'].head(15), mode='markers+text', text=p_df['points'].head(15), textposition="top center", marker=dict(size=12, color=p_df['points'], colorscale='Viridis', showscale=True)))
        eff_fig.update_layout(template="plotly_dark", height=300, xaxis_title="Usage Volume", yaxis_title="Efficiency (PPS)")
        st.plotly_chart(eff_fig, use_container_width=True)

        st.subheader("📈 Last 10 Games Performance")
        last_10 = p_df.head(10).iloc[::-1]
        trend_fig = go.Figure(go.Scatter(x=list(range(1, 11)), y=last_10[stat_category], mode='lines+markers', line=dict(color='#00ff96', width=3)))
        trend_fig.add_hline(y=user_line, line_dash="dash", line_color="#ff4b4b", annotation_text="Vegas Line")
        trend_fig.update_layout(template="plotly_dark", height=250)
        st.plotly_chart(trend_fig, use_container_width=True)

        st.subheader("🎲 Monte Carlo Simulation")
        sim_df, sim_raw = run_monte_carlo(sharp_lambda, user_line)
        mc_fig = go.Figure(go.Histogram(x=sim_raw, nbinsx=30, marker_color='#00ff96', opacity=0.7, histnorm='probability'))
        mc_fig.add_vline(x=user_line, line_width=3, line_dash="dash", line_color="#ff4b4b")
        mc_fig.update_layout(template="plotly_dark", height=250, xaxis_title=f"Projected {stat_category.capitalize()}")
        st.plotly_chart(mc_fig, use_container_width=True)

    with col_side:
        st.subheader("📊 Model Output")
        st.metric("Sharp Projection", round(sharp_lambda, 1))
        st.metric("Model Win Prob", f"{over_prob}%")
        
        implied_prob = american_to_implied(market_odds) * 100
        edge = over_prob - implied_prob
        st.metric("Market Implied", f"{round(implied_prob, 1)}%", delta=f"{round(edge, 1)}% Edge")
        
        dec_odds = american_to_decimal(market_odds)
        kelly_f = ((dec_odds - 1) * (over_prob / 100) - (1 - (over_prob / 100))) / (dec_odds - 1) if dec_odds > 1 else 0
        stake = max(0, kelly_f * bankroll * kelly_mult)
        
        st.divider()
        st.subheader("🎯 Kelly Recommendation")
        if edge > 0 and stake > 0:
            st.header(f"${round(stake, 2)}")
            st.success("🔥 VALUE DETECTED")
        else:
            st.header("$0.00")
            st.error("❌ NO EDGE")

elif mode == "Team Scout Radar":
    st.subheader(f"📡 {selected_team_abbr} Roster Radar")
    if st.button("🚀 Run Team Scan"):
        t_id = abbr_to_id.get(selected_team_abbr)
        if t_id:
            with st.status(f"Scanning {selected_team_abbr} Roster...", expanded=True) as status:
                # Top 10 roster members
                roster_data = commonteamroster.CommonTeamRoster(team_id=t_id).get_data_frames()[0].head(10)
                results = []
                for _, row in roster_data.iterrows():
                    p_name, p_id_roster = row['PLAYER'], row['PLAYER_ID']
                    t_df, _, _ = get_player_data(p_id_roster, is_id=True)
                    if not t_df.empty:
                        t_mean = t_df[stat_category].mean()
                        t_proj = calculate_sharp_lambda(t_mean, pace_mult, sos_mult, star_out, is_home, is_b2b)
                        results.append({"Player": p_name, "Avg": round(t_mean, 1), "Proj": round(t_proj, 1), "Bump": round(t_proj - t_mean, 1)})
                    time.sleep(0.15)
                status.update(label="Radar Scan Complete!", state="complete")

            res_df = pd.DataFrame(results).sort_values(by="Bump", ascending=False)
            try:
                st.dataframe(res_df.style.background_gradient(subset=['Bump'], cmap='RdYlGn'), use_container_width=True)
            except:
                st.dataframe(res_df, use_container_width=True)
            
            radar_fig = go.Figure(go.Bar(x=res_df['Player'], y=res_df['Bump'], marker_color='#00ff96'))
            radar_fig.update_layout(template="plotly_dark", title=f"Situational Impact: {selected_team_abbr}")
            st.plotly_chart(radar_fig, use_container_width=True)
