import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from scipy.stats import poisson
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, commonteamroster, boxscoretraditionalv2

# --- 1. CORE DATA & ENHANCED LOGIC ---
@st.cache_data(ttl=3600)
def load_nba_universe():
    all_30 = {
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
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced', season='2025-26').get_data_frames()[0]
        avg_d = stats['DEF_RATING'].mean()
        sos = { [k for k,v in all_30.items() if v==row['TEAM_ID']][0]: row['DEF_RATING']/avg_d for _,row in stats.iterrows() }
        return sos, all_30
    except:
        return {k: 1.0 for k in all_30.keys()}, all_30

@st.cache_data(ttl=600)
def get_player_data(player_input, is_id=False):
    if not is_id:
        nba_players = players.get_players()
        match = [p for p in nba_players if player_input.lower() in p['full_name'].lower()]
        if not match: return pd.DataFrame(), None, None, None
        p_id = match[0]['id']
    else:
        p_id = player_input
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
        team_abbr = info['TEAM_ABBREVIATION'].iloc[0]
        pos = info['POSITION'].iloc[0]
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if not log.empty:
            log = log.rename(columns={'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'FGA': 'fga', 'FTA': 'fta', 'TOV': 'tov'})
            log['pra'] = log['points'] + log['rebounds'] + log['assists']
            log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
            log['pps'] = log['points'] / log['fga'].replace(0, 1)
        return log, p_id, team_abbr, pos
    except: return pd.DataFrame(), None, None, None

def calculate_sharp_lambda(p_mean, pace, sos, star, home, b2b, injury_impact):
    return p_mean * pace * sos * (1.15 if star else 1.0) * (1.03 if home else 0.97) * (0.95 if b2b else 1.0) * injury_impact

# --- 2. UI LAYOUT & SIDEBAR ---
st.set_page_config(page_title="Sharp Pro Hub v4.1", layout="wide", page_icon="📈")
sos_data, team_map = load_nba_universe()

with st.sidebar:
    st.header("🏦 Bankroll & Odds")
    total_purse = st.number_input("Starting Purse ($)", value=1000)
    kelly_multiplier = st.slider("Kelly Fraction", 0.1, 1.0, 0.5)
    st.divider()
    
    st.header("🩹 Injury Impact")
    injury_pos = st.selectbox("Key Player Out (Position)", ["None", "Point Guard", "Center", "Wing/Forward"])
    impact_map = {"None": 1.0, "Point Guard": 1.12, "Center": 1.08, "Wing/Forward": 1.05}
    current_impact = impact_map[injury_pos]
    
    st.header("🎯 System Mode")
    mode = st.radio("Switch View", ["Single Player", "Team Scout Radar", "Parlay Builder", "Box Score Scraper"])
    st.divider()
    stat_cat = st.selectbox("Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    selected_opp = st.selectbox("Opponent", sorted(list(team_map.keys())))
    is_home = st.toggle("Home Game", value=True)
    pace_mult = st.select_slider("Pace", options=[0.92, 1.0, 1.08], value=1.0)

# --- 3. SINGLE PLAYER DASHBOARD (VISUALS RESTORED) ---
if mode == "Single Player":
    search_q = st.text_input("Player Search", "Jalen Brunson")
    p_df, p_id, team_abbr, p_pos = get_player_data(search_q)
    
    if not p_df.empty:
        p_mean = p_df[stat_cat].mean()
        sharp_lambda = calculate_sharp_lambda(p_mean, pace_mult, sos_data.get(selected_opp, 1.0), False, is_home, False, current_impact)
        
        # Line Movement Inputs
        c_line1, c_line2 = st.columns(2)
        opening_line = c_line1.number_input("Opening Line", value=float(round(p_mean, 1)))
        current_line = c_line2.number_input("Current Line", value=float(round(p_mean, 1)))

        main_col, side_col = st.columns([2, 1])
        
        with main_col:
            # CHART 1: Last 10 Trend
            st.subheader(f"📈 {stat_cat.title()} Trend (Last 10)")
            trend_fig = go.Figure(go.Scatter(x=list(range(1, 11)), y=p_df[stat_cat].head(10).iloc[::-1], mode='lines+markers', line=dict(color='#00ff96', width=3)))
            trend_fig.add_hline(y=current_line, line_dash="dash", line_color="red", annotation_text="Market Line")
            trend_fig.update_layout(template="plotly_dark", height=250, margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(trend_fig, use_container_width=True)

            # CHART 2: Efficiency Scatter Plot (RE-ADDED)
            st.subheader("📊 Efficiency Matrix (Usage vs PPS)")
            eff_fig = go.Figure(go.Scatter(
                x=p_df['usage'].head(15), y=p_df['pps'].head(15), 
                mode='markers+text', text=p_df['points'],
                marker=dict(size=14, color=p_df['points'], colorscale='Plasma', showscale=True)
            ))
            eff_fig.update_layout(template="plotly_dark", xaxis_title="Usage Volume", yaxis_title="Points Per Shot", height=300)
            st.plotly_chart(eff_fig, use_container_width=True)

        with side_col:
            # CHART 3: Monte Carlo Poisson (RE-ADDED)
            st.subheader("🎲 Outcome Probability")
            win_p = (1 - poisson.cdf(current_line - 0.5, sharp_lambda))
            st.metric("Win Probability", f"{round(win_p * 100, 1)}%")
            
            sims = np.random.poisson(sharp_lambda, 10000)
            dist_fig = go.Figure(go.Histogram(x=sims, nbinsx=15, marker_color='#00ff96', opacity=0.6))
            dist_fig.add_vline(x=current_line, line_color="red", line_width=3)
            dist_fig.update_layout(template="plotly_dark", height=200, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            st.plotly_chart(dist_fig, use_container_width=True)
            
            # Betting Suggestion
            dec_odds = 1.91 # Standard -110
            k_f = ((dec_odds - 1) * win_p - (1 - win_p)) / (dec_odds - 1)
            stake = max(0, round(k_f * total_purse * kelly_multiplier, 2))
            st.success(f"**Recommended Stake:** ${stake}")
            st.info(f"Reflects {injury_pos} injury boost.")

# --- 4. BOX SCORE SCRAPER ---
elif mode == "Box Score Scraper":
    st.subheader("📋 Last Game Deep-Dive")
    search_q = st.text_input("Enter Player Name", "Jalen Brunson")
    p_df, p_id, team_abbr, p_pos = get_player_data(search_q)
    if p_id:
        last_game_id = p_df.iloc[0]['Game_ID']
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=last_game_id).get_data_frames()[0]
        player_stats = box[box['PLAYER_ID'] == p_id]
        st.table(player_stats[['MIN', 'PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT']])

# --- 5. PARLAY BUILDER ---
elif mode == "Parlay Builder":
    st.subheader("🔗 Multi-Leg Optimizer")
    if 'radar_results' not in st.session_state:
        st.warning("Please run a 'Team Scout Radar' scan first.")
    else:
        selections = st.multiselect("Select Legs", st.session_state['radar_results']['Player'].tolist())
        if selections:
            total_p = 1.0
            for player in selections:
                total_p *= (st.session_state['radar_results'].loc[st.session_state['radar_results']['Player'] == player, 'Hit%'].iloc[0] / 100)
            st.metric("Combined Win Prob", f"{round(total_p * 100, 1)}%")

# --- 6. TEAM SCOUT RADAR ---
elif mode == "Team Scout Radar":
    team_to_scout = st.selectbox("Select Team", sorted(list(team_map.keys())))
    if st.button(f"🚀 Scan {team_to_scout} Roster"):
        results = []
        roster = commonteamroster.CommonTeamRoster(team_id=team_map[team_to_scout]).get_data_frames()[0].head(10)
        for _, row in roster.iterrows():
            p_log, _, _, _ = get_player_data(row['PLAYER_ID'], is_id=True)
            if not p_log.empty:
                m = p_log[stat_cat].mean()
                proj = calculate_sharp_lambda(m, pace_mult, sos_data.get(selected_opp, 1.0), False, is_home, False, current_impact)
                prob = round((1 - poisson.cdf(m - 0.5, proj)) * 100, 1)
                results.append({"Player": row['PLAYER'], "Avg": round(m,1), "Proj": round(proj,1), "Edge": round(proj-m,1), "Hit%": prob})
        st.session_state['radar_results'] = pd.DataFrame(results)
        st.table(st.session_state['radar_results'])
