import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from scipy.stats import poisson
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, commonteamroster

# --- 1. CORE DATA ENGINE ---
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
        if not match: return pd.DataFrame(), None, None
        p_id = match[0]['id']
    else:
        p_id = player_input
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
        team_abbr = info['TEAM_ABBREVIATION'].iloc[0]
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if not log.empty:
            log = log.rename(columns={'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'FGA': 'fga', 'FTA': 'fta', 'TOV': 'tov'})
            log['pra'] = log['points'] + log['rebounds'] + log['assists']
            log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
            log['pps'] = log['points'] / log['fga'].replace(0, 1)
        return log, p_id, team_abbr
    except: return pd.DataFrame(), None, None

def calculate_sharp_lambda(p_mean, pace, sos, star, home, b2b):
    return p_mean * pace * sos * (1.15 if star else 1.0) * (1.03 if home else 0.97) * (0.95 if b2b else 1.0)

# --- 2. UI LAYOUT & SIDEBAR ---
st.set_page_config(page_title="Sharp Pro v3.4", layout="wide", page_icon="🏦")
sos_data, team_map = load_nba_universe()

with st.sidebar:
    st.header("💰 Bankroll & Staking")
    total_purse = st.number_input("Starting Purse ($)", value=1000, step=50)
    kelly_multiplier = st.slider("Kelly Fraction", 0.1, 1.0, 0.5)
    st.divider()
    
    st.header("🎯 System Mode")
    mode = st.radio("View", ["Single Player", "Team Scout Radar", "Parlay Builder"])
    st.divider()
    stat_cat = st.selectbox("Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    selected_opp = st.selectbox("Opponent", sorted(list(team_map.keys())), index=0)
    is_home = st.toggle("Home Game", value=True)
    is_b2b = st.toggle("Back-to-Back")
    star_out = st.toggle("Star Teammate Out?")
    pace_mult = st.select_slider("Pace", options=[0.92, 1.0, 1.08], value=1.0)

# --- 3. SINGLE PLAYER DASHBOARD ---
if mode == "Single Player":
    search_q = st.text_input("Player Search", "Jalen Brunson")
    p_df, p_id, team_abbr = get_player_data(search_q)
    
    if not p_df.empty:
        p_mean = p_df[stat_cat].mean()
        sharp_lambda = calculate_sharp_lambda(p_mean, pace_mult, sos_data.get(selected_opp, 1.0), star_out, is_home, is_b2b)
        
        col_bet1, col_bet2 = st.columns(2)
        user_line = col_bet1.number_input("Vegas Line", value=float(round(p_mean, 1)), step=0.5)
        market_odds = col_bet2.number_input("Odds (American)", value=-110)

        main_col, side_col = st.columns([2, 1])
        with main_col:
            st.subheader("📈 Last 10 Games Trend")
            last_10 = p_df.head(10).iloc[::-1]
            trend_fig = go.Figure(go.Scatter(x=list(range(1, 11)), y=last_10[stat_cat], mode='lines+markers', line=dict(color='#00ff96', width=4)))
            trend_fig.add_hline(y=user_line, line_dash="dash", line_color="red")
            trend_fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(trend_fig, use_container_width=True)

            st.subheader("📊 Efficiency Matrix")
            eff_fig = go.Figure(go.Scatter(x=p_df['usage'].head(15), y=p_df['pps'].head(15), mode='markers', marker=dict(size=12, color=p_df['points'], colorscale='Viridis')))
            eff_fig.update_layout(template="plotly_dark", xaxis_title="Usage", yaxis_title="PPS", height=350)
            st.plotly_chart(eff_fig, use_container_width=True)

        with side_col:
            st.subheader("🎲 Probability")
            st.metric("Sharp Projection", round(sharp_lambda, 1))
            over_prob = round((1 - poisson.cdf(user_line - 0.5, sharp_lambda)) * 100, 1)
            st.metric("Win Probability", f"{over_prob}%")
            
            # Kelly Logic
            dec_odds = (market_odds / 100) + 1 if market_odds > 0 else (100 / abs(market_odds)) + 1
            win_p = over_prob / 100
            k_f = ( (dec_odds-1)*win_p - (1-win_p) ) / (dec_odds-1) if dec_odds > 1 else 0
            st.metric("Suggested Stake", f"${max(0, round(k_f * total_purse * kelly_multiplier, 2))}")

# --- 4. TEAM SCOUT RADAR ---
elif mode == "Team Scout Radar":
    team_to_scout = st.selectbox("Select Team", sorted(list(team_map.keys())), index=0)
    if st.button(f"🚀 Scan {team_to_scout} Roster"):
        t_id = team_map[team_to_scout]
        with st.status("Analyzing Matchup...") as s:
            roster = commonteamroster.CommonTeamRoster(team_id=t_id).get_data_frames()[0].head(10)
            results = []
            for _, row in roster.iterrows():
                p_log, _, _ = get_player_data(row['PLAYER_ID'], is_id=True)
                if not p_log.empty:
                    m = p_log[stat_cat].mean()
                    p = calculate_sharp_lambda(m, pace_mult, sos_data.get(selected_opp, 1.0), star_out, is_home, is_b2b)
                    prob = round((1 - poisson.cdf(m - 0.5, p)) * 100, 1)
                    results.append({"Player": row['PLAYER'], "Avg": round(m,1), "Proj": round(p,1), "Edge": round(p-m,1), "Hit%": prob})
            s.update(label="Scan Complete", state="complete")
        
        st.session_state['radar_results'] = pd.DataFrame(results).sort_values("Edge", ascending=False)
        st.table(st.session_state['radar_results'])

# --- 5. PARLAY BUILDER (NEW FEATURE) ---
elif mode == "Parlay Builder":
    st.subheader("🔗 Multi-Leg Parlay Optimizer")
    if 'radar_results' not in st.session_state:
        st.warning("Please run a 'Team Scout Radar' scan first to populate available legs.")
    else:
        available_players = st.session_state['radar_results']['Player'].tolist()
        selections = st.multiselect("Select Legs for Parlay", available_players)
        
        if selections:
            parlay_prob = 1.0
            total_decimal_odds = 1.0
            
            st.write("### Parlay Composition")
            for player in selections:
                row = st.session_state['radar_results'][st.session_state['radar_results']['Player'] == player].iloc[0]
                p_win = row['Hit%'] / 100
                parlay_prob *= p_win
                
                # Assume standard -110 juice for each leg unless specified
                leg_odds = st.number_input(f"Odds for {player} Over {row['Avg']}", value=-110, key=f"odds_{player}")
                dec_leg = (leg_odds / 100) + 1 if leg_odds > 0 else (100 / abs(leg_odds)) + 1
                total_decimal_odds *= dec_leg
                
                st.write(f"✅ **{player}**: {row['Hit%']}% Win Prob | {leg_odds} Odds")
            
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Parlay Win Prob", f"{round(parlay_prob * 100, 2)}%")
            
            # Convert decimal back to American for display
            parlay_american = round((total_decimal_odds - 1) * 100) if total_decimal_odds >= 2 else round(-100 / (total_decimal_odds - 1))
            c2.metric("Combined Odds", f"{parlay_american}")
            
            # Parlay Kelly
            p_k_f = ( (total_decimal_odds-1)*parlay_prob - (1-parlay_prob) ) / (total_decimal_odds-1)
            parlay_stake = max(0, p_k_f * total_purse * kelly_multiplier)
            c3.metric("Suggested Stake", f"${round(parlay_stake, 2)}")
            
            if parlay_prob * total_decimal_odds > 1.0:
                st.success("🔥 This parlay has positive Expected Value (+EV)!")
            else:
                st.error("⚠️ Negative Expected Value. The juice outweighs the probability.")
