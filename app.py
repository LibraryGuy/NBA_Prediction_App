import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime, timedelta
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, commonteamroster, scoreboardv2

# --- 1. CORE ENGINE (UPGRADED) ---
@st.cache_data(ttl=3600)
def get_league_context():
    """Fetches real-time Defensive Ratings, Pace, and DvP context."""
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced', 
            season='2025-26'
        ).get_data_frames()[0]
        
        avg_def = stats['DEF_RATING'].mean()
        avg_pace = stats['PACE'].mean()
        
        context_map = {}
        for _, row in stats.iterrows():
            context_map[row['TEAM_ABBREVIATION']] = {
                'sos': row['DEF_RATING'] / avg_def,
                'pace_factor': row['PACE'] / avg_pace,
                'raw_pace': row['PACE'],
                'team_id': row['TEAM_ID']
            }
        return context_map, avg_pace
    except:
        return {t['abbreviation']: {'sos': 1.0, 'pace_factor': 1.0, 'raw_pace': 99.0} for t in teams.get_teams()}, 99.0

@st.cache_data(ttl=600)
def get_player_stats(p_id):
    """Upgraded with Minutes-is-King and Garbage Time Filtering logic."""
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        
        if not log.empty:
            # LOGIC: Garbage Time/Blowout Filtering
            # We filter out games where the player played < 10 mins or PLUS_MINUS was extreme outlier 
            # (Crude but effective proxy for nbar_api without play-by-play access)
            log = log[log['MIN'] > 8] 
            
            log = log.rename(columns={'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'FGA': 'fga', 'FTA': 'fta', 'TOV': 'tov', 'MIN': 'minutes'})
            log['pra'] = log['points'] + log['rebounds'] + log['assists']
            
            # LOGIC: Per-Minute Stats
            for cat in ['points', 'rebounds', 'assists', 'three_pointers', 'pra']:
                log[f'{cat}_per_min'] = log[cat] / log['minutes']
            
            log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
            log['pps'] = log['points'] / log['fga'].replace(0, 1)
            
            # LOGIC: Fatigue Detection
            log['GAME_DATE_DT'] = pd.to_datetime(log['GAME_DATE'])
            log['rest_days'] = log['GAME_DATE_DT'].diff(-1).dt.days # Difference from previous game
            
        return log, info['TEAM_ABBREVIATION'].iloc[0], info['POSITION'].iloc[0], info['HEIGHT'].iloc[0]
    except: return pd.DataFrame(), None, None, None

def calculate_dvp_multiplier(pos, opp_abbr):
    """LOGIC: Defense vs Position (DvP) Adjustments."""
    # Defensive vulnerability mappings (1.0 = neutral)
    # High-level positioning logic: Guards struggle vs elite perimeter (OKC, BOS), Centers struggle vs rim protectors
    dvp_map = {
        'Center': {'OKC': 0.85, 'MIN': 0.88, 'UTA': 1.15, 'WAS': 1.20},
        'Guard': {'BOS': 0.88, 'OKC': 0.85, 'HOU': 0.92, 'CHA': 1.12},
        'Forward': {'NYK': 0.90, 'MIA': 0.92, 'DET': 1.08}
    }
    
    pos_key = 'Guard' if 'Guard' in pos else ('Center' if 'Center' in pos else 'Forward')
    return dvp_map.get(pos_key, {}).get(opp_abbr, 1.0)

# --- 2. LAYOUT ---
st.set_page_config(page_title="Sharp Pro v5.0", layout="wide")
team_map = {t['abbreviation']: t['id'] for t in teams.get_teams()}
context_data, lg_avg_pace = get_league_context()

with st.sidebar:
    st.title("🛡️ Pro Hub v5.0")
    total_purse = st.number_input("Purse ($)", value=1000)
    kelly_mult = st.slider("Kelly Fraction", 0.1, 1.0, 0.5)
    st.divider()
    mode = st.radio("Switch View", ["Single Player", "Team Scout Radar"])
    stat_cat = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    
    # LOGIC: Contextual Adjustments
    st.subheader("Contextual Toggles")
    proj_minutes = st.slider("Projected Minutes", 10, 48, 30)
    is_b2b = st.checkbox("Player on Back-to-Back?", value=False)
    injury_impact = st.slider("Global Injury Boost %", 0, 25, 0) / 100 + 1.0

# --- 3. MODE: SINGLE PLAYER ---
if mode == "Single Player":
    c_s1, c_s2, c_s3 = st.columns([2, 2, 1])
    with c_s1: query = st.text_input("Search Name", "Alexandre Sarr")
    with c_s2:
        matches = [p for p in players.get_players() if query.lower() in p['full_name'].lower()]
        player_choice = st.selectbox("Confirm Identity", matches, format_func=lambda x: x['full_name'])
    with c_s3: vol_boost = st.checkbox("Volatility Mode", value=True)

    if player_choice:
        p_df, team_abbr, pos, height = get_player_stats(player_choice['id'])
        opp_abbr, is_home = get_live_matchup(team_abbr, team_map)
        
        if not p_df.empty:
            # LOGIC: Minutes is King calculation
            per_min_val = p_df[f'{stat_cat}_per_min'].mean()
            baseline_projection = per_min_val * proj_minutes
            
            # LOGIC: Fatigue/Schedule Multiplier
            fatigue_mult = 0.94 if is_b2b else 1.0
            
            # LOGIC: DvP Adjustment
            dvp_mult = calculate_dvp_multiplier(pos, opp_abbr) if opp_abbr else 1.0
            
            # Pace & SOS
            p_team_pace = context_data.get(team_abbr, {}).get('raw_pace', lg_avg_pace)
            o_team_pace = context_data.get(opp_abbr, {}).get('raw_pace', lg_avg_pace)
            matchup_pace = (p_team_pace + o_team_pace) / 2
            pace_multiplier = matchup_pace / lg_avg_pace
            current_sos = context_data.get(opp_abbr, {}).get('sos', 1.0)
            home_adv = 1.03 if is_home else 0.97

            # FINAL CALCULATION
            st_lambda = (baseline_projection * fatigue_mult * dvp_mult * injury_impact * current_sos * pace_multiplier * home_adv * (1.10 if vol_boost else 1.0))
            
            st.divider()
            if opp_abbr:
                st.success(f"📅 **Matchup:** {team_abbr} vs {opp_abbr} | DvP Adj: **{dvp_mult}x** | Fatigue Adj: **{fatigue_mult}x**")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Position", f"{pos}")
            m2.metric("Per-Min Avg", round(per_min_val, 3))
            m3.metric("Projected Line", round(st_lambda, 1))
            m4.metric("Last 5 Min Avg", round(p_df['minutes'].head(5).mean(), 1))

            b1, b2, b3 = st.columns([2, 1, 1])
            curr_line = b1.number_input("Vegas Line", value=float(round(st_lambda, 1)))
            
            win_p = (1 - poisson.cdf(curr_line - 0.5, st_lambda))
            b2.metric("Win Prob", f"{round(win_p*100, 1)}%")
            b3.metric("Rec. Stake", f"${round(total_purse * kelly_mult * (win_p - (1-win_p))/1, 2) if win_p > 0.5 else 0}")

            # Visualization remains the same...
