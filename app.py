import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime, timedelta # Added timedelta for the window fix
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, scoreboardv2

# --- 1. CORE ENGINE (REPAIR & UPGRADE) ---

def get_live_matchup(team_abbr, team_map):
    """Hardened function to catch late-night games and prevent unpacking errors."""
    default_return = (None, True) 
    
    try:
        t_id = team_map.get(team_abbr)
        if not t_id:
            return default_return

        # LOGIC: Check Today and Tomorrow to account for EST/Local Timezone rollovers
        # Late night games (like Wizards @ Clippers at 10:30 PM) often trigger this.
        check_dates = [
            datetime.now().strftime('%Y-%m-%d'),
            (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        ]
        
        for date_str in check_dates:
            sb = scoreboardv2.ScoreboardV2(game_date=date_str)
            board_dfs = sb.get_data_frames()
            
            if not board_dfs or board_dfs[0].empty:
                continue
                
            board = board_dfs[0]
            game = board[(board['HOME_TEAM_ID'] == t_id) | (board['VISITOR_TEAM_ID'] == t_id)]
            
            if not game.empty:
                is_home = (game.iloc[0]['HOME_TEAM_ID'] == t_id)
                opp_id = game.iloc[0]['VISITOR_TEAM_ID'] if is_home else game.iloc[0]['HOME_TEAM_ID']
                
                # Map the opponent ID back to an Abbreviation
                opp_abbr = next((abbr for abbr, tid in team_map.items() if tid == opp_id), "OPP")
                return opp_abbr, is_home
                
    except Exception as e:
        # Log error to Streamlit for transparency if the API fails
        st.sidebar.error(f"API Matchup Error: {e}")
        
    return default_return 

@st.cache_data(ttl=3600)
def get_league_context():
    """Fetches real-time 2026 Defensive Ratings and Pace."""
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
                'raw_pace': row['PACE']
            }
        return context_map, avg_pace
    except Exception:
        return {t['abbreviation']: {'sos': 1.0, 'pace_factor': 1.0, 'raw_pace': 99.0} for t in teams.get_teams()}, 99.0

@st.cache_data(ttl=600)
def get_player_stats(p_id):
    """Fetches player logs and metadata."""
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        
        if not log.empty:
            log = log.rename(columns={'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'FGA': 'fga', 'FTA': 'fta', 'TOV': 'tov', 'MIN': 'minutes'})
            log = log[log['minutes'] > 8]
            
            log['pra'] = log['points'] + log['rebounds'] + log['assists']
            for cat in ['points', 'rebounds', 'assists', 'three_pointers', 'pra']:
                log[f'{cat}_per_min'] = log[cat] / log['minutes'].replace(0, 1)
                
            return log, info['TEAM_ABBREVIATION'].iloc[0], info['POSITION'].iloc[0], info['HEIGHT'].iloc[0]
    except Exception:
        pass
    return pd.DataFrame(), None, None, None

def calculate_dvp_multiplier(pos, opp_abbr):
    """Logic for Position-specific defensive adjustments."""
    dvp_map = {
        'Center': {'OKC': 0.85, 'MIN': 0.88, 'UTA': 1.15, 'WAS': 1.20, 'LAC': 0.95},
        'Guard': {'BOS': 0.88, 'OKC': 0.85, 'HOU': 0.92, 'CHA': 1.12, 'LAC': 0.92},
        'Forward': {'NYK': 0.90, 'MIA': 0.92, 'DET': 1.08, 'LAC': 0.94}
    }
    pos_key = 'Guard' if 'Guard' in pos else ('Center' if 'Center' in pos else 'Forward')
    return dvp_map.get(pos_key, {}).get(opp_abbr, 1.0)

# --- 2. LAYOUT & UI ---
st.set_page_config(page_title="Sharp Pro v5.1", layout="wide")
team_map = {t['abbreviation']: t['id'] for t in teams.get_teams()}
context_data, lg_avg_pace = get_league_context()

with st.sidebar:
    st.title("🛡️ Pro Hub v5.1")
    total_purse = st.number_input("Purse ($)", value=1000)
    kelly_mult = st.slider("Kelly Fraction", 0.1, 1.0, 0.5)
    st.divider()
    mode = st.radio("Switch View", ["Single Player", "Team Scout Radar"])
    stat_cat = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    
    st.subheader("Contextual Toggles")
    proj_minutes = st.slider("Projected Minutes", 10, 48, 30)
    is_b2b = st.checkbox("Player on Back-to-Back?", value=False)
    injury_impact = st.slider("Global Injury Boost %", 0, 25, 0) / 100 + 1.0

# --- 3. SINGLE PLAYER MODE ---
if mode == "Single Player":
    c_s1, c_s2, c_s3 = st.columns([2, 2, 1])
    with c_s1: query = st.text_input("Search Name", "Alexandre Sarr")
    with c_s2:
        matches = [p for p in players.get_players() if query.lower() in p['full_name'].lower()]
        player_choice = st.selectbox("Confirm Identity", matches, format_func=lambda x: x['full_name'])
    with c_s3: vol_boost = st.checkbox("Volatility Mode", value=True)

    if player_choice:
        p_df, team_abbr, pos, height = get_player_stats(player_choice['id'])
        
        # This will now check a 2-day window to catch the 10:30pm Clippers game
        opp_abbr, is_home = get_live_matchup(team_abbr, team_map)
        
        if not p_df.empty:
            per_min_val = p_df[f'{stat_cat}_per_min'].mean()
            baseline = per_min_val * proj_minutes
            
            fatigue_mult = 0.94 if is_b2b else 1.0
            dvp_mult = calculate_dvp_multiplier(pos or "Guard", opp_abbr)
            home_adv = 1.03 if is_home else 0.97
            
            p_pace = context_data.get(team_abbr, {}).get('raw_pace', lg_avg_pace)
            o_pace = context_data.get(opp_abbr, {}).get('raw_pace', lg_avg_pace) if opp_abbr else lg_avg_pace
            pace_mult = ((p_pace + o_pace) / 2) / lg_avg_pace
            sos_adj = context_data.get(opp_abbr, {}).get('sos', 1.0) if opp_abbr else 1.0
            
            st_lambda = baseline * fatigue_mult * dvp_mult * injury_impact * sos_adj * pace_mult * home_adv * (1.10 if vol_boost else 1.0)
            
            st.divider()
            if opp_abbr:
                st.success(f"📅 **Matchup Found:** {team_abbr} vs {opp_abbr} | DvP: {dvp_mult}x | Fatigue: {fatigue_mult}x")
            else:
                st.warning(f"⚠️ No game found for {team_abbr} in the immediate window. Using neutral metrics.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pos", pos)
            m2.metric("Stat/Min", round(per_min_val, 3))
            m3.metric("Proj. Line", round(st_lambda, 1))
            m4.metric("Season Avg Mins", round(p_df['minutes'].mean(), 1))

            b1, b2, b3 = st.columns([2, 1, 1])
            curr_line = b1.number_input("Vegas Line", value=float(round(st_lambda, 1)))
            win_p = (1 - poisson.cdf(curr_line - 0.5, st_lambda))
            
            b2.metric("Win Prob", f"{round(win_p*100, 1)}%")
            edge = win_p - (1 - win_p)
            stake = max(0, total_purse * kelly_mult * edge)
            b3.metric("Rec. Stake", f"${round(stake, 2)}")
