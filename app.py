import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime, timedelta
import pytz
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, scoreboardv2, commonteamroster

# --- 1. CORE ENGINE ---

def get_live_matchup(team_abbr, team_map):
    try:
        t_id = team_map.get(team_abbr)
        if not t_id: return None, True
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)
        dates_to_check = [now.strftime('%Y-%m-%d'), (now - timedelta(days=1)).strftime('%Y-%m-%d')]
        for date_str in dates_to_check:
            sb = scoreboardv2.ScoreboardV2(game_date=date_str, league_id='00')
            board = sb.get_data_frames()[0]
            if not board.empty:
                game = board[(board['HOME_TEAM_ID'] == t_id) | (board['VISITOR_TEAM_ID'] == t_id)]
                if not game.empty:
                    is_home = (game.iloc[0]['HOME_TEAM_ID'] == t_id)
                    opp_id = game.iloc[0]['VISITOR_TEAM_ID'] if is_home else game.iloc[0]['HOME_TEAM_ID']
                    opp_abbr = next((abbr for abbr, tid in team_map.items() if tid == opp_id), "OPP")
                    return opp_abbr, is_home
    except Exception: pass
    return None, True 

@st.cache_data(ttl=3600)
def get_league_context():
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced', season='2025-26').get_data_frames()[0]
        avg_def = stats['DEF_RATING'].mean()
        avg_pace = stats['PACE'].mean()
        context_map = {row['TEAM_ABBREVIATION']: {
            'raw_pace': row['PACE'], 'off_rtg': row['OFF_RATING'], 'def_rtg': row['DEF_RATING'],
            'ast_pct': row['AST_PCT'], 'reb_pct': row['REB_PCT']
        } for _, row in stats.iterrows()}
        return context_map, avg_pace
    except Exception: return {}, 99.0

@st.cache_data(ttl=600)
def get_player_stats(p_id):
    try:
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if not log.empty:
            log = log.rename(columns={'MATCHUP': 'matchup', 'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'MIN': 'minutes'})
            log = log[log['minutes'] > 5]
            log['pra'] = log['points'] + log['rebounds'] + log['assists']
            for cat in ['points', 'rebounds', 'assists', 'three_pointers', 'pra']:
                log[f'{cat}_per_min'] = log[cat] / log['minutes'].replace(0, 1)
            return log
    except Exception: pass
    return pd.DataFrame()

def calculate_dvp(pos, opp_abbr):
    # Simplified DvP Logic
    weights = {'WAS': 1.15, 'UTA': 1.10, 'CHA': 1.12, 'OKC': 0.85, 'BOS': 0.88}
    return weights.get(opp_abbr, 1.0)

# --- 2. TEAM SCANNER LOGIC ---

@st.cache_data(ttl=3600)
def get_team_roster(team_id):
    roster = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
    return roster[['PLAYER_ID', 'PLAYER', 'POSITION']]

# --- 3. APP LAYOUT ---

st.set_page_config(page_title="Sharp Pro v5.9", layout="wide")
team_map = {t['abbreviation']: t['id'] for t in teams.get_teams()}
context_data, lg_avg_pace = get_league_context()

with st.sidebar:
    st.title("🛡️ Sharp Pro v5.9")
    app_mode = st.radio("Analysis Mode", ["Single Player", "Team Value Scanner"])
    st.divider()
    stat_cat = st.selectbox("Category", ["points", "rebounds", "assists", "pra"])
    total_purse = st.number_input("Purse ($)", value=1000)
    kelly_mult = st.slider("Kelly Fraction", 0.1, 1.0, 0.5)
    vol_boost = st.checkbox("Volatility Mode (1.1x)", value=True)

if app_mode == "Single Player":
    query = st.text_input("Search Player", "James Harden")
    matches = [p for p in players.get_players() if query.lower() in p['full_name'].lower() and p['is_active']]
    player_choice = st.selectbox("Select Player", matches, format_func=lambda x: x['full_name'])
    
    if player_choice:
        p_df = get_player_stats(player_choice['id'])
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_choice['id']).get_data_frames()[0]
        team_abbr = info['TEAM_ABBREVIATION'].iloc[0]
        opp_abbr, is_home = get_live_matchup(team_abbr, team_map)
        
        if not p_df.empty:
            # Calculation
            rate = p_df[f'{stat_cat}_per_min'].mean()
            st_lambda = rate * 32 * calculate_dvp(info['POSITION'].iloc[0], opp_abbr) * (1.1 if vol_boost else 1.0)
            
            st.metric(f"Projected {stat_cat.upper()} vs {opp_abbr}", round(st_lambda, 2))
            # [Previous Visualization code from v5.8 goes here...]

else:
    st.header("📋 Team Value Scanner")
    team_choice = st.selectbox("Select Team", sorted(list(team_map.keys())))
    
    if team_choice:
        opp_abbr, is_home = get_live_matchup(team_choice, team_map)
        st.subheader(f"Analyzing {team_choice} Roster vs {opp_abbr}")
        
        roster = get_team_roster(team_map[team_choice])
        results = []
        
        progress_bar = st.progress(0)
        for idx, row in roster.iterrows():
            p_log = get_player_stats(row['PLAYER_ID'])
            if not p_log.empty:
                rate = p_log[f'{stat_cat}_per_min'].mean()
                proj = rate * 30 * calculate_dvp(row['POSITION'], opp_abbr)
                
                # Monte Carlo for this specific player
                sims = np.random.poisson(proj, 5000)
                # Compare vs a hypothetical market line (Season Avg)
                market_line = round(p_log[stat_cat].mean(), 1)
                win_p = (sims > market_line).mean()
                
                results.append({
                    "Player": row['PLAYER'],
                    "Proj": round(proj, 2),
                    "Avg": market_line,
                    "Edge": round(win_p * 100, 1),
                    "Position": row['POSITION']
                })
            progress_bar.progress((idx + 1) / len(roster))
            
        res_df = pd.DataFrame(results).sort_values(by="Edge", ascending=False)
        
        # Display Heatmap/Table
        st.table(res_df)
        
        # Highlight Top Pick
        if not res_df.empty:
            top_p = res_df.iloc[0]
            st.success(f"🔥 Best Value Pick: **{top_p['Player']}** with a {top_p['Edge']}% chance to exceed season average.")
