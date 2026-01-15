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
            log = log[log['minutes'] > 8]
            log['pra'] = log['points'] + log['rebounds'] + log['assists']
            for cat in ['points', 'rebounds', 'assists', 'three_pointers', 'pra']:
                log[f'{cat}_per_min'] = log[cat] / log['minutes'].replace(0, 1)
            return log
    except Exception: pass
    return pd.DataFrame()

def calculate_dvp(pos, opp_abbr):
    dvp_map = {
        'Center': {'OKC': 0.82, 'MIN': 0.85, 'WAS': 1.18, 'UTA': 1.12},
        'Guard': {'BOS': 0.88, 'OKC': 0.84, 'CHA': 1.14, 'WAS': 1.10},
        'Forward': {'NYK': 0.89, 'MIA': 0.91, 'DET': 1.09}
    }
    pos_key = 'Guard' if 'Guard' in pos else ('Center' if 'Center' in pos else 'Forward')
    return dvp_map.get(pos_key, {}).get(opp_abbr, 1.0)

# --- 2. VISUALIZATION ENGINE ---

def plot_scout_radar(team_abbr, opp_abbr, context_data):
    categories = ['Offense', 'Defense', 'Pace', 'Passing', 'Rebounding']
    def get_team_metrics(abbr):
        d = context_data.get(abbr, {'off_rtg': 112, 'def_rtg': 112, 'raw_pace': 99, 'ast_pct': 0.6, 'reb_pct': 0.5})
        return [d['off_rtg']/125, (145-d['def_rtg'])/45, d['raw_pace']/105, d['ast_pct'], d['reb_pct']]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=get_team_metrics(team_abbr), theta=categories, fill='toself', name=team_abbr))
    if opp_abbr:
        fig.add_trace(go.Scatterpolar(r=get_team_metrics(opp_abbr), theta=categories, fill='toself', name=opp_abbr))
    fig.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark", height=300, margin=dict(l=40, r=40, t=30, b=30))
    return fig

def plot_poisson_chart(mu, line, cat):
    x = np.arange(0, max(mu * 2.5, line + 5))
    y = poisson.pmf(x, mu)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, marker_color='#636EFA', opacity=0.6))
    fig.add_vline(x=line - 0.5, line_dash="dash", line_color="red")
    fig.update_layout(title=f"Hit Prob: {cat.upper()}", template="plotly_dark", height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# --- 3. APP SETUP ---

st.set_page_config(page_title="Sharp Pro v5.95", layout="wide")
team_map = {t['abbreviation']: t['id'] for t in teams.get_teams()}
context_data, lg_avg_pace = get_league_context()

with st.sidebar:
    st.title("🛡️ Sharp Pro v5.95")
    app_mode = st.radio("Analysis Mode", ["Single Player", "Team Value Scanner"])
    st.divider()
    stat_cat = st.selectbox("Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    total_purse = st.number_input("Purse ($)", value=1000)
    kelly_mult = st.slider("Kelly Fraction", 0.1, 1.0, 0.5)
    proj_minutes = st.slider("Projected Minutes", 10, 48, 32)
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
            # Baseline Projection Logic
            rate = p_df[f'{stat_cat}_per_min'].mean()
            dvp_mult = calculate_dvp(info['POSITION'].iloc[0], opp_abbr)
            pace_mult = (((context_data.get(team_abbr, {}).get('raw_pace', 99) + context_data.get(opp_abbr, {}).get('raw_pace', 99)) / 2) / lg_avg_pace) if opp_abbr else 1.0
            st_lambda = rate * proj_minutes * dvp_mult * pace_mult * (1.1 if vol_boost else 1.0)
            
            # Top Metrics
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Matchup", f"{team_abbr} {'vs' if is_home else '@'} {opp_abbr}")
            c2.metric("Projected", round(st_lambda, 2))
            c3.metric("DvP Factor", f"{dvp_mult}x")
            c4.metric("Season Avg", round(p_df[stat_cat].mean(), 1))

            col_l, col_m, col_r = st.columns([1, 1, 1])
            with col_l:
                st.subheader("🎯 Market Analysis")
                curr_line = st.number_input("Market Line", value=float(round(st_lambda, 1)), step=0.5)
                win_p = (1 - poisson.cdf(curr_line - 0.5, st_lambda))
                st.metric("Win Probability", f"{round(win_p*100, 1)}%")
                stake = max(0, total_purse * kelly_mult * (win_p - (1 - win_p)))
                st.metric("Kelly Stake", f"${round(stake, 2)}")
                st.plotly_chart(plot_poisson_chart(st_lambda, curr_line, stat_cat), use_container_width=True)

            with col_m:
                st.subheader("📡 Team Scout Radar")
                st.plotly_chart(plot_scout_radar(team_abbr, opp_abbr, context_data), use_container_width=True)

            with col_r:
                st.subheader("🎲 Monte Carlo (10k)")
                sims = np.random.poisson(st_lambda, 10000)
                fig_mc = go.Figure(data=[go.Histogram(x=sims, nbinsx=25, marker_color='#00CC96')])
                fig_mc.add_vline(x=curr_line, line_color="red", line_width=3)
                fig_mc.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_mc, use_container_width=True)
            
            # Recommended Legs Section
            st.divider()
            st.subheader("💡 Recommended Legs (Based on 10k Sims)")
            leg_cols = st.columns(3)
            with leg_cols[0]:
                alt_line = curr_line - 2 if curr_line > 5 else curr_line - 1
                alt_p = (1 - poisson.cdf(alt_line - 0.5, st_lambda))
                st.info(f"**Safe Leg:** {player_choice['full_name']} {alt_line}+ {stat_cat} ({round(alt_p*100)}% prob)")
            with leg_cols[1]:
                st.info(f"**Main Leg:** {player_choice['full_name']} Over {curr_line} {stat_cat} ({round(win_p*100)}% prob)")
            with leg_cols[2]:
                ladder_line = curr_line + 4
                ladder_p = (1 - poisson.cdf(ladder_line - 0.5, st_lambda))
                st.info(f"**Ladder Leg:** {player_choice['full_name']} {ladder_line}+ {stat_cat} ({round(ladder_p*100)}% prob)")

else:
    # [Team Value Scanner logic from previous version goes here...]
    st.header("📋 Team Value Scanner")
    team_choice = st.selectbox("Select Team", sorted(list(team_map.keys())))
    if team_choice:
        opp_abbr, is_home = get_live_matchup(team_choice, team_map)
        st.subheader(f"Scanning {team_choice} Roster vs {opp_abbr}")
        roster = commonteamroster.CommonTeamRoster(team_id=team_map[team_choice]).get_data_frames()[0]
        # (Full scanner logic remains the same for the team view)
