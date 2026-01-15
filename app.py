import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime, timedelta
import pytz
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, scoreboardv2

# --- 1. CORE ENGINE (Fixed Matchup Logic) ---

def get_live_matchup(team_abbr, team_map):
    """Improved matchup detection to fix team-sync issues."""
    try:
        t_id = team_map.get(team_abbr)
        if not t_id: return None, True
        
        # Get current date in US/Eastern (NBA Time)
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)
        
        # Check Today and Yesterday (in case game is currently live/late night)
        dates_to_check = [now.strftime('%Y-%m-%d'), (now - timedelta(days=1)).strftime('%Y-%m-%d')]
        
        for date_str in dates_to_check:
            sb = scoreboardv2.ScoreboardV2(game_date=date_str, league_id='00')
            board = sb.get_data_frames()[0]
            
            if not board.empty:
                # Filter for the specific team ID
                game = board[(board['HOME_TEAM_ID'] == t_id) | (board['VISITOR_TEAM_ID'] == t_id)]
                if not game.empty:
                    is_home = (game.iloc[0]['HOME_TEAM_ID'] == t_id)
                    opp_id = game.iloc[0]['VISITOR_TEAM_ID'] if is_home else game.iloc[0]['HOME_TEAM_ID']
                    opp_abbr = next((abbr for abbr, tid in team_map.items() if tid == opp_id), "OPP")
                    return opp_abbr, is_home
                    
    except Exception as e: 
        st.sidebar.warning(f"Live Syncing Matchup... ({team_abbr})")
    return None, True 

@st.cache_data(ttl=3600)
def get_league_context():
    try:
        # Season set to 2025-26
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced', season='2025-26').get_data_frames()[0]
        avg_def = stats['DEF_RATING'].mean()
        avg_pace = stats['PACE'].mean()
        context_map = {row['TEAM_ABBREVIATION']: {
            'sos': row['DEF_RATING'] / avg_def, 
            'pace_factor': row['PACE'] / avg_pace, 
            'raw_pace': row['PACE'],
            'off_rtg': row['OFF_RATING'],
            'def_rtg': row['DEF_RATING'],
            'ast_pct': row['AST_PCT'],
            'reb_pct': row['REB_PCT']
        } for _, row in stats.iterrows()}
        return context_map, avg_pace
    except Exception: return {t['abbreviation']: {'sos': 1.0, 'pace_factor': 1.0, 'raw_pace': 99.0, 'off_rtg': 115, 'def_rtg': 115, 'ast_pct': 0.6, 'reb_pct': 0.5} for t in teams.get_teams()}, 99.0

@st.cache_data(ttl=600)
def get_player_stats(p_id):
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
        # Pulling current 2025-26 Season
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if not log.empty:
            log = log.rename(columns={'MATCHUP': 'matchup', 'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'FGA': 'fga', 'FTA': 'fta', 'TOV': 'tov', 'MIN': 'minutes'})
            log = log[log['minutes'] > 8]
            log['pra'] = log['points'] + log['rebounds'] + log['assists']
            for cat in ['points', 'rebounds', 'assists', 'three_pointers', 'pra']:
                log[f'{cat}_per_min'] = log[cat] / log['minutes'].replace(0, 1)
            return log, info['TEAM_ABBREVIATION'].iloc[0], info['POSITION'].iloc[0], info['HEIGHT'].iloc[0]
    except Exception: pass
    return pd.DataFrame(), None, None, None

def calculate_dvp_multiplier(pos, opp_abbr):
    dvp_map = {
        'Center': {'OKC': 0.82, 'MIN': 0.85, 'WAS': 1.18, 'UTA': 1.12, 'LAC': 0.92},
        'Guard': {'BOS': 0.88, 'OKC': 0.84, 'HOU': 0.91, 'CHA': 1.14, 'WAS': 1.10},
        'Forward': {'NYK': 0.89, 'MIA': 0.91, 'DET': 1.09, 'GSW': 1.05}
    }
    pos_key = 'Guard' if 'Guard' in pos else ('Center' if 'Center' in pos else 'Forward')
    return dvp_map.get(pos_key, {}).get(opp_abbr, 1.0)

# --- 2. BAYESIAN & LOGIC ---

def get_bayesian_adjusted_rate(p_df, stat_cat):
    if p_df.empty: return 0.0
    prior_mu = p_df[f'{stat_cat}_per_min'].mean()
    prior_var = p_df[f'{stat_cat}_per_min'].var() or 0.01
    recent_data = p_df.head(5)[f'{stat_cat}_per_min']
    evidence_mu = recent_data.mean()
    evidence_var = recent_data.var() or 0.01
    w_prior = 1 / prior_var
    w_evidence = 1 / evidence_var
    return (w_prior * prior_mu + w_evidence * evidence_mu) / (w_prior + w_evidence)

def get_h2h_performance(p_df, opp_abbr, stat_cat):
    if not opp_abbr or p_df.empty: return None
    h2h_games = p_df[p_df['matchup'].str.contains(opp_abbr)]
    if h2h_games.empty: return None
    return {'count': len(h2h_games), 'avg': h2h_games[stat_cat].mean(), 'last': h2h_games[stat_cat].iloc[0]}

# --- 3. VISUALIZATION ---

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

# --- 4. APP LAYOUT ---

st.set_page_config(page_title="Sharp Pro v5.8", layout="wide")
team_map = {t['abbreviation']: t['id'] for t in teams.get_teams()}
context_data, lg_avg_pace = get_league_context()

with st.sidebar:
    st.title("🛡️ Sharp Pro v5.8")
    total_purse = st.number_input("Purse ($)", value=1000)
    kelly_mult = st.slider("Kelly Fraction", 0.1, 1.0, 0.5)
    st.divider()
    stat_cat = st.selectbox("Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    proj_minutes = st.slider("Projected Minutes", 10, 48, 32)
    vol_boost = st.checkbox("Volatility Mode", value=True)

query = st.text_input("Search Player", "James Harden")
matches = [p for p in players.get_players() if query.lower() in p['full_name'].lower()]
player_choice = st.selectbox("Select Player", matches, format_func=lambda x: x['full_name'])

if player_choice:
    p_df, team_abbr, pos, height = get_player_stats(player_choice['id'])
    opp_abbr, is_home = get_live_matchup(team_abbr, team_map)
    
    if not p_df.empty:
        # Bayesian Logic
        bayes_per_min = get_bayesian_adjusted_rate(p_df, stat_cat)
        baseline = bayes_per_min * proj_minutes
        dvp_mult = calculate_dvp_multiplier(pos or "Guard", opp_abbr)
        pace_mult = (((context_data.get(team_abbr, {}).get('raw_pace', 99) + context_data.get(opp_abbr, {}).get('raw_pace', 99)) / 2) / lg_avg_pace) if opp_abbr else 1.0
        
        st_lambda = baseline * dvp_mult * pace_mult * (1.1 if vol_boost else 1.0)
        
        # Metrics Row
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matchup", f"{team_abbr} {'vs' if is_home else '@'} {opp_abbr}")
        c2.metric("Bayesian Proj", round(st_lambda, 2))
        c3.metric("DvP Factor", f"{dvp_mult}x")
        c4.metric("Season Avg", round(p_df[stat_cat].mean(), 1))

        # Main Visualization Row
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
