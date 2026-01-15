import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime, timedelta
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, scoreboardv2

# --- 1. CORE ENGINE (Preserved & Enhanced) ---

def get_live_matchup(team_abbr, team_map):
    default_return = (None, True) 
    try:
        t_id = team_map.get(team_abbr)
        if not t_id: return default_return
        check_dates = [datetime.now().strftime('%Y-%m-%d'), (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')]
        for date_str in check_dates:
            sb = scoreboardv2.ScoreboardV2(game_date=date_str)
            board_dfs = sb.get_data_frames()
            if not board_dfs or board_dfs[0].empty: continue
            board = board_dfs[0]
            game = board[(board['HOME_TEAM_ID'] == t_id) | (board['VISITOR_TEAM_ID'] == t_id)]
            if not game.empty:
                is_home = (game.iloc[0]['HOME_TEAM_ID'] == t_id)
                opp_id = game.iloc[0]['VISITOR_TEAM_ID'] if is_home else game.iloc[0]['HOME_TEAM_ID']
                opp_abbr = next((abbr for abbr, tid in team_map.items() if tid == opp_id), "OPP")
                return opp_abbr, is_home
    except Exception as e: st.sidebar.error(f"Matchup Error: {e}")
    return default_return 

@st.cache_data(ttl=3600)
def get_league_context():
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced', season='2025-26').get_data_frames()[0]
        avg_def = stats['DEF_RATING'].mean()
        avg_pace = stats['PACE'].mean()
        context_map = {row['TEAM_ABBREVIATION']: {'sos': row['DEF_RATING'] / avg_def, 'pace_factor': row['PACE'] / avg_pace, 'raw_pace': row['PACE']} for _, row in stats.iterrows()}
        return context_map, avg_pace
    except Exception: return {t['abbreviation']: {'sos': 1.0, 'pace_factor': 1.0, 'raw_pace': 99.0} for t in teams.get_teams()}, 99.0

@st.cache_data(ttl=600)
def get_player_stats(p_id):
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
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
        'Center': {'OKC': 0.85, 'MIN': 0.88, 'UTA': 1.15, 'WAS': 1.20, 'LAC': 0.95},
        'Guard': {'BOS': 0.88, 'OKC': 0.85, 'HOU': 0.92, 'CHA': 1.12, 'LAC': 0.92},
        'Forward': {'NYK': 0.90, 'MIA': 0.92, 'DET': 1.08, 'LAC': 0.94}
    }
    pos_key = 'Guard' if 'Guard' in pos else ('Center' if 'Center' in pos else 'Forward')
    return dvp_map.get(pos_key, {}).get(opp_abbr, 1.0)

# --- 2. BAYESIAN & H2H LOGIC ---

def get_bayesian_adjusted_rate(p_df, stat_cat):
    """Calculates a Bayesian Posterior Rate by weighting Season Avg vs L5 Hot Streak"""
    if p_df.empty: return 0.0
    
    # Prior (Season Long)
    prior_mu = p_df[f'{stat_cat}_per_min'].mean()
    prior_var = p_df[f'{stat_cat}_per_min'].var() or 0.01
    
    # Evidence (Last 5 Games)
    recent_data = p_df.head(5)[f'{stat_cat}_per_min']
    evidence_mu = recent_data.mean()
    evidence_var = recent_data.var() or 0.01
    
    # Bayesian Update: (Prior_Precision * Prior_Mu + Evidence_Precision * Evidence_Mu) / Total_Precision
    # Precision = 1 / Variance
    w_prior = 1 / prior_var
    w_evidence = 1 / evidence_var
    
    posterior_mu = (w_prior * prior_mu + w_evidence * evidence_mu) / (w_prior + w_evidence)
    return posterior_mu

def get_h2h_performance(p_df, opp_abbr, stat_cat):
    if not opp_abbr or p_df.empty: return None
    h2h_games = p_df[p_df['matchup'].str.contains(opp_abbr)]
    if h2h_games.empty: return None
    return {'count': len(h2h_games), 'avg': h2h_games[stat_cat].mean(), 'last': h2h_games[stat_cat].iloc[0]}

def get_smart_recommendations(mu, line, win_p, p_df, stat_cat, opp_abbr, p10, h2h_data):
    recs = []
    edge = (mu - line) / line if line > 0 else 0
    top_5_def = ['OKC', 'DET', 'BOS', 'MIA', 'PHI']
    
    if win_p > 0.62 and edge > 0.20:
        recs.append({"label": "🔥 PRO SIGNAL", "val": f"{round(edge*100)}% Edge detected", "type": "success"})
    
    if h2h_data and h2h_data['count'] > 0:
        if h2h_data['avg'] > line * 1.1:
            recs.append({"label": "⚔️ H2H DOMINANCE", "val": f"Avg {round(h2h_data['avg'],1)} vs {opp_abbr}", "type": "success"})
        elif h2h_data['avg'] < line * 0.9:
            recs.append({"label": "❄️ H2H STRUGGLE", "val": f"Avg {round(h2h_data['avg'],1)} vs {opp_abbr}", "type": "error"})

    if opp_abbr in top_5_def:
        recs.append({"label": "🛡️ DEFENSE ALERT", "val": f"{opp_abbr} is Elite Defense", "type": "error"})
    
    if p10 >= line:
        recs.append({"label": "💎 SAFETY FLOOR", "val": "Bottom 10% Sim covers line", "type": "success"})
        
    return recs

# --- 3. VISUALIZATION ---

def plot_poisson_chart(mu, line, cat):
    x = np.arange(0, max(mu * 2.5, line + 5))
    y = poisson.pmf(x, mu)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, name="Prob", marker_color='rgba(100, 149, 237, 0.6)'))
    fig.add_vline(x=line - 0.5, line_dash="dash", line_color="red", annotation_text=f"Line: {line}")
    fig.update_layout(title=f"Poisson Hit Prob: {cat.upper()}", template="plotly_dark", height=280, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_monte_carlo(mu, line):
    sims = np.random.poisson(mu, 10000)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sims, nbinsx=30, marker_color='#636EFA', opacity=0.7))
    p10, p90 = np.percentile(sims, 10), np.percentile(sims, 90)
    fig.add_vline(x=line, line_color="red", line_width=3)
    fig.update_layout(title="Monte Carlo (10k Sims)", template="plotly_dark", height=280, margin=dict(l=10, r=10, t=40, b=10))
    return fig, p10, p90

# --- 4. APP LAYOUT ---

st.set_page_config(page_title="Sharp Pro v5.6", layout="wide")
team_map = {t['abbreviation']: t['id'] for t in teams.get_teams()}
context_data, lg_avg_pace = get_league_context()

with st.sidebar:
    st.title("🛡️ Sharp Pro v5.6")
    total_purse = st.number_input("Purse ($)", value=1000)
    kelly_mult = st.slider("Kelly Fraction", 0.1, 1.0, 0.5)
    st.divider()
    stat_cat = st.selectbox("Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    proj_minutes = st.slider("Projected Minutes", 10, 48, 30)
    is_b2b = st.checkbox("Back-to-Back?", value=False)
    vol_boost = st.checkbox("Volatility Mode", value=True)

query = st.text_input("Search Player", "Alexandre Sarr")
matches = [p for p in players.get_players() if query.lower() in p['full_name'].lower()]
player_choice = st.selectbox("Select Player", matches, format_func=lambda x: x['full_name'])

if player_choice:
    p_df, team_abbr, pos, height = get_player_stats(player_choice['id'])
    opp_abbr, is_home = get_live_matchup(team_abbr, team_map)
    
    if not p_df.empty:
        # Adjustment Logic with Bayesian Smoothing
        bayes_per_min = get_bayesian_adjusted_rate(p_df, stat_cat)
        baseline = bayes_per_min * proj_minutes
        
        dvp_mult = calculate_dvp_multiplier(pos or "Forward", opp_abbr)
        pace_mult = (((context_data.get(team_abbr, {}).get('raw_pace', 99) + context_data.get(opp_abbr, {}).get('raw_pace', 99)) / 2) / lg_avg_pace) if opp_abbr else 1.0
        
        st_lambda = baseline * dvp_mult * pace_mult * (0.94 if is_b2b else 1.0) * (1.1 if vol_boost else 1.0)
        
        # UI Metrics
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matchup", f"{team_abbr} vs {opp_abbr}" if opp_abbr else "No Matchup Found")
        c2.metric("Bayesian Proj", round(st_lambda, 2), delta=f"{round((bayes_per_min - p_df[f'{stat_cat}_per_min'].mean())*proj_minutes, 1)} Hot/Cold Adj")
        c3.metric("DvP Adj", f"{dvp_mult}x")
        c4.metric("Season Avg", round(p_df[stat_cat].mean(), 1))

        # Inputs
        b1, b2, b3 = st.columns([2, 1, 1])
        curr_line = b1.number_input("Market Line", value=float(round(st_lambda, 1)), step=0.5)
        win_p = (1 - poisson.cdf(curr_line - 0.5, st_lambda))
        b2.metric("Win Probability", f"{round(win_p*100, 1)}%")
        stake = max(0, total_purse * kelly_mult * (win_p - (1 - win_p)))
        b3.metric("Kelly Stake", f"${round(stake, 2)}")

        # Charts
        col_l, col_r = st.columns(2)
        with col_l: st.plotly_chart(plot_poisson_chart(st_lambda, curr_line, stat_cat), use_container_width=True)
        with col_r:
            mc_fig, p10, p90 = plot_monte_carlo(st_lambda, curr_line)
            st.plotly_chart(mc_fig, use_container_width=True)

        # H2H and Recs
        h2h_data = get_h2h_performance(p_df, opp_abbr, stat_cat)
        st.subheader("🎯 Smart Prop Recommendations")
        recs = get_smart_recommendations(st_lambda, curr_line, win_p, p_df, stat_cat, opp_abbr, p10, h2h_data)
        
        if recs:
            r_cols = st.columns(len(recs))
            for idx, r in enumerate(recs):
                with r_cols[idx]:
                    if r['type'] == "success": st.success(f"**{r['label']}**\n\n{r['val']}")
                    elif r['type'] == "error": st.error(f"**{r['label']}**\n\n{r['val']}")
                    else: st.info(f"**{r['label']}**\n\n{r['val']}")
        else:
            st.info("No standout trends for this line.")
