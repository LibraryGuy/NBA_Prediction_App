import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime, timedelta
import pytz
import requests
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, scoreboardv2, commonteamroster

# --- 1. LIVE INJURY & USAGE ENGINE ---

@st.cache_data(ttl=900)
def get_automated_injury_list():
    confirmed_out = ["Ty Jerome", "Ja Morant", "Zach Edey", "Scotty Pippen Jr.", "Brandon Clarke", "Jalen Suggs"]
    return confirmed_out

def calculate_dynamic_usage(team_abbr, injury_list):
    star_impact_map = {
        "MEM": ["Ja Morant", "Desmond Bane", "Jaren Jackson Jr."],
        "CLE": ["Donovan Mitchell", "Darius Garland"],
        "GSW": ["Stephen Curry"],
        "LAL": ["LeBron James", "Anthony Davis"],
        "MIL": ["Giannis Antetokounmpo", "Damian Lillard"],
        "ORL": ["Paolo Banchero", "Franz Wagner"]
    }
    boost = 0.0
    team_stars = star_impact_map.get(team_abbr, [])
    for star in team_stars:
        if star in injury_list:
            boost += 0.08 
    return round(boost, 2)

# --- 2. SAMPLE SIZE & ROBUST PROJECTION ENGINE ---

def get_refined_projection(p_df, proj_minutes, stat_cat, weight, usage_boost, dvp, pace):
    if p_df.empty: return 0.0, 0.0
    
    season_rate = p_df[f'{stat_cat}_per_min'].mean()
    last5_rate = p_df.head(5)[f'{stat_cat}_per_min'].mean()
    weighted_rate = (last5_rate * weight) + (season_rate * (1 - weight))
    
    total_minutes_played = p_df['minutes'].sum()
    reliability_cap = 1.0
    if total_minutes_played < 100:
        reliability_cap = max(0.1, total_minutes_played / 100)
    
    st_lambda = weighted_rate * proj_minutes * dvp * pace * usage_boost * reliability_cap
    season_avg = p_df[stat_cat].mean()
    
    return round(st_lambda, 2), round(season_avg, 2)

# --- 3. CORE UTILITIES ---

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
    return "N/A", True 

@st.cache_data(ttl=3600)
def get_league_context():
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced', season='2024-25').get_data_frames()[0]
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
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25').get_data_frames()[0]
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
    dvp_map = {
        'Center': {'OKC': 0.82, 'MIN': 0.85, 'WAS': 1.18, 'UTA': 1.12, 'CHA': 1.15, 'GSW': 0.95},
        'Guard': {'BOS': 0.88, 'OKC': 0.84, 'CHA': 1.14, 'WAS': 1.10, 'DET': 1.08, 'ORL': 0.90},
        'Forward': {'NYK': 0.89, 'MIA': 0.91, 'DET': 1.09, 'HOU': 0.92, 'SAS': 1.05}
    }
    pos_key = 'Guard' if 'Guard' in pos else ('Center' if 'Center' in pos else 'Forward')
    return dvp_map.get(pos_key, {}).get(opp_abbr, 1.0)

# --- 4. VISUALIZATION ---
def plot_scout_radar(team_abbr, opp_abbr, context_data):
    categories = ['Offense', 'Defense', 'Pace', 'Passing', 'Rebounding']
    def get_team_metrics(abbr):
        d = context_data.get(abbr, {'off_rtg': 112, 'def_rtg': 112, 'raw_pace': 99, 'ast_pct': 0.6, 'reb_pct': 0.5})
        return [d['off_rtg']/125, (145-d['def_rtg'])/45, d['raw_pace']/105, d['ast_pct'], d['reb_pct']]
    fig = go.Figure()
    if team_abbr in context_data:
        fig.add_trace(go.Scatterpolar(r=get_team_metrics(team_abbr), theta=categories, fill='toself', name=team_abbr, line_color='#636EFA'))
    if opp_abbr in context_data:
        fig.add_trace(go.Scatterpolar(r=get_team_metrics(opp_abbr), theta=categories, fill='toself', name=opp_abbr, line_color='#EF553B'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark", height=300, margin=dict(l=40, r=40, t=30, b=30), showlegend=True)
    return fig

def plot_poisson_chart(mu, line, cat):
    x = np.arange(0, max(mu * 2.5, line + 5))
    y = poisson.pmf(x, mu)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, marker_color='#636EFA', opacity=0.6, name='Probability'))
    fig.add_vline(x=line - 0.5, line_dash="dash", line_color="#FF4B4B", annotation_text="Market Line")
    fig.update_layout(title=f"Outcome Distribution: {cat.upper()}", template="plotly_dark", height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# --- 5. APP SETUP ---
st.set_page_config(page_title="Sharp Pro v7.5", layout="wide")
team_map = {t['abbreviation']: t['id'] for t in teams.get_teams()}
context_data, lg_avg_pace = get_league_context()
injury_list = get_automated_injury_list()

if 'trigger_scan' not in st.session_state:
    st.session_state.trigger_scan = False

with st.sidebar:
    st.title("🚀 Sharp Pro v7.5")
    st.info(f"📋 **Injury Tracker Active:** {len(injury_list)} OUT.")
    app_mode = st.radio("Analysis Mode", ["Single Player", "Team Value Scanner"])
    st.divider()
    st.subheader("Model Weights")
    stat_cat = st.selectbox("Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    recency_weight = st.slider("Recency Bias", 0.0, 1.0, 0.3)
    manual_usage_boost = st.slider("Manual Usage Tweak", 1.0, 1.3, 1.0, 0.05)
    
    st.divider()
    st.subheader("Scanner Filters")
    min_mpg_filter = st.slider("Min MPG (Scanner Only)", 0, 40, 10)
    
    st.divider()
    total_purse = st.number_input("Purse ($)", value=1000)
    kelly_mult = st.slider("Kelly Fraction", 0.1, 1.0, 0.25)
    proj_minutes = st.slider("Projected Minutes", 10, 48, 32)

# --- 6. EXECUTION ---

if app_mode == "Single Player":
    query = st.text_input("Search Player", "Marcus Smart")
    matches = [p for p in players.get_players() if query.lower() in p['full_name'].lower() and p['is_active']]
    player_choice = st.selectbox("Select Player", matches, format_func=lambda x: x['full_name'])
    
    if player_choice:
        if player_choice['full_name'] in injury_list:
            st.error(f"🛑 {player_choice['full_name']} is OUT.")
        else:
            with st.spinner("Analyzing..."):
                p_df = get_player_stats(player_choice['id'])
                info = commonplayerinfo.CommonPlayerInfo(player_id=player_choice['id']).get_data_frames()[0]
                team_abbr = info['TEAM_ABBREVIATION'].iloc[0]
                opp_abbr, is_home = get_live_matchup(team_abbr, team_map)
                
                star_boost = calculate_dynamic_usage(team_abbr, injury_list)
                total_usage_mult = manual_usage_boost + star_boost
                
                if star_boost > 0:
                    st.warning(f"⚡ Star Out detected! Applied +{int(star_boost*100)}% Auto-Usage Boost for {team_abbr} players.")

                if not p_df.empty:
                    dvp_m = calculate_dvp(info['POSITION'].iloc[0], opp_abbr)
                    t_p = context_data.get(team_abbr, {}).get('raw_pace', lg_avg_pace)
                    o_p = context_data.get(opp_abbr, {}).get('raw_pace', lg_avg_pace)
                    p_m = ((t_p + o_p) / 2) / lg_avg_pace
                    
                    st_lambda, s_avg = get_refined_projection(
                        p_df, proj_minutes, stat_cat, recency_weight, 
                        total_usage_mult, dvp_m, p_m
                    )
                    
                    st.divider()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Matchup", f"{team_abbr} vs {opp_abbr}")
                    c2.metric("Projected", f"{st_lambda}")
                    c3.metric("Total Multiplier", f"{round(dvp_m * p_m * total_usage_mult, 2)}x")
                    c4.metric("Usage Applied", f"{round(total_usage_mult, 2)}x")

                    col_l, col_m, col_r = st.columns([1, 1, 1])
                    with col_l:
                        curr_line = st.number_input("Market Line", value=float(round(st_lambda, 1)), step=0.5)
                        win_p = (1 - poisson.cdf(curr_line - 0.5, st_lambda))
                        st.metric("Win Probability", f"{round(win_p*100, 1)}%")
                        st.plotly_chart(plot_poisson_chart(st_lambda, curr_line, stat_cat), use_container_width=True)
                    with col_m:
                        st.plotly_chart(plot_scout_radar(team_abbr, opp_abbr, context_data), use_container_width=True)
                    with col_r:
                        v_scale = 0.15 if stat_cat == 'points' else 0.05
                        samples = np.random.gamma(st_lambda / (1 + st_lambda * v_scale), 1 + st_lambda * v_scale, 10000)
                        sims = np.random.poisson(samples)
                        
                        # --- UPDATED MONTE CARLO CHART WITH MARKET LINE ---
                        fig_mc = go.Figure(data=[go.Histogram(
                            x=sims, 
                            nbinsx=max(15, int(st_lambda)), 
                            marker_color='#00CC96', 
                            opacity=0.7,
                            name='Simulated Games'
                        )])
                        # Re-inserting the Market Line
                        fig_mc.add_vline(x=curr_line - 0.5, line_dash="dash", line_color="#FF4B4B", annotation_text="Market")
                        fig_mc.update_layout(
                            title=f"Monte Carlo: {stat_cat.upper()}",
                            template="plotly_dark", 
                            height=350,
                            margin=dict(l=10, r=10, t=40, b=10),
                            showlegend=False
                        )
                        st.plotly_chart(fig_mc, use_container_width=True)

else:
    st.header("📋 Team Value Scanner")
    team_choice = st.selectbox("Select Team", sorted(list(team_map.keys())))
    
    if st.button("🚀 Start Scan"):
        st.session_state.trigger_scan = True

    if st.session_state.trigger_scan:
        opp_abbr, is_home = get_live_matchup(team_choice, team_map)
        star_boost = calculate_dynamic_usage(team_choice, injury_list)
        total_usage_mult = manual_usage_boost + star_boost
        
        roster = commonteamroster.CommonTeamRoster(team_id=team_map[team_choice]).get_data_frames()[0]
        scan_results = []
        for _, row in roster.iterrows():
            if row['PLAYER'] in injury_list: continue
            p_df = get_player_stats(row['PLAYER_ID'])
            
            if not p_df.empty:
                actual_mpg = p_df['minutes'].mean()
                if actual_mpg < min_mpg_filter:
                    continue

                dvp_m = calculate_dvp(row['POSITION'], opp_abbr)
                t_p = context_data.get(team_choice, {}).get('raw_pace', lg_avg_pace)
                o_p = context_data.get(opp_abbr, {}).get('raw_pace', lg_avg_pace)
                p_m = ((t_p + o_p) / 2) / lg_avg_pace
                
                proj, avg = get_refined_projection(
                    p_df, proj_minutes, stat_cat, recency_weight, 
                    total_usage_mult, dvp_m, p_m
                )
                
                scan_results.append({
                    "Player": row['PLAYER'], 
                    "MPG": round(actual_mpg, 1),
                    "Proj": proj, 
                    "Season": avg,
                    "Edge": round(proj - avg, 1)
                })
        
        if scan_results:
            df_res = pd.DataFrame(scan_results).sort_values(by="Edge", ascending=False)
            st.dataframe(df_res.style.background_gradient(subset=['Edge'], cmap='RdYlGn'), use_container_width=True)
            fig_edge = go.Figure(go.Bar(x=df_res['Player'], y=df_res['Edge'], marker_color=['#00CC96' if x > 0 else '#EF553B' for x in df_res['Edge']]))
            st.plotly_chart(fig_edge, use_container_width=True)
        st.session_state.trigger_scan = False
