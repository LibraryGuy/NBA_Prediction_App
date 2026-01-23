import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from datetime import datetime, timedelta
import pytz
import requests
import time
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, scoreboardv2, commonteamroster

# --- 1. NEW: REAL-TIME INJURY SCRAPER ---

@st.cache_data(ttl=1800)
def get_automated_injury_list():
    confirmed_out = []
    try:
        # We use requests to get the HTML first, then pass it to pandas
        url = "https://www.cbssports.com/nba/injuries/"
        header = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=header)
        
        # Try different flavors if lxml is missing
        try:
            tables = pd.read_html(response.text, flavor='bs4')
        except ImportError:
            tables = pd.read_html(response.text, flavor='html5lib')
        
        for table in tables:
            if 'Status' in table.columns and 'Player' in table.columns:
                out_players = table[table['Status'].str.contains('Out|Sidelined', case=False, na=False)]
                confirmed_out.extend(out_players['Player'].tolist())
        
        confirmed_out = [name.split('  ')[0].strip() for name in confirmed_out]
        
    except Exception as e:
        st.sidebar.warning(f"Injury Sync: Using manual fallback. (Error: {str(e)[:50]}...)")
        confirmed_out = ["Nikola Jokic", "Ja Morant", "Fred VanVleet", "Ty Jerome"]
        
    return list(set(confirmed_out))

def calculate_dynamic_usage(team_abbr, injury_list):
    # Expanded star map to trigger boosts when players like Jokic or FVV are out
    star_impact_map = {
        "DEN": ["Nikola Jokic", "Jamal Murray"],
        "HOU": ["Alperen Sengun", "Fred VanVleet"],
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
        if any(star in injured for injured in injury_list):
            boost += 0.10 # 10% usage boost for teammates when a star is out
    return round(boost, 2)

# --- 2. SAMPLE SIZE & PROJECTION ENGINE ---

def get_refined_projection(p_df, proj_minutes, stat_cat, weight, usage_boost, dvp, pace):
    if p_df.empty: return 0.0, 0.0
    season_rate = p_df[f'{stat_cat}_per_min'].mean()
    last5_rate = p_df.head(5)[f'{stat_cat}_per_min'].mean()
    weighted_rate = (last5_rate * weight) + (season_rate * (1 - weight))
    
    total_minutes_played = p_df['minutes'].sum()
    reliability_cap = 1.0 if total_minutes_played > 100 else max(0.1, total_minutes_played / 100)
    
    st_lambda = weighted_rate * proj_minutes * dvp * pace * (1 + usage_boost) * reliability_cap
    season_avg = p_df[stat_cat].mean()
    return round(st_lambda, 2), round(season_avg, 2)

# --- 3. CORE UTILITIES ---

@st.cache_data(ttl=3600)
def get_league_context():
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced', season='2024-25', timeout=30).get_data_frames()[0]
        avg_pace = stats['PACE'].mean()
        context_map = {row['TEAM_ABBREVIATION']: {
            'raw_pace': row['PACE'], 'off_rtg': row['OFF_RATING'], 'def_rtg': row['DEF_RATING'],
            'ast_pct': row['AST_PCT'], 'reb_pct': row['REB_PCT']
        } for _, row in stats.iterrows()}
        return context_map, avg_pace
    except Exception: return {}, 99.0

def get_live_matchup(team_abbr, team_map):
    try:
        t_id = team_map.get(team_abbr)
        sb = scoreboardv2.ScoreboardV2(league_id='00', timeout=30)
        board = sb.get_data_frames()[0]
        game = board[(board['HOME_TEAM_ID'] == t_id) | (board['VISITOR_TEAM_ID'] == t_id)]
        if not game.empty:
            is_home = (game.iloc[0]['HOME_TEAM_ID'] == t_id)
            opp_id = game.iloc[0]['VISITOR_TEAM_ID'] if is_home else game.iloc[0]['HOME_TEAM_ID']
            opp_abbr = next((abbr for abbr, tid in team_map.items() if tid == opp_id), "OPP")
            return opp_abbr, is_home
    except: pass
    return "N/A", True

@st.cache_data(ttl=600)
def get_player_stats(p_id):
    try:
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2024-25', timeout=30).get_data_frames()[0]
        if not log.empty:
            log = log.rename(columns={'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'MIN': 'minutes'})
            log['pra'] = log['points'] + log['rebounds'] + log['assists']
            for cat in ['points', 'rebounds', 'assists', 'three_pointers', 'pra']:
                log[f'{cat}_per_min'] = log[cat] / log['minutes'].replace(0, 1)
            return log
    except: return pd.DataFrame()

# --- 4. APP SETUP ---
st.set_page_config(page_title="Sharp Pro v7.5", layout="wide")
team_map = {t['abbreviation']: t['id'] for t in teams.get_teams()}
context_data, lg_avg_pace = get_league_context()
injury_list = get_automated_injury_list()

with st.sidebar:
    st.title("🚀 Sharp Pro v7.5")
    st.success(f"📋 **Injury Scraper Active**")
    st.caption(f"Currently filtering out {len(injury_list)} players.")
    app_mode = st.radio("Analysis Mode", ["Single Player", "Team Value Scanner"])
    stat_cat = st.selectbox("Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
    recency_weight = st.slider("Recency Bias", 0.0, 1.0, 0.3)
    proj_minutes = st.slider("Projected Minutes", 10, 48, 32)
    min_mpg_filter = st.slider("Min MPG (Scanner)", 0, 40, 15)

# --- 5. EXECUTION ---

if app_mode == "Single Player":
    query = st.text_input("Search Player", "Marcus Smart")
    matches = [p for p in players.get_players() if query.lower() in p['full_name'].lower() and p['is_active']]
    player_choice = st.selectbox("Select Player", matches, format_func=lambda x: x['full_name'])
    
    if player_choice:
        is_injured = any(player_choice['full_name'] in injured for injured in injury_list)
        if is_injured:
            st.error(f"🛑 {player_choice['full_name']} is listed as OUT on the Injury Report.")
        else:
            with st.spinner("Calculating..."):
                p_df = get_player_stats(player_choice['id'])
                info = commonplayerinfo.CommonPlayerInfo(player_id=player_choice['id'], timeout=30).get_data_frames()[0]
                team_abbr = info['TEAM_ABBREVIATION'].iloc[0]
                opp_abbr, _ = get_live_matchup(team_abbr, team_map)
                
                usage_boost = calculate_dynamic_usage(team_abbr, injury_list)
                st_lambda, _ = get_refined_projection(p_df, proj_minutes, stat_cat, recency_weight, usage_boost, 1.0, 1.0)
                
                st.metric(f"Projected {stat_cat.title()}", st_lambda, delta=f"{usage_boost*100}% Usage Boost" if usage_boost > 0 else None)
                
                # Parlay Legs
                st.divider()
                st.subheader("🎯 Suggested Parlay Legs")
                cols = st.columns(3)
                lines = [round(st_lambda * 0.7, 1), round(st_lambda * 0.85, 1), round(st_lambda * 0.95, 1)]
                for i, line in enumerate(lines):
                    prob = (1 - poisson.cdf(line - 0.5, st_lambda))
                    cols[i].markdown(f"""<div style="padding:15px; border-radius:10px; border:1px solid #444; background-color:#1e1e1e; text-align:center;">
                        <h4 style="margin:0; color:#00CC96;">Leg {i+1}</h4>
                        <p style="font-size:20px; font-weight:bold; margin:5px 0;">{line}+ {stat_cat.title()}</p>
                        <p style="color:#00CC96; font-weight:bold;">{round(prob*100, 1)}%</p>
                    </div>""", unsafe_allow_html=True)

else:
    st.header("📋 Team Value Scanner")
    team_choice = st.selectbox("Select Team", sorted(list(team_map.keys())))
    
    if st.button("🚀 Start Scan"):
        opp_abbr, _ = get_live_matchup(team_choice, team_map)
        usage_boost = calculate_dynamic_usage(team_choice, injury_list)
        
        roster_call = commonteamroster.CommonTeamRoster(team_id=team_map[team_choice], timeout=60)
        roster = roster_call.get_data_frames()[0]
        
        results = []
        for _, row in roster.iterrows():
            # EXACT NAME MATCH FOR INJURY FILTER
            if any(row['PLAYER'] in injured for injured in injury_list):
                continue
                
            p_df = get_player_stats(row['PLAYER_ID'])
            if not p_df.empty and p_df['minutes'].mean() >= min_mpg_filter:
                proj, avg = get_refined_projection(p_df, proj_minutes, stat_cat, recency_weight, usage_boost, 1.0, 1.0)
                results.append({"Player": row['PLAYER'], "Proj": proj, "Season Avg": avg, "Edge": round(proj-avg, 2)})
            time.sleep(0.2)
        
        if results:
            st.table(pd.DataFrame(results).sort_values(by="Edge", ascending=False))
        else:
            st.warning("No players met the criteria or all active players are injured.")
