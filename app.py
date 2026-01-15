import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, commonteamroster, scoreboardv2

# --- 1. CORE ENGINE (UPGRADED) ---
@st.cache_data(ttl=3600)
def get_league_context():
    """Fetches real-time 2026 Defensive Ratings AND Pace for all teams."""
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced', 
            season='2025-26'
        ).get_data_frames()[0]
        
        avg_def = stats['DEF_RATING'].mean()
        avg_pace = stats['PACE'].mean()
        
        # Create a combined lookup for SOS and Pace
        context_map = {}
        for _, row in stats.iterrows():
            context_map[row['TEAM_ABBREVIATION']] = {
                'sos': row['DEF_RATING'] / avg_def,
                'pace_factor': row['PACE'] / avg_pace,
                'raw_pace': row['PACE']
            }
        return context_map, avg_pace
    except:
        return {t['abbreviation']: {'sos': 1.0, 'pace_factor': 1.0, 'raw_pace': 99.0} for t in teams.get_teams()}, 99.0

@st.cache_data(ttl=600)
def get_player_stats(p_id):
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
        log = playergamelog.PlayerGameLog(player_id=p_id, season='2025-26').get_data_frames()[0]
        if not log.empty:
            log = log.rename(columns={'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'FG3M': 'three_pointers', 'FGA': 'fga', 'FTA': 'fta', 'TOV': 'tov'})
            log['pra'] = log['points'] + log['rebounds'] + log['assists']
            log['usage'] = log['fga'] + (0.44 * log['fta']) + log['tov']
            log['pps'] = log['points'] / log['fga'].replace(0, 1)
        return log, info['TEAM_ABBREVIATION'].iloc[0], info['POSITION'].iloc[0], info['HEIGHT'].iloc[0]
    except: return pd.DataFrame(), None, None, None

def get_live_matchup(team_abbr, team_map):
    try:
        # Note: update to datetime.now() for production
        board = scoreboardv2.ScoreboardV2(game_date='2026-01-14').get_data_frames()[0]
        t_id = team_map.get(team_abbr)
        game = board[(board['HOME_TEAM_ID'] == t_id) | (board['VISITOR_TEAM_ID'] == t_id)]
        if not game.empty:
            is_home = (game.iloc[0]['HOME_TEAM_ID'] == t_id)
            opp_id = game.iloc[0]['VISITOR_TEAM_ID'] if is_home else game.iloc[0]['HOME_TEAM_ID']
            opp_abbr = [abbr for abbr, id in team_map.items() if id == opp_id][0]
            return opp_abbr, is_home
    except: pass
    return None, True

# --- 2. LAYOUT ---
st.set_page_config(page_title="Sharp Pro v4.8", layout="wide")
team_map = {t['abbreviation']: t['id'] for t in teams.get_teams()}
context_data, lg_avg_pace = get_league_context()

with st.sidebar:
    st.title("🛡️ Pro Hub v4.8")
    total_purse = st.number_input("Purse ($)", value=1000)
    kelly_mult = st.slider("Kelly Fraction", 0.1, 1.0, 0.5)
    st.divider()
    mode = st.radio("Switch View", ["Single Player", "Team Scout Radar", "Box Score Scraper"])
    stat_cat = st.selectbox("Stat Category", ["points", "rebounds", "assists", "three_pointers", "pra"])
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
            p_mean = p_df[stat_cat].mean()
            
            p_team_pace = context_data.get(team_abbr, {}).get('raw_pace', lg_avg_pace)
            o_team_pace = context_data.get(opp_abbr, {}).get('raw_pace', lg_avg_pace)
            matchup_pace = (p_team_pace + o_team_pace) / 2
            pace_multiplier = matchup_pace / lg_avg_pace
            current_sos = context_data.get(opp_abbr, {}).get('sos', 1.0)

            st.divider()
            if opp_abbr:
                st.success(f"📅 **Matchup:** {team_abbr} vs {opp_abbr} | Pace: **{round(matchup_pace, 1)}** | SOS: **{round(current_sos, 2)}x**")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Player", f"{team_abbr} | {pos}")
            m2.metric("Season Avg", round(p_mean, 1))
            m3.metric("Last Game", p_df[stat_cat].iloc[0])
            m4.metric("Height", height)

            b1, b2, b3 = st.columns([2, 1, 1])
            curr_line = b1.number_input("Vegas Line", value=float(round(p_mean, 1)))
            
            home_adv = 1.03 if is_home else 0.97
            st_lambda = p_mean * (1.10 if vol_boost else 1.0) * injury_impact * current_sos * pace_multiplier * home_adv
            
            win_p = (1 - poisson.cdf(curr_line - 0.5, st_lambda))
            b2.metric("Win Prob", f"{round(win_p*100, 1)}%")
            b3.metric("Rec. Stake", f"${round(total_purse * kelly_mult * 0.05, 2)}")

            st.divider()
            t_col, e_col = st.columns(2)
            with t_col:
                fig_t = go.Figure(go.Scatter(y=p_df[stat_cat].head(10).iloc[::-1], mode='lines+markers', line=dict(color='#00ff96')))
                fig_t.update_layout(title="Last 10 Games Trend", template="plotly_dark", height=300)
                st.plotly_chart(fig_t, use_container_width=True)
            with e_col:
                fig_e = go.Figure(go.Scatter(x=p_df['usage'], y=p_df['pps'], mode='markers', marker=dict(color='#ffaa00')))
                fig_e.update_layout(title="Efficiency Matrix", template="plotly_dark", height=300)
                st.plotly_chart(fig_e, use_container_width=True)

            st.write("### 🎯 Full-Scale Outcome Distribution")
            sims = np.random.poisson(st_lambda, 10000)
            fig_mc = go.Figure(go.Histogram(x=sims, nbinsx=35, marker_color='#00ff96', opacity=0.6))
            fig_mc.add_vline(x=curr_line, line_color="red", line_dash="dash", line_width=4)
            fig_mc.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_mc, use_container_width=True)

# --- 4. MODE: TEAM SCOUT RADAR (ALPHA ROTATION UPDATE) ---
elif mode == "Team Scout Radar":
    st.header("🚀 Team Scout Radar")
    sel_team = st.selectbox("Select Team to Scan", sorted(list(team_map.keys())))
    
    if st.button("Generate Alpha Roster Analysis"):
        with st.spinner(f"Identifying {sel_team} rotation alphas..."):
            try:
                # 1. Fetch raw roster
                roster_data = commonteamroster.CommonTeamRoster(team_id=team_map[sel_team]).get_data_frames()[0]
                
                # 2. Matchup & Context
                opp, home = get_live_matchup(sel_team, team_map)
                p_team_pace = context_data.get(sel_team, {}).get('raw_pace', lg_avg_pace)
                o_team_pace = context_data.get(opp, {}).get('raw_pace', lg_avg_pace) if opp else lg_avg_pace
                matchup_pace = (p_team_pace + o_team_pace) / 2
                pace_multiplier = matchup_pace / lg_avg_pace
                sos_adj = context_data.get(opp, {}).get('sos', 1.0) if opp else 1.0
                home_factor = 1.03 if home else 0.97
                
                # 3. Alpha Identification: Scan everyone to find the real rotation leaders
                full_team_data = []
                for _, row in roster_data.iterrows():
                    try:
                        p_log, _, _, _ = get_player_stats(row['PLAYER_ID'])
                        if not p_log.empty:
                            avg_stat = p_log[stat_cat].mean()
                            full_team_data.append({
                                "Player": row['PLAYER'],
                                "Avg": avg_stat,
                                "p_id": row['PLAYER_ID']
                            })
                    except: continue
                
                # Sort by average descending (Alphas first) and take top 10
                alpha_rotation = sorted(full_team_data, key=lambda x: x['Avg'], reverse=True)[:10]
                
                # 4. Final Projection Loop
                radar_results = []
                for player in alpha_rotation:
                    m = player['Avg']
                    proj = m * injury_impact * sos_adj * pace_multiplier * home_factor
                    prob = (1 - poisson.cdf(m - 0.5, proj)) * 100
                    radar_results.append({
                        "Player": player['Player'], 
                        "Avg": round(m, 1), 
                        "Proj": round(proj, 1), 
                        "Win%": f"{round(prob, 1)}%"
                    })

                if radar_results:
                    df_radar = pd.DataFrame(radar_results)
                    st.subheader(f"📋 Alpha Rotation: {sel_team} vs {opp if opp else 'N/A'}")
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.dataframe(df_radar, use_container_width=True, hide_index=True)
                    with c2:
                        df_radar['WinVal'] = df_radar['Win%'].str.replace('%','').astype(float)
                        fig_radar = go.Figure(go.Bar(x=df_radar['Player'], y=df_radar['WinVal'], marker_color='#00ff96'))
                        fig_radar.update_layout(template="plotly_dark", height=400, title="Projected Edge (Pace/SOS Adjusted)")
                        st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.warning("Could not find enough active data for this roster.")
            except Exception as e:
                st.error(f"Error executing Alpha Radar: {e}")

# --- 5. MODE: BOX SCORE SCRAPER ---
elif mode == "Box Score Scraper":
    st.info("Box Score Scraper logic can be implemented here.")
