import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguegamefinder

@st.cache_data(ttl=3600)
def load_nba_data(player_name, season='2024-25'):
    # Get Player ID
    nba_players = players.get_players()
    p_info = [p for p in nba_players if p['full_name'] == player_name][0]
    p_id = p_info['id']
    
    # Fetch Game Logs
    log = playergamelog.PlayerGameLog(player_id=p_id, season=season)
    df = log.get_data_frames()[0]
    
    # Standardize columns to match your original logic
    df = df.rename(columns={
        'PTS': 'points', 
        'REB': 'rebounds', 
        'AST': 'assists',
        'MATCHUP': 'opponent',
        'GAME_DATE': 'date'
    })
    return df, p_id

def get_rest_multiplier(days_rest):
    """NBA replacement for weather: Performance on Back-to-Backs."""
    if days_rest == 0:  # Back-to-back
        return 0.92, "Back-to-Back (-8%)"
    elif days_rest >= 3:
        return 1.05, "Well Rested (+5%)"
    return 1.0, "Standard Rest"
