import hashlib
import json
from SharedFunctions import evaluate_picks, get_or_fetch_from_cache

current_season = 2022
week = "2"

espn_api_base_url = "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
team_links = get_or_fetch_from_cache(espn_api_base_url + "teams?limit=32")
for link in team_links['items']:
    team_info = get_or_fetch_from_cache(link['$ref'])
    injuries_url = espn_api_base_url + "teams/" + team_info['id'] + "/injuries"
    injuries = get_or_fetch_from_cache(injuries_url, "caches/week"+str(week))
