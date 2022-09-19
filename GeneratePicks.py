import random
import json
from SharedFunctions import generate_picks

current_season = 2022
week = "3"

base_position_weights = {
    'WR': 3,
    'LT': 3,
    'LG': 3,
    'C': 2,
    'RG': 1,
    'RT': 3,
    'TE': 3,
    'QB': 5,
    'RB': 4,
    'FB': 3,
    'NT': 3,
    'RDE': 2,
    'LOLB': 2,
    'LILB': 2,
    'RILB': 2,
    'ROLB': 2,
    'LCB': 3,
    'RCB': 3,
    'SS': 3,
    'FS': 3,
    'PK': 2,
    'P': 1,
    'LDT': 4,
    'WLB': 4,
    'MLB': 4,
    'SLB': 4,
    'CB': 4,
    'LB': 3,
    'DE': 3,
    'DT': 4,
    'UT': 1,
    'NB': 1,
    'DB': 2,
    'S': 3,
    'DL': 4,
    'H': 1,
    'PR': 2,
    'KR': 2,
    'LS': 2,
    'OT': 3,
    'G': 3,
}

base_injury_type_weights = {
    "Active": 0,
    "Questionable": 0.5,
    "Out": 1,
    "Injured Reserve": 0.8
}

base_pyth_constant = 2.3
base_uh_oh_multiplier = 1
base_home_advantage_multiplier = 1.15
base_freshness_coefficient = 1

picks = generate_picks(
    current_season,
    week,
    base_pyth_constant,
    base_uh_oh_multiplier,
    base_home_advantage_multiplier,
    base_freshness_coefficient,
    base_position_weights,
    base_injury_type_weights
)

f = open("predictions/week" + week + "/" + "picks.json", "w")
f.write(json.dumps(picks, indent=4))
f.close()
