import random
import json
from SharedFunctions import generate_picks
from statistics import mean

current_season = 2022
week = "3"

base_position_weights = {
    'WR': 3,
    'LT': 4,
    'LG': 3,
    'C': 2,
    'RG': 1,
    'RT': 2,
    'TE': 3,
    'QB': 5,
    'RB': 5,
    'FB': 3,
    'NT': 3,
    'RDE': 1,
    'LOLB': 2,
    'LILB': 2,
    'RILB': 1,
    'ROLB': 2,
    'LCB': 3,
    'RCB': 3,
    'SS': 4,
    'FS': 3,
    'PK': 2,
    'P': 1,
    'LDT': 4,
    'WLB': 5,
    'MLB': 4,
    'SLB': 5,
    'CB': 4,
    'LB': 3,
    'DE': 3,
    'DT': 4,
    'UT': 2,
    'NB': 1,
    'DB': 2,
    'S': 3,
    'DL': 4,
    'H': 2,
    'PR': 2,
    'KR': 2,
    'LS': 2,
    'OT': 3,
    'G': 2,
}

base_injury_type_weights = {
    "Active": 0,
    "Questionable": 0.5,
    "Out": 1,
    "Suspension": 1,
    "Injured Reserve": 0.8,
    "Doubtful": 0.8
}

weekly_coefficients = {
    "base_pyth_constant": [],
    "base_uh_oh_multiplier": [],
    "base_home_advantage_multiplier": [],
    "base_freshness_coefficient": [],
    "base_spread_coefficient": []
}


# set 1
weekly_coefficients["base_pyth_constant"].append(2.86)
weekly_coefficients["base_uh_oh_multiplier"].append(2.25)
weekly_coefficients["base_home_advantage_multiplier"].append(1.47)
weekly_coefficients["base_freshness_coefficient"].append(0.83)
weekly_coefficients["base_spread_coefficient"].append(0.928)

# set 2
weekly_coefficients["base_pyth_constant"].append(3.09)
weekly_coefficients["base_uh_oh_multiplier"].append(2.08)
weekly_coefficients["base_home_advantage_multiplier"].append(1.55)
weekly_coefficients["base_freshness_coefficient"].append(0.44)
weekly_coefficients["base_spread_coefficient"].append(0.84)

base_pyth_constant = mean(weekly_coefficients["base_pyth_constant"])
base_uh_oh_multiplier = mean(weekly_coefficients["base_uh_oh_multiplier"])
base_home_advantage_multiplier = mean(weekly_coefficients["base_home_advantage_multiplier"])
base_freshness_coefficient = mean(weekly_coefficients["base_freshness_coefficient"])
base_spread_coefficient = mean(weekly_coefficients["base_spread_coefficient"])

picks = generate_picks(
    current_season,
    week,
    base_pyth_constant,
    base_uh_oh_multiplier,
    base_home_advantage_multiplier,
    base_freshness_coefficient,
    base_position_weights,
    base_injury_type_weights,
    base_spread_coefficient
)

f = open("predictions/week" + week + "/" + "picks.json", "w")
f.write(json.dumps(picks, indent=4))
f.close()
