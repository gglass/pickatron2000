import random
import json
from SharedFunctions import generate_picks
from statistics import mean

current_season = 2022
week = "18"

base_position_weights = {
    "WR": 3,
    "LT": 4,
    "LG": 3,
    "C": 2,
    "RG": 1,
    "RT": 2,
    "TE": 3,
    "QB": 5,
    "RB": 5,
    "FB": 3,
    "NT": 3,
    "RDE": 1,
    "LOLB": 2,
    "LILB": 2,
    "RILB": 1,
    "ROLB": 2,
    "LCB": 3,
    "RCB": 3,
    "SS": 4,
    "FS": 3,
    "PK": 2,
    "P": 1,
    "LDT": 4,
    "WLB": 5,
    "MLB": 4,
    "SLB": 5,
    "CB": 4,
    "LB": 3,
    "DE": 3,
    "DT": 4,
    "UT": 2,
    "NB": 1,
    "DB": 2,
    "S": 3,
    "DL": 4,
    "H": 2,
    "PR": 2,
    "KR": 2,
    "LS": 2,
    "OT": 3,
    "G": 2
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

base_pyth_constant = 5.494840294411889
base_uh_oh_multiplier = 0.11516040154045012
base_home_advantage_multiplier = 1.8166362407431245
base_freshness_coefficient = 1.6591138156524625
base_spread_coefficient = 1.5571444078514984
base_ls_weight = 1.854183287246492

picks = {
    "predictions": generate_picks(
        current_season,
        week,
        base_pyth_constant,
        base_uh_oh_multiplier,
        base_home_advantage_multiplier,
        base_freshness_coefficient,
        base_position_weights,
        base_injury_type_weights,
        base_spread_coefficient,
        base_ls_weight
    )
}

f = open("predictions/week" + week + "/" + "picks.json", "w")
f.write(json.dumps(picks, indent=4))
f.close()
