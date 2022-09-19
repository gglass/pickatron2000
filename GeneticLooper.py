import random
import json
from SharedFunctions import generate_picks, mutate_constants, evaluate_picks

current_season = 2022
week = "2"

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

generation_counter = 1
generation = []

while generation_counter < 1001:
    print("Generating picks for generation " + str(generation_counter))

    if generation:
        seeders = generation.copy()
    else:
        seeders = [
            {
                "pyth_constant": base_pyth_constant,
                "uh_oh_multiplier": base_uh_oh_multiplier,
                "home_advantage_multiplier": base_home_advantage_multiplier,
                "freshness_coefficient": base_freshness_coefficient,
                "position_weights": base_position_weights.copy(),
                "injury_type_weights": base_injury_type_weights.copy()
            }
        ]

    # generate a set of 100 picks for this generation
    for count in range(40 - len(generation)):
        seed = random.choice(seeders)

        mutated = mutate_constants(
            seed["pyth_constant"],
            seed["uh_oh_multiplier"],
            seed["home_advantage_multiplier"],
            seed["freshness_coefficient"],
            seed["position_weights"].copy(),
            seed["injury_type_weights"].copy()
        )

        generation.append(
            generate_picks(
                current_season, week, mutated['pyth_constant'], mutated['uh_oh_multiplier'],
                mutated['home_advantage_multiplier'], mutated['freshness_coefficient'], mutated['position_weights'],
                mutated['injury_type_weights']
            )
        )

    # evaluate this generation
    print("Evaluating generation " + str(generation_counter))
    evaluations = evaluate_picks(current_season, week, generation)

    leader = 0
    for score in evaluations.keys():
        if int(score) > leader:
            leader = int(score)

    print("Leading score for this generation: " + str(leader))
    generation = evaluations[leader].copy()
    print("Candidates with that score: " + str(len(generation)))
    if len(generation) > 20:
        generation = random.choices(generation, k=20)

    generation_counter = generation_counter + 1

best_performer = random.choice(generation)

f = open("predictions/week" + week + "/" + "genetics.json", "w")
f.write(json.dumps(best_performer, indent=4))
f.close()
