import os
import random
import json
from SharedFunctions import generate_picks, mutate_constants, evaluate_picks

current_season = 2022
week = "2"

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

base_pyth_constant = 2.86
base_uh_oh_multiplier = 2.26
base_home_advantage_multiplier = 1.47
base_freshness_coefficient = 0.83
base_spread_coefficient = 0.928

desired_generations = 50
generation_size = 20
keep_each_gen = 10

generation_counter = 1
generation = []
visualization_set = []

while generation_counter <= desired_generations:
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
                "spread_coefficient": base_spread_coefficient,
                "position_weights": base_position_weights.copy(),
                "injury_type_weights": base_injury_type_weights.copy()
            }
        ]

    # generate a set of 100 picks for this generation
    for count in range(generation_size - len(generation)):
        seed = random.choice(seeders)

        mutated = mutate_constants(
            seed["pyth_constant"],
            seed["uh_oh_multiplier"],
            seed["home_advantage_multiplier"],
            seed["freshness_coefficient"],
            seed["position_weights"].copy(),
            seed["injury_type_weights"].copy(),
            seed["spread_coefficient"]
        )

        # parameters = mutated['position_weights'].keys()
        #
        # for parameter in parameters:
        #     vis = {
        #         "candidate": count,
        #         "parameter": parameter,
        #         "value": mutated['position_weights'][parameter],
        #         "generation": generation_counter
        #     }
        #     visualization_set.append(vis)

        parameters = [
            "pyth_constant",
            "uh_oh_multiplier",
            "home_advantage_multiplier",
            "freshness_coefficient",
            "spread_coefficient"
        ]
        for parameter in parameters:
            vis = {
                "candidate": count,
                "parameter": parameter,
                "value": mutated[parameter],
                "generation": generation_counter
            }
            visualization_set.append(vis)

        generation.append(
            generate_picks(
                current_season, week, mutated['pyth_constant'], mutated['uh_oh_multiplier'],
                mutated['home_advantage_multiplier'], mutated['freshness_coefficient'], mutated['position_weights'],
                mutated['injury_type_weights'], mutated['spread_coefficient']
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

    # now lets use the predicted spread to figure out which of the predictions was the most accurate
    sorted_array = sorted(generation, key=lambda pick: abs(pick["spread_score"]))

    if len(generation) > keep_each_gen:
        generation = sorted_array[:keep_each_gen:]

    if(len(generation) > 4):
        print("Best spreads this generation: " + str(generation[0]["spread_score"]) + ", " + str(generation[1]["spread_score"]) + ", " + str(generation[2]["spread_score"]) + ", " + str(generation[3]["spread_score"]))
    else:
        print("Best spread this generation: " + str(generation[0]["spread_score"]))

    generation_counter = generation_counter + 1

best_performer = generation[0]

f = open("predictions/week" + week + "/" + "genetics.json", "w")
f.write(json.dumps(best_performer, indent=4))
f.close()

f = open("predictions/week" + week + "/" + "visualization.json", "w")
f.write(json.dumps(visualization_set, indent=4))
f.close()