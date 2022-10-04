import os
import random
import json
from SharedFunctions import generate_picks, mutate_constants, evaluate_picks

current_season = 2022
starting_week = 2
ending_week = 4

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

base_pyth_constant = 3.0887176944759176
base_uh_oh_multiplier = 1.3755873118229176
base_home_advantage_multiplier = 1.6990493451587603
base_freshness_coefficient = -0.06782574596541502
base_spread_coefficient = 0.07564801240328997

desired_generations = 200
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
                "parameters" : {
                    "pyth_constant": base_pyth_constant,
                    "uh_oh_multiplier": base_uh_oh_multiplier,
                    "home_advantage_multiplier": base_home_advantage_multiplier,
                    "freshness_coefficient": base_freshness_coefficient,
                    "spread_coefficient": base_spread_coefficient,
                    "position_weights": base_position_weights.copy(),
                    "injury_type_weights": base_injury_type_weights.copy()
                }
            }
        ]

    count = 1
    for seeder in seeders:
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
                "value": seeder['parameters'][parameter],
                "generation": generation_counter
            }
            visualization_set.append(vis)
        count += 1

    print("Generating mutations on seeders for this round")
    # generate mutators for the rest of this generation
    for count in range(generation_size - len(generation)):
        seed = random.choice(seeders)
        mutated = mutate_constants(
            seed['parameters']["pyth_constant"],
            seed['parameters']["uh_oh_multiplier"],
            seed['parameters']["home_advantage_multiplier"],
            seed['parameters']["freshness_coefficient"],
            seed['parameters']["position_weights"].copy(),
            seed['parameters']["injury_type_weights"].copy(),
            seed['parameters']["spread_coefficient"]
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
                "candidate": count + len(seeders),
                "parameter": parameter,
                "value": mutated[parameter],
                "generation": generation_counter
            }
            visualization_set.append(vis)

        # foreach of those mutators, generate picks for each week of play
        newpick = {
            "parameters": {
                'pyth_constant': mutated['pyth_constant'],
                'uh_oh_multiplier': mutated['uh_oh_multiplier'],
                'home_advantage_multiplier': mutated['home_advantage_multiplier'],
                'freshness_coefficient': mutated['freshness_coefficient'],
                'position_weights': mutated['position_weights'],
                'injury_type_weights': mutated['injury_type_weights'],
                'spread_coefficient': mutated['spread_coefficient']
            }
        }

        pick_week = starting_week
        while pick_week <= ending_week:
            newpick["week"+str(pick_week)] = {
                "predictions": generate_picks(
                    current_season, pick_week, mutated['pyth_constant'], mutated['uh_oh_multiplier'],
                    mutated['home_advantage_multiplier'], mutated['freshness_coefficient'],
                    mutated['position_weights'], mutated['injury_type_weights'],
                    mutated['spread_coefficient']
                )
            }
            pick_week += 1
        generation.append(newpick)

    # foreach of the mutators, evaluate week 2 ... X and come up with a total score for that mutator
    print("Evaluating generation " + str(generation_counter))
    for prediction_set in generation:
        pick_week = starting_week
        prediction_set['accuracy_score'] = 0
        prediction_set['spread_score'] = 0
        prediction_set['total_money_won'] = 0
        while pick_week <= ending_week:
            prediction_set['week'+str(pick_week)] = evaluate_picks(current_season, pick_week, [prediction_set['week'+str(pick_week)]])[0]
            prediction_set['accuracy_score'] += prediction_set['week'+str(pick_week)]['accuracy_score']
            prediction_set['spread_score'] += abs(prediction_set['week'+str(pick_week)]['spread_score'])
            prediction_set['total_money_won'] += prediction_set['week'+str(pick_week)]['total_money_won']
            pick_week += 1

    # keep the best mutators as seeds for the next round
    leader = 0
    for prediction_set in generation:
        if int(prediction_set['accuracy_score']) > leader:
            leader = int(prediction_set['accuracy_score'])

    print("Leading score for this generation: " + str(leader))

    keep = []
    for prediction_set in generation:
        if prediction_set['accuracy_score'] == leader:
            keep.append(prediction_set)

    print("Candidates with that score: " + str(len(keep)))

    # now lets use the predicted spread to figure out which of the predictions was the most accurate
    sorted_array = sorted(keep, key=lambda pick: pick["total_money_won"], reverse=True)
    if len(keep) > keep_each_gen:
        generation = sorted_array[:keep_each_gen:]
    else:
        generation = keep

    if(len(generation) > 4):
        print("Best money this generation: " + str(generation[0]["total_money_won"]) + ", " + str(generation[1]["total_money_won"]) + ", " + str(generation[2]["total_money_won"]) + ", " + str(generation[3]["total_money_won"]))
    else:
        print("Best money this generation: " + str(generation[0]["total_money_won"]))
    generation_counter = generation_counter + 1

f = open("predictions/genetics/genetics.json", "w")
f.write(json.dumps(generation, indent=4))
f.close()

f = open("predictions/genetics/visualization.json", "w")
f.write(json.dumps(visualization_set, indent=4))
f.close()