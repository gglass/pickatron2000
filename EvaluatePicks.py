import json
from SharedFunctions import evaluate_picks

current_season = 2022
week = "17"

f = open("predictions/week" + str(week) + "/picks.json", "r")
generation = json.load(f)
f.close()

evaluations = evaluate_picks(current_season, week, [generation], overwrite=True)

f = open("predictions/week" + week + "/" + "evaluations.json", "w")
f.write(json.dumps(evaluations, indent=4))
f.close()
