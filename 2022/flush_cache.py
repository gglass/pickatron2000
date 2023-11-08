import json



f = open("predictions/week1/genetics.json", "r")
data = json.load(f)
f.close()

f = open('position_constants.csv', "w")
f.write("pyth, uhoh, homefield")
f.write("\n")
for candidate in data:
    f.write(str(candidate['pyth_constant']) + "," + str(candidate['uh_oh_multiplier']) + "," + str(candidate['home_advantage_multiplier']))
    f.write("\n")
f.close()
