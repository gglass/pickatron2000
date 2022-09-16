import json

f = open("predictions/week1/genetics.json", "r")
data = json.load(f)
f.close()

f = open('position_weights.csv', "w")
f.write("WR,LT,LG,C,RG,RT,TE,QB,RB,FB,NT,RDE,LOLB,LILB,RILB,ROLB,LCB,RCB,SS,FS,PK,P,LDT,WLB,MLB,SLB,CB,LB,DE,DT,UT,NB,DB,S,DL,H,PR,KR,LS,OT,G")
f.write("\n")
for candidate in data:
    for position in candidate['position_weights']:
        f.write(str(candidate['position_weights'][position]) + ",")
    f.write("\n")
f.close()
