import hashlib
import json

teams = {
    "Atlanta" : {
        "id": 1
    },
    "Buffalo": {
        "id": 2
    },
    "Chicago": {
        "id": 3
    },
    "Cincinnati": {
        "id": 4
    },
    "Cleveland": {
        "id": 5
    },
    "Dallas": {
        "id": 6
    },
    "Denver": {
        "id": 7
    },
    "Detroit": {
        "id": 8
    },
    "Green Bay": {
        "id": 9
    },
    "Tennessee": {
        "id": 10
    },
    "Indianapolis": {
        "id": 11
    },
    "Kansas City": {
        "id": 12
    },
    "Las Vegas": {
        "id": 13
    },
    "L.A. Rams": {
        "id": 14
    },
    "Miami": {
        "id": 15
    },
    "Minnesota": {
        "id": 16
    },
    "New England": {
        "id": 17
    },
    "New Orleans": {
        "id": 18
    },
    "N.Y. Giants": {
        "id": 19
    },
    "N.Y. Jets": {
        "id": 20
    },
    "Philadelphia": {
        "id": 21
    },
    "Arizona": {
        "id": 22
    },
    "Pittsburgh": {
        "id": 23
    },
    "L.A. Chargers": {
        "id": 24
    },
    "San Francisco": {
        "id": 25
    },
    "Seattle": {
        "id": 26
    },
    "Tampa Bay": {
        "id": 27
    },
    "Washington": {
        "id": 28
    },
    "Carolina": {
        "id": 29
    },
    "Jacksonville": {
        "id": 30
    },
    "Baltimore": {
        "id": 33
    },
    "Houston": {
        "id": 34
    }
}
records = {
    "2": {
        "Miami":[1,0,20,7],
        "Buffalo":[1,0,31,10],
        "N.Y. Jets":[0,1,9,24],
        "New England":[0,1,7,20],
        "Pittsburgh":[1,0,23,20],
        "Baltimore":[1,0,24,9],
        "Cleveland":[1,0,26,24],
        "Cincinnati":[0,1,20,23],
        "Houston":[0,0,20,20],
        "Indianapolis":[0,0,20,20],
        "Tennessee":[0,1,20,21],
        "Jacksonville":[0,1,22,28],
        "L.A. Chargers":[1,0,24,19],
        "Kansas City":[1,0,44,21],
        "Denver":[0,1,16,17],
        "Las Vegas":[0,1,19,24],
        "Philadelphia":[1,0,38,35],
        "Washington":[1,0,28,22],
        "N.Y. Giants":[1,0,21,20],
        "Dallas":[0,1,3,19],
        "Minnesota":[1,0,23,7],
        "Chicago":[1,0,19,10],
        "Detroit":[0,1,35,38],
        "Green Bay":[0,1,7,23],
        "New Orleans":[1,0,27,26],
        "Tampa Bay":[1,0,19,3],
        "Carolina":[0,1,24,26],
        "Atlanta":[0,1,26,27],
        "Seattle":[1,0,17,16],
        "Arizona":[0,1,21,44],
        "L.A. Rams":[0,1,10,31],
        "San Francisco":[0,1,10,19]
    },
    "3": {
        "Miami":[2,0,62,45],
        "Buffalo":[2,0,72,17],
        "N.Y. Jets":[1,1,40,54],
        "New England":[1,1,24,34],
        "Pittsburgh":[1,1,37,37],
        "Baltimore":[1,1,62,51],
        "Cleveland":[1,1,56,55],
        "Cincinnati":[0,2,37,43],
        "Jacksonville":[1,1,46,28],
        "Houston":[0,1,29,36],
        "Indianapolis":[0,1,20,44],
        "Tennessee":[0,2,27,62],
        "Kansas City":[2,0,71,45],
        "L.A. Chargers":[1,1,48,46],
        "Denver":[1,1,32,26],
        "Las Vegas":[0,2,42,53],
        "Philadelphia":[2,0,62,42],
        "N.Y. Giants":[2,0,40,36],
        "Washington":[1,1,55,58],
        "Dallas":[1,1,23,36],
        "Minnesota":[1,1,30,31],
        "Green Bay":[1,1,34,33],
        "Detroit":[1,1,71,65],
        "Chicago":[1,1,29,37],
        "Tampa Bay":[2,0,39,13],
        "New Orleans":[1,1,37,46],
        "Carolina":[0,2,40,45],
        "Atlanta":[0,2,53,58],
        "San Francisco":[1,1,37,26],
        "L.A. Rams":[1,1,41,58],
        "Arizona":[1,1,50,67],
        "Seattle":[1,1,24,43]
    },
    "4": {
        "Miami":[3,0,83,64],
        "Buffalo":[2,1,91,38],
        "N.Y. Jets":[1,2,52,81],
        "New England":[1,2,50,71],
        "Cleveland":[2,1,85,72],
        "Baltimore":[2,1,99,77],
        "Pittsburgh":[1,2,54,66],
        "Cincinnati":[1,2,64,55],
        "Jacksonville":[2,1,84,38],
        "Indianapolis":[1,1,40,61],
        "Tennessee":[1,2,51,84],
        "Houston":[0,2,49,59],
        "Kansas City":[2,1,88,65],
        "Denver":[2,1,43,36],
        "L.A. Chargers":[1,2,58,84],
        "Las Vegas":[0,3,64,77],
        "Philadelphia":[3,0,86,50],
        "Dallas":[2,1,46,52],
        "N.Y. Giants":[2,1,56,59],
        "Washington":[1,2,63,82],
        "Minnesota":[2,1,58,55],
        "Green Bay":[2,1,48,45],
        "Chicago":[2,1,52,57],
        "Detroit":[1,2,95,93],
        "Tampa Bay":[2,1,51,27],
        "Carolina":[1,2,62,59],
        "New Orleans":[1,2,51,68],
        "Atlanta":[1,2,80,81],
        "L.A. Rams":[2,1,61,70],
        "San Francisco":[1,2,47,37],
        "Arizona":[1,2,62,87],
        "Seattle":[1,2,47,70]
    },
    "5": {
        "Miami":[3,1,98,91],
        "Buffalo":[3,1,114,58],
        "N.Y. Jets":[2,2,76,101],
        "New England":[1,3,74,98],
        "Cleveland":[2,2,105,95],
        "Baltimore":[2,2,119,100],
        "Cincinnati":[2,2,91,70],
        "Pittsburgh":[1,3,74,90],
        "Jacksonville":[2,2,105,67],
        "Tennessee":[2,2,75,101],
        "Indianapolis":[1,2,57,85],
        "Houston":[0,3,73,93],
        "Kansas City":[3,1,129,96],
        "L.A. Chargers":[2,2,92,108],
        "Denver":[2,2,66,68],
        "Las Vegas":[1,3,96,100],
        "Philadelphia":[4,0,115,71],
        "Dallas":[3,1,71,62],
        "N.Y. Giants":[3,1,76,71],
        "Washington":[1,3,73,107],
        "Minnesota":[3,1,86,80],
        "Green Bay":[3,1,75,69],
        "Chicago":[2,2,64,77],
        "Detroit":[1,3,140,141],
        "Tampa Bay":[2,2,82,68],
        "Atlanta":[2,2,103,101],
        "Carolina":[1,3,78,85],
        "New Orleans":[1,3,76,96],
        "L.A. Rams":[2,1,61,70],
        "Arizona":[2,2,88,103],
        "Seattle":[2,2,95,115],
        "San Francisco":[1,2,47,37]
    },
    "6": {
        "Buffalo":[4,1,152,61],
        "N.Y. Jets":[3,2,116,118],
        "Miami":[3,2,115,131],
        "New England":[2,3,103,98],
        "Baltimore":[3,2,138,117],
        "Cleveland":[2,3,133,125],
        "Cincinnati":[2,3,108,89],
        "Pittsburgh":[1,4,77,128],
        "Tennessee":[3,2,96,118],
        "Indianapolis":[2,2,69,94],
        "Jacksonville":[2,3,111,80],
        "Houston":[1,3,86,99],
        "Kansas City":[4,1,159,125],
        "L.A. Chargers":[3,2,122,136],
        "Denver":[2,3,75,80],
        "Las Vegas":[1,4,125,130],
        "Philadelphia":[5,0,135,88],
        "Dallas":[4,1,93,72],
        "N.Y. Giants":[4,1,103,93],
        "Washington":[2,4,102,135],
        "Minnesota":[4,1,115,102],
        "Green Bay":[3,2,97,96],
        "Chicago":[2,4,93,118],
        "Detroit":[1,4,140,170],
        "Tampa Bay":[3,2,103,83],
        "New Orleans":[2,3,115,128],
        "Atlanta":[2,3,118,122],
        "Carolina":[1,4,93,122],
        "San Francisco":[3,2,108,61],
        "L.A. Rams":[2,3,80,116],
        "Arizona":[2,3,105,123],
        "Seattle":[2,3,127,154]
    },
    "7": {
        "Buffalo":[5,1,176,81],
        "N.Y. Jets":[4,2,143,128],
        "Miami":[3,3,131,155],
        "New England":[3,3,141,113],
        "Baltimore":[3,3,158,141],
        "Cincinnati":[3,3,138,115],
        "Cleveland":[2,4,148,163],
        "Pittsburgh":[2,4,97,146],
        "Tennessee":[3,2,96,118],
        "Indianapolis":[3,2,103,121],
        "Jacksonville":[2,4,138,114],
        "Houston":[1,3,86,99],
        "Kansas City":[4,2,179,149],
        "L.A. Chargers":[4,2,141,152],
        "Denver":[2,4,91,99],
        "Las Vegas":[1,4,125,130],
        "Philadelphia":[6,0,161,105],
        "N.Y. Giants":[5,1,127,113],
        "Dallas":[4,2,110,98],
        "Washington":[2,4,102,135],
        "Minnesota":[5,1,139,118],
        "Green Bay":[3,3,107,123],
        "Chicago":[2,4,93,118],
        "Detroit":[1,4,140,170],
        "Tampa Bay":[3,3,121,103],
        "Atlanta":[3,3,146,136],
        "New Orleans":[2,4,141,158],
        "Carolina":[1,5,103,146],
        "San Francisco":[3,3,122,89],
        "L.A. Rams":[3,3,104,126],
        "Seattle":[3,3,146,163],
        "Arizona":[2,4,114,142]
    },
    "8": {
        "Buffalo":[5,1,176,81],
        "N.Y. Jets":[5,2,159,137],
        "Miami":[4,3,147,165],
        "New England":[3,4,155,146],
        "Baltimore":[4,3,181,161],
        "Cincinnati":[4,3,173,132],
        "Cleveland":[2,5,168,186],
        "Pittsburgh":[2,5,107,162],
        "Tennessee":[4,2,115,128],
        "Indianapolis":[3,3,113,140],
        "Jacksonville":[2,5,155,137],
        "Houston":[1,4,106,137],
        "Kansas City":[5,2,223,172],
        "L.A. Chargers":[4,3,164,189],
        "Las Vegas":[2,4,163,150],
        "Denver":[2,5,100,115],
        "Philadelphia":[6,0,161,105],
        "N.Y. Giants":[6,1,150,130],
        "Dallas":[5,2,134,104],
        "Washington":[3,4,125,156],
        "Minnesota":[5,1,139,118],
        "Green Bay":[3,4,128,146],
        "Chicago":[3,4,126,132],
        "Detroit":[1,5,146,194],
        "Tampa Bay":[3,4,124,124],
        "Atlanta":[3,4,163,171],
        "Carolina":[2,5,124,149],
        "New Orleans":[2,5,175,200],
        "Seattle":[4,3,183,186],
        "L.A. Rams":[3,3,104,126],
        "San Francisco":[3,4,145,133],
        "Arizona":[3,4,156,176]
    },
    "9": {
        "Buffalo":[6,1,203,98],
        "N.Y. Jets":[5,3,176,159],
        "Miami":[5,3,178,192],
        "New England":[4,4,177,163],
        "Baltimore":[5,3,208,183],
        "Cincinnati":[4,4,186,164],
        "Cleveland":[3,5,200,199],
        "Pittsburgh":[2,6,120,197],
        "Tennessee":[5,2,132,138],
        "Indianapolis":[3,4,129,157],
        "Jacksonville":[2,6,172,158],
        "Houston":[1,5,116,154],
        "Kansas City":[5,2,223,172],
        "L.A. Chargers":[4,3,164,189],
        "Denver":[3,5,121,132],
        "Las Vegas":[2,5,163,174],
        "Philadelphia":[7,0,196,118],
        "Dallas":[6,2,183,133],
        "N.Y. Giants":[6,2,163,157],
        "Washington":[4,4,142,172],
        "Minnesota":[6,1,173,144],
        "Green Bay":[3,5,145,173],
        "Chicago":[3,5,155,181],
        "Detroit":[1,6,173,225],
        "Atlanta":[4,4,200,205],
        "Tampa Bay":[3,5,146,151],
        "New Orleans":[3,5,199,200],
        "Carolina":[2,6,158,186],
        "Seattle":[5,3,210,199],
        "San Francisco":[4,4,176,147],
        "L.A. Rams":[3,4,118,157],
        "Arizona":[3,5,182,210]
    },
    "10": {
        "Buffalo":[6,2,220,118],
        "N.Y. Jets":[6,3,196,176],
        "Miami":[6,3,213,224],
        "New England":[5,4,203,166],
        "Baltimore":[6,3,235,196],
        "Cincinnati":[5,4,228,185],
        "Cleveland":[3,5,200,199],
        "Pittsburgh":[2,6,120,197],
        "Tennessee":[5,3,149,158],
        "Indianapolis":[3,5,132,183],
        "Jacksonville":[3,6,199,178],
        "Houston":[1,6,133,183],
        "Kansas City":[6,2,243,189],
        "L.A. Chargers":[5,3,184,206],
        "Denver":[3,5,121,132],
        "Las Vegas":[2,6,183,201],
        "Philadelphia":[8,0,225,135],
        "Dallas":[6,2,183,133],
        "N.Y. Giants":[6,2,163,157],
        "Washington":[4,5,159,192],
        "Minnesota":[7,1,193,161],
        "Green Bay":[3,6,154,188],
        "Chicago":[3,6,187,216],
        "Detroit":[2,6,188,234],
        "Tampa Bay":[4,5,162,164],
        "Atlanta":[4,5,217,225],
        "New Orleans":[3,6,212,227],
        "Carolina":[2,7,179,228],
        "Seattle":[6,3,241,220],
        "San Francisco":[4,4,176,147],
        "L.A. Rams":[3,5,131,173],
        "Arizona":[3,6,203,241]
    },
    "11": {
        "Miami":[7,3,252,241],
        "N.Y. Jets":[6,3,196,176],
        "Buffalo":[6,3,250,151],
        "New England":[5,4,203,166],
        "Baltimore":[6,3,235,196],
        "Cincinnati":[5,4,228,185],
        "Cleveland":[3,6,217,238],
        "Pittsburgh":[3,6,140,207],
        "Tennessee":[6,3,166,168],
        "Indianapolis":[4,5,157,203],
        "Jacksonville":[3,7,216,205],
        "Houston":[1,7,149,207],
        "Kansas City":[7,2,270,206],
        "L.A. Chargers":[5,4,200,228],
        "Denver":[3,6,131,149],
        "Las Vegas":[2,7,203,226],
        "Philadelphia":[8,0,225,135],
        "N.Y. Giants":[7,2,187,173],
        "Dallas":[6,3,211,164],
        "Washington":[4,5,159,192],
        "Minnesota":[8,1,226,191],
        "Green Bay":[4,6,185,216],
        "Detroit":[3,6,219,264],
        "Chicago":[3,7,217,247],
        "Tampa Bay":[5,5,183,180],
        "Atlanta":[4,6,232,250],
        "Carolina":[3,7,204,243],
        "New Orleans":[3,7,222,247],
        "Seattle":[6,4,257,241],
        "San Francisco":[5,4,198,163],
        "Arizona":[4,6,230,258],
        "L.A. Rams":[3,6,148,200]
    },
    "12": {
        "Miami":[7,3,252,241],
        "Buffalo":[7,3,281,174],
        "New England":[6,4,213,169],
        "N.Y. Jets":[6,4,199,186],
        "Baltimore":[7,3,248,199],
        "Cincinnati":[6,4,265,215],
        "Cleveland":[3,7,240,269],
        "Pittsburgh":[3,7,170,244],
        "Tennessee":[7,3,193,185],
        "Indianapolis":[4,6,173,220],
        "Jacksonville":[3,7,216,205],
        "Houston":[1,8,159,230],
        "Kansas City":[8,2,300,233],
        "L.A. Chargers":[5,5,227,258],
        "Las Vegas":[3,7,225,242],
        "Denver":[3,7,147,171],
        "Philadelphia":[9,1,263,183],
        "Dallas":[7,3,251,167],
        "N.Y. Giants":[7,3,205,204],
        "Washington":[6,5,214,223],
        "Minnesota":[8,2,229,231],
        "Detroit":[4,6,250,282],
        "Green Bay":[4,7,202,243],
        "Chicago":[3,8,241,274],
        "Tampa Bay":[5,5,183,180],
        "Atlanta":[5,6,259,274],
        "New Orleans":[4,7,249,267],
        "Carolina":[3,8,207,256],
        "San Francisco":[6,4,236,173],
        "Seattle":[6,4,257,241],
        "Arizona":[4,7,240,296],
        "L.A. Rams":[3,7,168,227]
    }
}

for week in range(2,13):
    weeklyrecords = records[str(week)]
    for team, data in weeklyrecords.items():
        id = teams[team]["id"]
        url = "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2022/types/2/teams/"+str(id)+"/records/0/?lang=en&region=us"
        file_key = hashlib.md5(url.encode('UTF-8')).hexdigest()
        f = open("caches/week" + str(week) + "/" + file_key, "w")
        teamoutput = {
            "stats": [
                {
                    "shortDisplayName": "W",
                    "value": round(data[0], 1),
                    "displayValue": str(data[0])
                },
                {
                    "shortDisplayName": "L",
                    "value": round(data[1], 1),
                    "displayValue": str(data[1])
                },
                {
                    "shortDisplayName": "PF",
                    "value": round(data[2], 1),
                    "displayValue": str(data[2])
                },
                {
                    "shortDisplayName": "PA",
                    "value": round(data[3], 1),
                    "displayValue": str(data[3])
                },
                {
                    "shortDisplayName": "GP",
                    "value": round(data[0]+data[1], 1),
                    "displayValue": str(data[0]+data[1])
                }
            ]
        }
        f.write(json.dumps(teamoutput, indent=4))
        f.close()
        # print(team, week, url, file_key)
        # print(json.dumps(teamoutput, indent=4))
