import requests
import json

espn_api_base_url = "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
depth_chart = []
response = requests.get(espn_api_base_url+"teams?limit=32")
team_links = response.json()['items']
for link in team_links:
    response = requests.get(link['$ref'])
    team = response.json()
    response = requests.get(team['depthCharts']['$ref'])
    chart = response.json()
    team_data = {
        "id": team['id'],
        "displayName": team['displayName'],
        "$ref": team['$ref'],
        "depth": chart
    }
    depth_chart.append(team_data)

f = open('depth_charts.json', "w")
f.write(json.dumps(depth_chart, indent=4))
f.close()
