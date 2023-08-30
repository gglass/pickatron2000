import requests
import json
import hashlib

class SportsDataIOService:

    baseApiUrl = "https://api.sportsdata.io/v3/nfl/scores/json/"
    teamAvgs = []
    key = ""

    def __init__(self, key):
        self.key = key

    def get_or_fetch_from_cache(self, endpoint, directory="caches", overwrite=False):
        url = self.baseApiUrl + endpoint + "?key=" + self.key
        file_key = hashlib.md5(url.encode('UTF-8')).hexdigest()
        if overwrite:
            response = requests.get(url)
            data = response.json()
            f = open(directory + "/" + file_key, "w")
            f.write(json.dumps(data, indent=4))
            f.close()
            return data
        else:
            try:
                f = open(directory+"/"+file_key, "r")
                data = json.load(f)
                f.close()
                return data
            except:
                # couldn't find that file yet, lets fetch the thing and put it there in the file
                response = requests.get(url)
                data = response.json()
                f = open(directory+"/"+file_key, "x")
                f.write(json.dumps(data, indent=4))
                f.close()
                return data

    def get_weekly_inputs(self, season, week):
        yearWeekStats = self.get_or_fetch_from_cache(endpoint="ScoresByWeek/" + str(season) + "REG/" + str(week))
        rows = []
        print("Processing... ")
        for game in yearWeekStats:
            row = {}
            awayTeamId = game['AwayTeamID']
            homeTeamId = game['HomeTeamID']
            homeAvgs = self.get_team_recent_stats(season=season, teamId=homeTeamId)
            awayAvgs = self.get_team_recent_stats(season=season, teamId=awayTeamId)
            for key in homeAvgs.keys():
                row["home"+key] = homeAvgs[key]
            for key in awayAvgs.keys():
                row["away"+key] = awayAvgs[key]
            if game["AwayScore"] and game["HomeScore"]:
                row["actualSpread"] = game["AwayScore"] - game["HomeScore"]
                row["HomeScore"] = game["HomeScore"]
                row["AwayScore"] = game["AwayScore"]
            row["Date"] = game["Date"]
            rows.append(row)
        return rows

    def get_team_recent_stats(self, season, teamId):
        #start by getting their last 5 games. We can up this if we want to.
        recency = 5
        sums = {
            "Team": "",
            "Season": "",
            "Score": 0,
            "CompletionPercentage": 0,
            "ThirdDownPercentage": 0,
            "FieldGoalsMade": 0,
            "FirstDowns": 0,
            "Fumbles": 0,
            "Kickoffs": 0,
            "OffensiveYards": 0,
            "PassingYards": 0,
            "RushingYards": 0,
            "PenaltyYards": 0,
            "Penalties": 0,
            "Sacks": 0,
            "OpponentScore": 0,
            "OpponentPenalties": 0
        }
        recentGames = self.get_or_fetch_from_cache(endpoint="TeamGameStatsBySeason/" + str(season) + "REG/" + str(teamId) + "/" + str(recency))
        for game in recentGames:
            sums["Team"] = game["Team"]
            sums["Season"] = game["Season"]
            sums['Score'] += game['Score']
            sums['CompletionPercentage'] += game["CompletionPercentage"]
            sums['ThirdDownPercentage'] += game["ThirdDownPercentage"]
            sums['FieldGoalsMade'] += game["FieldGoalsMade"]
            sums['FirstDowns'] += game["FirstDowns"]
            sums['Fumbles'] += game["Fumbles"]
            sums['Kickoffs'] += game["Kickoffs"]
            sums['OffensiveYards'] += game["OffensiveYards"]
            sums['PassingYards'] += game["PassingYards"]
            sums['RushingYards'] += game["RushingYards"]
            sums['PenaltyYards'] += game["PenaltyYards"]
            sums['Penalties'] += game["Penalties"]
            sums['Sacks'] += game["Sacks"]
            sums['OpponentScore'] += game["OpponentScore"]
            sums['OpponentPenalties'] += game["OpponentPenalties"]

        teamAvg = {
            "team": sums['Team'],
            "season": sums['Season'],
            "avgScore": sums['Score']/recency,
            "completionPct": sums['CompletionPercentage']/recency,
            "thirdDownPct": sums['ThirdDownPercentage']/recency,
            "avgFieldGoals": sums['FieldGoalsMade']/recency,
            "avgFirstDowns": sums['FirstDowns'] / recency,
            "avgFumbles": sums['Fumbles'] / recency,
            "avgKickoffs": sums['Kickoffs'] / recency,
            "avgPassingYards": sums['PassingYards'] / recency,
            "avgRushingYards": sums['RushingYards'] / recency,
            "avgOffensiveYards": sums['OffensiveYards'] / recency,
            "avgPenaltyYards": sums['PenaltyYards'] / recency,
            "avgPenalties": sums['Penalties'] / recency,
            "avgSacks": sums['Sacks'] / recency,
            "avgOppScore": sums['OpponentScore'] / recency,
            "avgOppPenalties": sums['OpponentPenalties'] / recency,
        }
        return teamAvg

    def get_all_team_avgs_by_year(self):
        for year in range(2003,2023):
            yearStats = self.get_or_fetch_from_cache(endpoint="TeamSeasonStats/"+str(year)+"REG")
            for team in yearStats:
                teamAvg = {
                    "team": team['Team'],
                    "season": team['Season'],
                    "avgScore": team['Score']/team['Games'],
                    "completionPct": team['CompletionPercentage'],
                    "thirdDownPct": team['ThirdDownPercentage'],
                    "avgFieldGoals": team['FieldGoalsMade']/team['Games'],
                    "avgFirstDowns": team['FirstDowns'] / team['Games'],
                    "avgFumbles": team['Fumbles'] / team['Games'],
                    "avgKickoffs": team['Kickoffs'] / team['Games'],
                    "avgPassingYards": team['PassingYards'] / team['Games'],
                    "avgRushingYards": team['RushingYards'] / team['Games'],
                    "avgOffensiveYards": team['OffensiveYards'] / team['Games'],
                    "avgPenaltyYards": team['PenaltyYards'] / team['Games'],
                    "avgPenalties": team['Penalties'] / team['Games'],
                    "avgSacks": team['Sacks'] / team['Games'],
                    "avgOppScore": team['OpponentScore'] / team['Games'],
                    "avgOppPenalties": team['OpponentPenalties'] / team['Games'],
                    "avgTimeOfPoss": team['TimeOfPossession'],
                }
                self.teamAvgs.append(teamAvg)

        try:
            f = open("inputs/teamSeasonAvgs.json", "w")
            f.write(json.dumps(self.teamAvgs, indent=4))
            f.close()
        except:
            f = open("inputs/teamSeasonAvgs.json", "x")
            f.write(json.dumps(self.teamAvgs, indent=4))
            f.close()

    def generate_training_data(self):
        seasons = [
            {
                "season": 2022,
                "weeks": 18
            },
            {
                "season": 2021,
                "weeks": 18
            },
            {
                "season": 2020,
                "weeks": 17
            },
            {
                "season": 2019,
                "weeks": 17
            },
            {
                "season": 2018,
                "weeks": 17
            },
            {
                "season": 2017,
                "weeks": 17
            },
            {
                "season": 2016,
                "weeks": 17
            },
            {
                "season": 2015,
                "weeks": 17
            },
            {
                "season": 2014,
                "weeks": 17
            },
            {
                "season": 2013,
                "weeks": 17
            }
        ]
        rows = []
        for season in seasons:
            for week in range(1,season["weeks"]+1):
                rows = rows + self.get_weekly_inputs(season=season["season"], week=week)

        try:
            f = open("inputs/historicalmatchups.json", "w")
            f.write(json.dumps(rows, indent=4))
            f.close()
        except:
            f = open("inputs/historicalmatchups.json", "x")
            f.write(json.dumps(rows, indent=4))
            f.close()