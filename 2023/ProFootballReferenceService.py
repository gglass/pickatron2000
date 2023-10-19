import requests
import json
import hashlib
from bs4 import BeautifulSoup
from time import sleep
from multiprocessing import Pool
import os

class ProFootballReferenceService:

    baseApiUrl = "https://www.pro-football-reference.com/"
    teamAvgs = []

    teamMap = {
        "Buffalo Bills": "buf",
        "Los Angeles Rams": "ram",
        "St. Louis Rams": "ram",
        "New Orleans Saints": "nor",
        "Atlanta Falcons": "atl",
        "Cleveland Browns": "cle",
        "Carolina Panthers": "car",
        "Chicago Bears": "chi",
        "San Francisco 49ers": "sfo",
        "Pittsburgh Steelers": "pit",
        "Cincinnati Bengals": "cin",
        "Houston Texans": "htx",
        "Indianapolis Colts": "clt",
        "Philadelphia Eagles": "phi",
        "Detroit Lions": "det",
        "Washington Commanders": "was",
        "Washington Redskins": "was",
        "Washington Football Team": "was",
        "Jacksonville Jaguars": "jax",
        "Miami Dolphins": "mia",
        "New England Patriots": "nwe",
        "Baltimore Ravens": "rav",
        "New York Jets": "nyj",
        "Kansas City Chiefs": "kan",
        "Arizona Cardinals": "crd",
        "Minnesota Vikings": "min",
        "Green Bay Packers": "gnb",
        "New York Giants": "nyg",
        "Tennessee Titans": "oti",
        "Los Angeles Chargers": "sdg",
        "San Diego Chargers": "sdg",
        "Las Vegas Raiders": "rai",
        "Oakland Raiders": "rai",
        "Tampa Bay Buccaneers": "tam",
        "Dallas Cowboys": "dal",
        "Seattle Seahawks": "sea",
        "Denver Broncos": "den",
    }

    # def __init__(self):
    #     print("Initializing scraper")

    def get_or_fetch_from_cache(self, endpoint, directory="caches", overwrite=False):
        url = self.baseApiUrl + endpoint
        file_key = hashlib.md5(url.encode('UTF-8')).hexdigest()
        # print(url, file_key)
        if overwrite:
            print("Overwrite specified. Fetching it new.")
            sleep(2)
            response = requests.get(url)
            data = response.text
            f = open(directory + "/" + file_key, "w")
            f.write(data)
            f.close()
            return data
        else:
            try:
                f = open(directory+"/"+file_key, "r")
                data = f.readlines()
                f.close()
                text = "".join(data)
                # print("Found the requested resource in cache")
                if text.find("429 error") >= 0:
                    print("429 detected... refetching")
                    sleep(2)
                    os.remove(directory+"/"+file_key)
                    return self.get_or_fetch_from_cache(endpoint)
                else:
                    return text
            except:
                # couldn't find that file yet, lets fetch the thing and put it there in the file
                print("Couldn't find resource. Fetching it new.")
                sleep(2)
                response = requests.get(url)
                data = response.text
                if data.find("429 error") >= 0:
                    print("429 detected... refetching")
                    sleep(2)
                    return self.get_or_fetch_from_cache(endpoint)
                f = open(directory+"/"+file_key, "x")
                f.write(data)
                f.close()
                return data

    def get_weekly_results(self, season, week):
        yearlygames = self.get_or_fetch_from_cache(endpoint="years/" + str(season) + "/games.htm")
        soup = BeautifulSoup(yearlygames, features="html.parser")
        table = soup.find(id="games")
        headers = ['Season','Week', 'Day', 'Date', 'Time', 'Winner/tie', 'at', 'Loser/tie', 'boxlink', 'WPts', 'LPts', 'YdsW', 'TOW', 'YdsL', 'TOL']
        values = []
        try:
            tablerows = table.findAll("tbody")[0].findAll("tr")
        except:
            print(soup)
            print(table)
            exit()

        for idx, row in enumerate(tablerows):
            if "class" in row and row["class"] == "thead":
                print("found a thead... skipping")
                continue
            #get the week
            rowWeek = row.findAll("th")[0].getText()
            if str(rowWeek) == str(week):
                values.append([season, rowWeek] + [td.getText() for td in row.findAll("td")])
        yearWeekStats = []

        for game in values:
            formatted = {headers[i]: game[i] for i in range(len(headers))}
            yearWeekStats.append(formatted)

        rows = []
        # print("Processing... ", season, week)
        for game in yearWeekStats:
            row = {}
            if game["Winner/tie"] != "" and game["Loser/tie"] != "":
                #this signifies the hometeam lost
                if game["at"] == "@":
                    row["HomeTeam"] = game["Loser/tie"]
                    row["AwayTeam"] = game["Winner/tie"]
                    row["HomeScore"] = int(game["LPts"])
                    row["AwayScore"] = int(game["WPts"])
                else:
                    row["AwayTeam"] = game["Loser/tie"]
                    row["HomeTeam"] = game["Winner/tie"]
                    row["AwayScore"] = int(game["LPts"])
                    row["HomeScore"] = int(game["WPts"])
                #note on spread notation. Its always AWAY - HOME, so a spread of -3 indicates that the Home team won by 3
                if "AwayScore" in row and "HomeScore" in row:
                    row["actualSpread"] = row["AwayScore"] - row["HomeScore"]
                row["Date"] = game["Date"]
                rows.append(row)
        return rows

    def get_weekly_inputs(self, season, week):
        yearlygames = self.get_or_fetch_from_cache(endpoint="years/" + str(season) + "/games.htm")
        soup = BeautifulSoup(yearlygames, features="html.parser")
        table = soup.find(id="games")
        headers = ['Season','Week', 'Day', 'Date', 'Time', 'Winner/tie', 'at', 'Loser/tie', 'boxlink', 'WPts', 'LPts', 'YdsW', 'TOW', 'YdsL', 'TOL']
        values = []
        try:
            tablerows = table.findAll("tbody")[0].findAll("tr")
        except:
            print(soup)
            print(table)
            exit()

        for idx, row in enumerate(tablerows):
            if "class" in row and row["class"] == "thead":
                print("found a thead... skipping")
                continue
            #get the week
            rowWeek = row.findAll("th")[0].getText()
            if str(rowWeek) == str(week):
                values.append([season, rowWeek] + [td.getText() for td in row.findAll("td")])
        yearWeekStats = []

        for game in values:
            formatted = {headers[i]: game[i] for i in range(len(headers))}
            yearWeekStats.append(formatted)

        rows = []
        # print("Processing... ", season, week)
        for game in yearWeekStats:
            row = {}
            if game["Winner/tie"] != "" and game["Loser/tie"] != "":
                winnerAvgs = self.get_team_recent_stats(season=season, teamName=game["Winner/tie"], week=week)
                loserAvgs = self.get_team_recent_stats(season=season, teamName=game["Loser/tie"], week=week)

                if winnerAvgs != False and loserAvgs != False:
                    #this signifies the hometeam lost
                    if game["at"] == "@":
                        for key in winnerAvgs.keys():
                            row["away" + key] = winnerAvgs[key]
                        for key in loserAvgs.keys():
                            row["home" + key] = loserAvgs[key]
                        row["HomeScore"] = int(game["LPts"])
                        row["AwayScore"] = int(game["WPts"])
                    else:
                        for key in winnerAvgs.keys():
                            row["away" + key] = loserAvgs[key]
                        for key in loserAvgs.keys():
                            row["home" + key] = winnerAvgs[key]
                        row["AwayScore"] = int(game["LPts"])
                        row["HomeScore"] = int(game["WPts"])

                    #note on spread notation. Its always AWAY - HOME, so a spread of -3 indicates that the Home team won by 3
                    if "AwayScore" in row and "HomeScore" in row:
                        row["actualSpread"] = row["AwayScore"] - row["HomeScore"]
                    row["Date"] = game["Date"]
                    rows.append(row)
        return rows

    def get_upcoming_inputs(self, season, week, overwrite=True):
        yearlygames = self.get_or_fetch_from_cache(endpoint="years/" + str(season) + "/games.htm")
        soup = BeautifulSoup(yearlygames, features="html.parser")
        table = soup.find(id="games")
        headers = ['Season','Week', 'Day', 'Date', 'Time', 'VisTm', 'at', 'HomeTm', 'boxlink', 'WPts', 'LPts', 'YdsW', 'TOW', 'YdsL', 'TOL']
        values = []
        try:
            tablerows = table.findAll("tbody")[0].findAll("tr")
        except:
            print(soup)
            print(table)
            exit()
        for idx, row in enumerate(tablerows):
            if "class" in row and row["class"] == "thead":
                print("found a thead... skipping")
                continue
            #get the week
            rowWeek = row.findAll("th")[0].getText()
            if str(rowWeek) == str(week):
                values.append([season, rowWeek] + [td.getText() for td in row.findAll("td")])
        yearWeekStats = []

        for game in values:
            formatted = {headers[i]: game[i] for i in range(len(headers))}
            yearWeekStats.append(formatted)

        rows = []
        # print("Processing... ", season, week)
        for game in yearWeekStats:
            row = {}
            if game["HomeTm"] != "" and game["VisTm"] != "":
                HomeAvgs = self.get_team_recent_stats_future(season=season, teamName=game["HomeTm"], week=week, overwrite=overwrite)
                AwayAvgs = self.get_team_recent_stats_future(season=season, teamName=game["VisTm"], week=week, overwrite=overwrite)

                if HomeAvgs != False and AwayAvgs != False:
                    for key in AwayAvgs.keys():
                        row["away" + key] = AwayAvgs[key]
                    for key in HomeAvgs.keys():
                        row["home" + key] = HomeAvgs[key]

                    rows.append(row)
        return rows

    def get_team_recent_stats_future(self, season, week, teamName, overwrite=True):

        if teamName == "":
            print("Empty team name for recent stats!!!")

        #start by getting their last few games. We can up this if we want to.
        recency = 8
        sums = {
            "Team": "",
            "Score": 0,
            "FirstDowns": 0,
            "TurnoversLost": 0,
            "TurnoversForced": 0,
            "OffensiveYards": 0,
            "PassingYards": 0,
            "RushingYards": 0,
            "DefensiveYardsAllowed": 0,
            "RushingAllowed": 0,
            "PassingAllowed": 0,
            "OpponentScore": 0,
            "Wins": 0,
            "Streak": 0,
        }
        headers = ['Season','Week', 'Day', 'Date', 'Time', 'boxlink', 'W/L', 'OT', 'Rec', 'at', 'Opponent', 'Tm', 'Opp', 'O1stD', 'OTotYd', 'OPassY', 'ORushY', 'OTO', 'D1stD', 'DTotYd', 'DPassY', 'DRushY', 'DTO', 'Offense', 'Defense', 'Sp. Tms']
        teamGameStats = []

        #if we don't have enough data in this year, we need to go fetch the previous year as well
        if week <= recency:
            teamSeasonStats = self.get_or_fetch_from_cache(
                endpoint="teams/" + str(self.teamMap[teamName]) + "/" + str(season - 1) + ".htm")

            soup = BeautifulSoup(teamSeasonStats, features="html.parser")
            gamesTable = soup.find(id="games")
            values = []
            if gamesTable is not None and gamesTable.findAll("tbody")[0] is not None:
                tablerows = gamesTable.findAll("tbody")[0].findAll("tr")
                for idx, row in enumerate(tablerows):
                    if "class" in row and row["class"] == "thead":
                        print("found a thead... skipping")
                        continue
                    # get the week
                    rowWeek = row.findAll("th")[0].getText()
                    values.append([season - 1, rowWeek] + [td.getText() for td in row.findAll("td")])
                for game in values:
                    try:
                        if len(game) == 26 and int(game[1]):
                            formatted = {headers[i]: game[i] for i in range(len(headers))}
                            if formatted['Opponent'] == 'Bye Week' or formatted['Opponent'] == "":
                                continue
                            for key in headers:
                                if formatted[key] == "":
                                    formatted[key] = 0
                            teamGameStats.append(formatted)
                    except Exception as error:
                        continue

        teamSeasonStats = self.get_or_fetch_from_cache(endpoint="teams/" + str(self.teamMap[teamName]) + "/" + str(season) + ".htm", overwrite=overwrite)
        soup = BeautifulSoup(teamSeasonStats, features="html.parser")
        gamesTable = soup.find(id="games")
        values = []
        if gamesTable is not None and gamesTable.findAll("tbody")[0] is not None:
            tablerows = gamesTable.findAll("tbody")[0].findAll("tr")
            for idx, row in enumerate(tablerows):
                # get the week
                rowWeek = row.findAll("th")[0].getText()
                if int(rowWeek) >= int(week):
                    continue
                values.append([season, rowWeek] + [td.getText() for td in row.findAll("td")])
            for game in values:
                try:
                    if len(game) == 26 and int(game[1]):
                        formatted = {headers[i]: game[i] for i in range(len(headers))}
                        if formatted['Opponent'] == 'Bye Week' or formatted['Opponent'] == "":
                            continue
                        for key in headers:
                            if formatted[key] == "":
                                formatted[key] = 0
                        teamGameStats.append(formatted)
                except Exception as error:
                    continue

        recentGames = teamGameStats[-recency:]

        if len(recentGames) > 0:
            for game in recentGames:
                sums["Team"] = teamName
                sums['Score'] += int(game['Tm'])
                sums['FirstDowns'] += int(game["O1stD"])
                sums['TurnoversLost'] += int(game["OTO"])
                sums['TurnoversForced'] += int(game["DTO"])
                sums['OffensiveYards'] += int(game["OTotYd"])
                sums['PassingYards'] += int(game["OPassY"])
                sums['RushingYards'] += int(game["ORushY"])
                sums['DefensiveYardsAllowed'] += int(game["DTotYd"])
                sums['PassingAllowed'] += int(game["DPassY"])
                sums['RushingAllowed'] += int(game["DRushY"])
                sums['OpponentScore'] += int(game["Opp"])
                if int(game['Tm']) > int(game["Opp"]):
                    sums['Wins'] += 1
                    sums['Streak'] += 1
                else:
                    sums['Wins'] -= 1
                    sums['Streak'] = 0

            teamAvg = {
                "team": sums['Team'],
                "avgScore": sums['Score']/recency,
                "avgFirstDowns": sums['FirstDowns'] / recency,
                "avgTurnoversLost": sums['TurnoversLost'] / recency,
                "avgTurnoversForced": sums['TurnoversForced'] / recency,
                "avgPassingYards": sums['PassingYards'] / recency,
                "avgRushingYards": sums['RushingYards'] / recency,
                "avgOffensiveYards": sums['OffensiveYards'] / recency,
                "avgPassingYardsAllowed": sums['PassingAllowed'] / recency,
                "avgRushingYardsAllowed": sums['RushingAllowed'] / recency,
                "avgYardsAllowed": sums['DefensiveYardsAllowed'] / recency,
                "avgOppScore": sums['OpponentScore'] / recency,
                "Wins": sums['Wins'] / recency,
                "Streak": sums['Streak'],
            }
            return teamAvg

        else:
            return False

    def get_team_recent_stats(self, season, week, teamName):

        if teamName == "":
            print("Empty team name for recent stats!!!")

        #start by getting their last 10 games. We can up this if we want to.
        recency = 8
        sums = {
            "Team": "",
            "Score": 0,
            "FirstDowns": 0,
            "TurnoversLost": 0,
            "TurnoversForced": 0,
            "OffensiveYards": 0,
            "PassingYards": 0,
            "RushingYards": 0,
            "DefensiveYardsAllowed": 0,
            "RushingAllowed": 0,
            "PassingAllowed": 0,
            "OpponentScore": 0,
            "Wins": 0,
            "Streak": 0,
        }
        headers = ['Season','Week', 'Day', 'Date', 'Time', 'boxlink', 'W/L', 'OT', 'Rec', 'at', 'Opponent', 'Tm', 'Opp', 'O1stD', 'OTotYd', 'OPassY', 'ORushY', 'OTO', 'D1stD', 'DTotYd', 'DPassY', 'DRushY', 'DTO', 'Offense', 'Defense', 'Sp. Tms']
        teamGameStats = []

        #if we don't have enough data in this year, we need to go fetch the previous year as well
        if week <= recency:
            teamSeasonStats = self.get_or_fetch_from_cache(
                endpoint="teams/" + str(self.teamMap[teamName]) + "/" + str(season - 1) + ".htm")
            soup = BeautifulSoup(teamSeasonStats, features="html.parser")
            gamesTable = soup.find(id="games")
            gamesRows = []
            values = []
            if gamesTable is not None and gamesTable.findAll("tbody")[0] is not None:
                tablerows = gamesTable.findAll("tbody")[0].findAll("tr")
                for idx, row in enumerate(tablerows):
                    if "class" in row and row["class"] == "thead":
                        print("found a thead... skipping")
                        continue
                    # get the week
                    rowWeek = row.findAll("th")[0].getText()
                    values.append([season - 1, rowWeek] + [td.getText() for td in row.findAll("td")])
                for game in values:
                    try:
                        if len(game) == 26 and int(game[1]):
                            formatted = {headers[i]: game[i] for i in range(len(headers))}
                            if formatted['Opponent'] == 'Bye Week' or formatted['Opponent'] == "":
                                continue
                            for key in headers:
                                if formatted[key] == "":
                                    formatted[key] = 0
                            teamGameStats.append(formatted)
                    except Exception as error:
                        continue

        teamSeasonStats = self.get_or_fetch_from_cache(endpoint="teams/" + str(self.teamMap[teamName]) + "/" + str(season) + ".htm")
        soup = BeautifulSoup(teamSeasonStats, features="html.parser")
        gamesTable = soup.find(id="games")
        values = []
        if gamesTable is not None and gamesTable.findAll("tbody")[0] is not None:
            tablerows = gamesTable.findAll("tbody")[0].findAll("tr")
            for idx, row in enumerate(tablerows):
                # get the week
                rowWeek = row.findAll("th")[0].getText()
                values.append([season, rowWeek] + [td.getText() for td in row.findAll("td")])
            for game in values:
                try:
                    if len(game) == 26 and int(game[1]):
                        formatted = {headers[i]: game[i] for i in range(len(headers))}
                        if formatted['Opponent'] == 'Bye Week' or formatted['Opponent'] == "":
                            continue
                        for key in headers:
                            if formatted[key] == "":
                                formatted[key] = 0
                        teamGameStats.append(formatted)
                except Exception as error:
                    continue

        recentGames = []
        for idx, game in enumerate(teamGameStats):
            #progress until we find the season/week we are currently looking for
            if str(game["Season"]) == str(season) and str(game["Week"]) == str(week):
                for cursor in range(1, recency + 1):
                    recentGames.append(teamGameStats[idx - cursor])

        if len(recentGames) > 0:
            for game in recentGames:
                sums["Team"] = teamName
                sums['Score'] += int(game['Tm'])
                sums['FirstDowns'] += int(game["O1stD"])
                sums['TurnoversLost'] += int(game["OTO"])
                sums['OffensiveYards'] += int(game["OTotYd"])
                sums['PassingYards'] += int(game["OPassY"])
                sums['RushingYards'] += int(game["ORushY"])
                sums['DefensiveYardsAllowed'] += int(game["DTotYd"])
                sums['TurnoversForced'] += int(game["DTO"])
                sums['PassingAllowed'] += int(game["DPassY"])
                sums['RushingAllowed'] += int(game["DRushY"])
                sums['OpponentScore'] += int(game["Opp"])
                if int(game['Tm']) > int(game["Opp"]):
                    sums['Wins'] += 1
                    sums['Streak'] += 1
                else:
                    sums['Wins'] -= 1
                    sums['Streak'] = 0

            teamAvg = {
                "team": sums['Team'],
                "avgScore": sums['Score']/recency,
                "avgFirstDowns": sums['FirstDowns'] / recency,
                "avgTurnoversLost": sums['TurnoversLost'] / recency,
                "avgPassingYards": sums['PassingYards'] / recency,
                "avgRushingYards": sums['RushingYards'] / recency,
                "avgOffensiveYards": sums['OffensiveYards'] / recency,
                "avgPassingYardsAllowed": sums['PassingAllowed'] / recency,
                "avgRushingYardsAllowed": sums['RushingAllowed'] / recency,
                "avgTurnoversForced": sums['TurnoversForced'] / recency,
                "avgYardsAllowed": sums['DefensiveYardsAllowed'] / recency,
                "avgOppScore": sums['OpponentScore'] / recency,
                "Wins": sums['Wins'] / recency,
                "Streak": sums['Streak'],
            }
            return teamAvg

        else:
            return False

    def get_historical_data(self, seasons):
        tasks = []
        rows = []
        weeks = []
        for season in seasons:
            for week in range(1, season["weeks"] + 1):
                tasks.append([season["season"], week])

        with Pool() as p:
            weeks = weeks + p.starmap(self.get_weekly_inputs, tasks)

        for weekbatch in weeks:
            for game in weekbatch:
                rows.append(game)

        return rows


    def generate_training_data(self):
        training_seasons = [
            # {
            #     "season": 2022,
            #     "weeks": 18
            # },
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
            },
            {
                "season": 2012,
                "weeks": 17
            },
            {
                "season": 2011,
                "weeks": 17
            },
            {
                "season": 2010,
                "weeks": 17
            },
            {
                "season": 2009,
                "weeks": 17
            },
            {
                "season": 2008,
                "weeks": 17
            }
        ]
        test_seasons = [
            {
                "season": 2022,
                "weeks": 18
            }
        ]

        rows = self.get_historical_data(training_seasons)
        try:
            f = open("inputs/trainingdata.json", "w")
            f.write(json.dumps(rows, indent=4))
            f.close()
        except:
            f = open("inputs/trainingdata.json", "x")
            f.write(json.dumps(rows, indent=4))
            f.close()

        rows = self.get_historical_data(test_seasons)
        try:
            f = open("inputs/testdata.json", "w")
            f.write(json.dumps(rows, indent=4))
            f.close()
        except:
            f = open("inputs/testdata.json", "x")
            f.write(json.dumps(rows, indent=4))
            f.close()

