import requests
import json
import hashlib
from bs4 import BeautifulSoup
from time import sleep
from multiprocessing import Pool
import os
import datetime

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

    teams = [
        "Buffalo Bills",
        "Los Angeles Rams",
        "St. Louis Rams",
        "New Orleans Saints",
        "Atlanta Falcons",
        "Cleveland Browns",
        "Carolina Panthers",
        "Chicago Bears",
        "San Francisco 49ers",
        "Pittsburgh Steelers",
        "Cincinnati Bengals",
        "Houston Texans",
        "Indianapolis Colts",
        "Philadelphia Eagles",
        "Detroit Lions",
        "Washington Commanders",
        "Washington Redskins",
        "Washington Football Team",
        "Jacksonville Jaguars",
        "Miami Dolphins",
        "New England Patriots",
        "Baltimore Ravens",
        "New York Jets",
        "Kansas City Chiefs",
        "Arizona Cardinals",
        "Minnesota Vikings",
        "Green Bay Packers",
        "New York Giants",
        "Tennessee Titans",
        "Los Angeles Chargers",
        "San Diego Chargers",
        "Las Vegas Raiders",
        "Oakland Raiders",
        "Tampa Bay Buccaneers",
        "Dallas Cowboys",
        "Seattle Seahawks",
        "Denver Broncos"
    ]

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
            f = open(directory + "/" + file_key, "w", encoding="utf-8")
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
                print(endpoint)
                sleep(2)
                response = requests.get(url)
                data = response.text
                if data.find("429 error") >= 0:
                    print("429 detected... refetching")
                    print(data)
                    sleep(20)
                    return self.get_or_fetch_from_cache(endpoint)
                f = open(directory+"/"+file_key, "x", encoding="utf-8")
                f.write(data)
                f.close()
                return data

    def get_weekly_results(self, season, week, overwrite=False):
        yearlygames = self.get_or_fetch_from_cache(endpoint="years/" + str(season) + "/games.htm", overwrite=overwrite)
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
                parsed = [season, week]
                index = 2
                for td in row.findAll("td"):
                    if index == 8:
                        a = td.findAll("a")[0]
                        parsed.append(a['href'])
                    else:
                        parsed.append(td.getText())
                    index += 1
                values.append(parsed)
                # values.append([season, rowWeek] + [td.getText() for td in row.findAll("td")])
        yearWeekStats = []

        for game in values:
            formatted = {headers[i]: game[i] for i in range(len(headers))}
            yearWeekStats.append(formatted)

        rows = []
        for game in yearWeekStats:
            row = {}
            if game["Winner/tie"] != "" and game["Loser/tie"] != "":
                try:
                    LPTs = int(game["LPts"])
                    WPTs = int(game["WPts"])
                except:
                    print("Uh oh...")
                    print(game)
                    continue

                boxlink = self.get_or_fetch_from_cache(endpoint=game['boxlink'], overwrite=False)
                for line in boxlink.split("\n"):
                    if "Vegas Line" in line:
                        soup = BeautifulSoup(line, features="html.parser")
                vegas_line = soup.find("td").getText()

                # this signifies the hometeam lost
                if game["at"] == "@":
                    row["homeTeam"] = game["Loser/tie"]
                    row["awayTeam"] = game["Winner/tie"]
                    row["HomeScore"] = LPTs
                    row["AwayScore"] = WPTs
                    row["Winner"] = 0
                else:
                    row["homeTeam"] = game["Winner/tie"]
                    row["awayTeam"] = game["Loser/tie"]
                    row["AwayScore"] = LPTs
                    row["HomeScore"] = WPTs
                    row["Winner"] = 1

                # note on spread notation. Its always AWAY - HOME, so a spread of -3 indicates that the Home team won by 3
                if "AwayScore" in row and "HomeScore" in row:
                    row["actualSpread"] = row["AwayScore"] - row["HomeScore"]
                row["Date"] = game["Date"]
                row["VegasLine"] = vegas_line
                row["week"] = week
                row["season"] = season
                rows.append(row)

        return rows

    def get_weekly_inputs(self, season, week, overwrite=False):
        yearlygames = self.get_or_fetch_from_cache(endpoint="years/" + str(season) + "/games.htm",overwrite=overwrite)
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
            if formatted['at'] == '@':
                #this indicates the winner was a visitor
                formatted['HomeTm'] = formatted['Loser/tie']
                formatted['VisTm'] = formatted['Winner/tie']
            else:
                # this indicates the winner was the home team
                formatted['HomeTm'] = formatted['Winner/tie']
                formatted['VisTm'] = formatted['Loser/tie']
            yearWeekStats.append(formatted)

        return yearWeekStats

    def get_upcoming_inputs(self, season, week, overwrite=True):
        yearlygames = self.get_or_fetch_from_cache(endpoint="years/" + str(season) + "/games.htm")
        soup = BeautifulSoup(yearlygames, features="html.parser")
        table = soup.find(id="games")
        if(week == 1):
            headers = ['Season','Week', 'Day', 'Date', 'VisTm', 'Pts','at', 'HomeTm', 'Pts', 'Time']
        else:
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

        return yearWeekStats

    def get_team_season_data(self, season, teamName):

        if teamName == "":
            print("Empty team name for recent stats!!!")

        headers = ['Season','Week', 'Day', 'Date', 'Time', 'boxlink', 'W/L', 'OT', 'Rec', 'at', 'Opponent', 'Tm', 'Opp', 'O1stD', 'OTotYd', 'OPassY', 'ORushY', 'OTO', 'D1stD', 'DTotYd', 'DPassY', 'DRushY', 'DTO', 'Offense', 'Defense', 'Sp. Tms']
        teamGameStats = []

        #lets start by getting the teams SOS for this given year
        seasonStats = self.get_or_fetch_from_cache(
            endpoint="teams/" + str(self.teamMap[teamName]) + "/" + str(season) + ".htm", overwrite=False)
        try:
            soup = BeautifulSoup(seasonStats, features="html.parser")
            metaTable = soup.find(id="meta")
            summaryStats = metaTable.findAll("div")[1].findAll("p")
            strength = summaryStats[5]
            SRSSOSText = strength.getText()
            split = SRSSOSText.split(":")
            SOS = float(split[2])
        except:
            SOS = 0

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
                        if formatted['Opponent'] == 'Bye Week' or formatted['Opponent'] == "" or formatted['W/L'] == '':
                            continue
                        for key in headers:
                            if formatted[key] == "":
                                formatted[key] = 0
                        formatted['Team'] = teamName
                        formatted['SOS'] = SOS
                        teamGameStats.append(formatted)
                except Exception as error:
                    continue

        if len(teamGameStats) > 0:
            return teamGameStats
        else:
            return False

    def get_historical_data(self, seasons):
        tasks = []
        rows = []
        weeks = []

        #this does it in parallel
        for season in seasons:
            for week in range(1, season["weeks"] + 1):
                tasks.append([season["season"], week])

        with Pool() as p:
            weeks = weeks + p.starmap(self.get_weekly_results, tasks)

        #single thread
        # for season in seasons:
        #     for week in range(1, season["weeks"] + 1):
        #         weeks = weeks + self.get_weekly_results(season["season"], week)
        for idx, weekbatch in enumerate(weeks):
            for game in weekbatch:
                rows.append(game)

        return rows

    def dump_historic_data(self):
        seasons = [
            {
                "season": 2024,
                "weeks": 9
            },
            {
                "season": 2023,
                "weeks": 18
            },
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

        rows = self.get_historical_data(seasons)
        try:
            f = open("data/alldata.json", "w")
            f.write(json.dumps(rows, indent=4))
            f.close()
        except:
            f = open("data/alldata.json", "x")
            f.write(json.dumps(rows, indent=4))
            f.close()

    def generate_training_data(self):
        training_seasons = [
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
                "season": 2023,
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

