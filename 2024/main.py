import json
from ProFootballReferenceService import ProFootballReferenceService
import numpy as np
import pandas as pd
from tensorflow import keras
import keras_tuner
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import datetime

import warnings
warnings.filterwarnings("ignore")

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  port=49153,
  user="root",
  password="mysqlpw",
  database="pickatron"
)

def build_and_compile_model():
    dnnmodel = keras.Sequential([keras.layers.Input(shape=(29,))])
    dnnmodel.add(keras.layers.Dense(14, activation='relu'))
    dnnmodel.add(keras.layers.Dense(1))

    dnnmodel.compile(loss='mean_squared_error',
                optimizer=keras.optimizers.Adam(0.001),
                metrics=[keras.metrics.MeanAbsoluteError()]
    )
    return dnnmodel

def build_and_compile_model_hp(hp):
    dnnmodel = keras.Sequential()
    dnnmodel.add(keras.layers.Input(shape=(31,)))
    dnnmodel.add(keras.layers.BatchNormalization())
    for i in range(hp.Int("num_layers", 1, 20)):
        dnnmodel.add(
            keras.layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=15, max_value=550, step=29),
                activation="relu",
            )
        )

    if hp.Boolean("dropout"):
        dnnmodel.add(keras.layers.Dropout(rate=0.25))

    dnnmodel.add(keras.layers.Dense(1))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    dnnmodel.compile(loss='mean_absolute_error',
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=[keras.metrics.MeanSquaredError()]
    )
    return dnnmodel

def build_and_compile_classification_model_hp(hp):
    dnnmodel = keras.Sequential()
    dnnmodel.add(keras.layers.Input(shape=(31,)))
    dnnmodel.add(keras.layers.BatchNormalization())
    for i in range(hp.Int("num_layers", 1, 20)):
        dnnmodel.add(
            keras.layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=15, max_value=550, step=29),
                activation="relu",
            )
        )

    if hp.Boolean("dropout"):
        dnnmodel.add(keras.layers.Dropout(rate=0.25))

    dnnmodel.add(keras.layers.Dense(units=1, activation="sigmoid"))

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    dnnmodel.compile(loss='binary_crossentropy',
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['binary_accuracy']
    )
    return dnnmodel

def generate_training_data():
    service = ProFootballReferenceService()
    service.generate_training_data()

def generate_training_data_from_db(starting_season, ending_season):
    mycursor = mydb.cursor()
    sql = "SELECT * FROM game_log WHERE season >= %s AND season <= %s"
    val = (starting_season, ending_season)
    mycursor.execute(sql, val)
    columns = mycursor.description
    games = [{columns[index][0]: column for index, column in enumerate(value)} for value in mycursor.fetchall()]

    jsondata = json.dumps(games, default=str)
    converteddata = json.loads(jsondata)

    return converteddata

def calculate_previous_opp_record(games, numMatchups=4):
    # now that we have the base data, lets go calculate some historic stuff
    mycursor = mydb.cursor()
    output = []
    for game in games:
        # lets go get (up to) the last 4 matchups of these two teams to see how they did
        sql = "SELECT ROUND(AVG(IF(`W/L` = 'W', 1, 0)),2) as HomeWinPct FROM season_data_by_team WHERE Team = %s AND Opponent = %s ORDER BY Season desc, Week desc LIMIT %s"
        val = (game["homeTeam"], game["awayTeam"], numMatchups)
        mycursor.execute(sql, val)
        columns = mycursor.description

        pastMatchups = [{columns[index][0]: column for index, column in enumerate(value)} for value in
                        mycursor.fetchall()]

        if pastMatchups[0]:
            avgs = pastMatchups[0]
            game["awayRecordAgainstOpp"] = 1 - avgs['HomeWinPct']
            game["homeRecordAgainstOpp"] = avgs['HomeWinPct']
        else:
            game["homeRecordAgainstOpp"] = 0
            game["awayRecordAgainstOpp"] = 0
            continue
        output.append(game)
    return output

def preprocess_data(outlierspread=10, outputParam='Winner'):
    data = generate_training_data_from_db(2008, 2022)
    # print("training data set size =", len(data))
    raw = pd.DataFrame(data)
    dataset = raw.copy()
    dataset = dataset.dropna()
    # clean up games where the spread was wildly out of the norm
    dataset['outlier'] = np.abs(dataset['actualSpread']) > outlierspread
    dataset = dataset[dataset['outlier'] == False]

    if outputParam == "Winner":
        train_dataset = dataset.drop(
            columns=["AwayScore", "HomeScore", "homeTeam", "awayTeam", "homeTeamShort", "awayTeamShort", "Date",
                     'outlier', 'VegasLine', 'actualSpread', 'season'])
    if outputParam == "actualSpread":
        train_dataset = dataset.drop(
            columns=["AwayScore", "HomeScore", "homeTeam", "awayTeam", "homeTeamShort", "awayTeamShort", "Date",
                     'outlier', 'VegasLine', 'Winner', 'season'])
    # print(train_dataset.describe().transpose()[['mean', 'std']])

    data = generate_training_data_from_db(2023, 2023)
    # print("test data set size =", len(data))
    raw = pd.DataFrame(data)
    dataset = raw.copy()
    dataset = dataset.dropna()
    # clean up games where the spread was wildly out of the norm
    dataset['outlier'] = np.abs(dataset['actualSpread']) > outlierspread
    dataset = dataset[dataset['outlier'] == False]

    if outputParam == "Winner":
        test_dataset = dataset.drop(
            columns=["AwayScore", "HomeScore", "homeTeam", "awayTeam", "homeTeamShort", "awayTeamShort", "Date",
                     'outlier', 'VegasLine', 'actualSpread', 'season'])
    if outputParam == "actualSpread":
        test_dataset = dataset.drop(
            columns=["AwayScore", "HomeScore", "homeTeam", "awayTeam", "homeTeamShort", "awayTeamShort", "Date",
                     'outlier', 'VegasLine', 'Winner', 'season'])
    return train_dataset, test_dataset

def train_and_evaluate_model(outlierspread=10, outputParam='Winner', type="Classification", modelLabel='trained.keras'):
    train_dataset, test_dataset, = preprocess_data(outlierspread, outputParam)

    train_inputs = train_dataset.copy()
    eval_inputs = test_dataset.copy()
    train_outputs = train_inputs.pop(outputParam)
    eval_outputs = eval_inputs.pop(outputParam)

    train_inputs = train_inputs.to_numpy()
    eval_inputs = eval_inputs.to_numpy()
    train_outputs = train_outputs.to_numpy()
    eval_outputs = eval_outputs.to_numpy()

    label = modelLabel

    # dnn = build_and_compile_model()
    # history = dnn.fit(train_inputs,
    #           train_outputs,
    #           validation_split=0.15,
    #           verbose=1,
    #           epochs=250,
    #           batch_size=60,
    #           shuffle=True
    #       )

    if type == 'Classification':
        tuner = keras_tuner.RandomSearch(
            hypermodel=build_and_compile_classification_model_hp,
            objective=keras_tuner.Objective("val_binary_accuracy", direction="max"),
            max_trials=15,
            overwrite=True,
            executions_per_trial=5,
            directory="search_results",
            project_name="classification",
        )

    if type == 'Regression':
        tuner = keras_tuner.RandomSearch(
            hypermodel=build_and_compile_model_hp,
            objective=keras_tuner.Objective("val_loss", direction="min"),
            max_trials=15,
            overwrite=True,
            executions_per_trial=5,
            directory="search_results",
            project_name="regression",
        )

    # print("Tuner search space:")
    # print(tuner.search_space_summary())

    tuner.search(train_inputs,
        train_outputs,
        validation_split=0.15,
        verbose=1,
        epochs=100,
        batch_size=64,
        shuffle=True
     )

    # print("Tuner results:")
    # print(tuner.results_summary())

    dnn = tuner.get_best_models(1)[0]
    print("Best model:")
    print(dnn.summary())

    print(dnn.evaluate(eval_inputs, eval_outputs, verbose=0))
    test_predictions = dnn.predict(eval_inputs).flatten()

    thisr2 = r2_score(eval_outputs, test_predictions)
    print(thisr2)

    print("saving model")
    dnn.save(label)
    a = plt.axes(aspect='equal')
    plt.scatter(eval_outputs, test_predictions)
    plt.xlabel('True Values '+ outputParam)
    plt.ylabel('Predictions '+ outputParam)
    lims = [-2, 2]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.show()

def load_and_predict_from_training(games, model='trained.keras', outputParams='Winner'):
    dnn_model = keras.models.load_model(model)
    raw = pd.DataFrame(games)
    dataset = raw.copy()
    dataset = dataset.dropna()
    dataset = dataset.drop(columns=["homeTeam", "awayTeam", "homeTeamShort", "awayTeamShort", "AwayScore", "HomeScore", "actualSpread", "Date", "VegasLine", 'Winner', 'season'])

    predictions = dnn_model.predict(dataset)
    predicted_results = []
    for idx, game in enumerate(games):
        predicted_results.append({
            "awayTeam": game['awayTeam'],
            "homeTeam": game['homeTeam'],
            outputParams: str(predictions[idx][0]),
        })
        # print(game['awayteam'], "@", game['hometeam'], predictions[idx][0])
    return predicted_results

def load_and_predict(games, model='trained.keras', modeltype='Classification'):
    dnn_model = keras.models.load_model(model)
    raw = pd.DataFrame(games)
    dataset = raw.copy()
    dataset = dataset.dropna()
    dataset = dataset.drop(columns=["homeTeam", "awayTeam","homeTeamShort", "awayTeamShort", "Date"])

    freshPredictions = dnn_model.predict(dataset)
    predicted_results = []
    for idx, game in enumerate(games):
        prediction = ''
        if modeltype == 'Classification':
            if freshPredictions[idx][0] > 0.5:
                prediction = game['homeTeam']
            else:
                prediction = game['awayTeam']
        if modeltype == 'Regression':
            prediction = freshPredictions[idx][0]

        predicted_results.append({
            "awayTeam": game['awayTeam'],
            "homeTeam": game['homeTeam'],
            "prediction": str(prediction)
        })
    return predicted_results

def get_weekly_games(season, week, overwrite):
    service = ProFootballReferenceService()
    return service.get_upcoming_inputs(season, week, overwrite=overwrite)

def get_past_weekly_games(season, week, overwrite):
    service = ProFootballReferenceService()
    return service.get_weekly_inputs(season, week, overwrite=overwrite)

def get_weekly_results(season, week, overwrite):
    service = ProFootballReferenceService()
    return service.get_weekly_results(season, week, overwrite=overwrite)

def evaluate_past_week(season, week, model, overwrite=False):
    # print("Evaluation model "+model)
    f = open("week" + str(week) + "predictions.json", "r")
    allpredictions = json.load(f)
    f.close()
    predictions = allpredictions[model]
    results = get_weekly_results(season, week, overwrite=overwrite)

    #not gonna do this anymore because it has janky data in it
    # if overwrite == True:
    #     insert_games_into_db(results)

    spreadDiff = 0
    correctPicks = []
    correctPickNum = 0
    totalmoney = 0
    for id, prediction in enumerate(predictions):
        result = {}
        for game in results:
            if game['homeTeam'] == prediction['homeTeam'] and game['awayTeam'] == prediction['awayTeam']:
                result = game

        try:
            predictedspread = float(prediction['prediction'])
            actualspread = result['actualSpread']
            vegas = result["VegasLine"]
            diff = actualspread - predictedspread
            spreadDiff += abs(diff)
            correctPick = 'false'
            # try and figure out how we did against vegas
            money = -110
            vegas_parts = vegas.split(" -")
            vegas_fav = vegas_parts[0]
            vegas_spread = float(vegas_parts[1])
            # vegas thinks the hometeam is the favorite
            if vegas_fav == prediction['homeTeam']:
                if actualspread == -vegas_spread:
                    money = 0
                else:
                    if predictedspread < -vegas_spread and actualspread < -vegas_spread:
                        money = 100
                    if predictedspread > -vegas_spread and actualspread > -vegas_spread:
                        money = 100

            elif vegas_fav == prediction['awayTeam']:
                if actualspread == vegas_spread:
                    money = 0
                else:
                    if predictedspread > vegas_spread and actualspread > vegas_spread:
                        money = 100
                    if predictedspread < vegas_spread and actualspread < vegas_spread:
                        money = 100
            else:
                print("We couldn't find a match on teams... oh noes.")
                print(vegas_fav, prediction['awayTeam'], prediction['homeTeam'])

            # the away team was predicted to win
            if predictedspread > 0 and actualspread > 0:
                correctPick = 'true'
                correctPickNum += 1
            # the home team was predicted to win and did
            if predictedspread < 0 and actualspread < 0:
                correctPick = 'true'
                correctPickNum += 1
            totalmoney += money
            correctPicks.append(correctPick)
        except:
            money = 0
            spreadDiff += 0
            predictedwinner = prediction['prediction']
            actualspread = result['actualSpread']
            correctPick = 'false'
            # the away team was predicted to win
            if predictedwinner == prediction['homeTeam'] and actualspread < 0:
                correctPick = 'true'
                correctPickNum += 1
            # the home team was predicted to win and did
            if predictedwinner == prediction['awayTeam'] and actualspread > 0:
                correctPick = 'true'
                correctPickNum += 1
            totalmoney += money
            correctPicks.append(correctPick)

    return {"spreadDiff": spreadDiff, "correctPickNum": correctPickNum, "totalmoney": totalmoney}

def evaluate_full_season(games, predictions):
    # print("Evaluation model "+model)
    spreadDiff = 0
    correctPickNum = 0
    vegasCorrectPickNum = 0
    totalmoney = 0
    upsidedowntotal = 0
    hometeamWins = 0
    awayteamWins = 0
    vegasSpreadDiff = 0

    for id, prediction in enumerate(predictions):
        result = games[id]
        predictedspread = float(prediction['actualSpread'])
        actualspread = result['actualSpread']
        diff = actualspread - predictedspread
        spreadDiff += abs(diff)
        vegas = result["VegasLine"]
        if actualspread < 0:
            hometeamWins += 1
        elif actualspread > 0:
            awayteamWins += 1

        # the away team was predicted to win
        if predictedspread > 0 and actualspread > 0:
            correctPickNum += 1
        # the home team was predicted to win and did
        if predictedspread < 0 and actualspread < 0:
            correctPickNum += 1

        money = -110
        vegas_parts = vegas.split(" -")
        vegas_fav = vegas_parts[0]
        vegas_spread = float(vegas_parts[1])
        upsidedown = 0

        if vegas_fav == prediction['homeTeam']:
            vegas_spread = -vegas_spread
            if actualspread == vegas_spread:
                money = 0
                upsidedown = 0
            else:
                if predictedspread < vegas_spread and actualspread < vegas_spread:
                    money = 100
                if predictedspread > vegas_spread and actualspread > vegas_spread:
                    money = 100
                #if pickatron predicted an upset
                if predictedspread > 0:
                    if actualspread > vegas_spread:
                        upsidedown = 100
                    if actualspread <= vegas_spread:
                        upsidedown = -110

        elif vegas_fav == prediction['awayTeam']:
            if actualspread == vegas_spread:
                money = 0
            else:
                if predictedspread > vegas_spread and actualspread > vegas_spread:
                    money = 100
                if predictedspread < vegas_spread and actualspread < vegas_spread:
                    money = 100
                # if pickatron predicted an upset
                if predictedspread < 0:
                    if actualspread < vegas_spread:
                        upsidedown = 100
                    if actualspread >= vegas_spread:
                        upsidedown = -110
        else:
            print("We couldn't find a match on teams... oh noes.")
            print(vegas_fav, prediction['awayTeam'], prediction['homeTeam'])

        vegasdiff = actualspread - vegas_spread
        vegasSpreadDiff += abs(vegasdiff)

        if vegas_spread > 0 and actualspread > 0:
            vegasCorrectPickNum += 1
        if vegas_spread < 0 and actualspread < 0:
            vegasCorrectPickNum += 1

        print("Predicted Spread:", predictedspread, "Actual Spread", actualspread, "Vegas Spread", vegas_spread, "Bet Result", money)
        totalmoney += money
        upsidedowntotal += upsidedown

    return {"spreadDiff": spreadDiff, "avgSpreadDiff": spreadDiff/len(games)," vegasSpreadDiff": vegasSpreadDiff, "vegasAvgSpreadDiff": vegasSpreadDiff/len(games),"correctPickNum": correctPickNum, "vegasCorrectPickNum": vegasCorrectPickNum, "homeWins": hometeamWins, "awayWins": awayteamWins, "totalgames": len(games), "totalMoney": totalmoney, "upsidedownMoney": upsidedowntotal}

def evaluate_full_season_classification(games, predictions):
    # print("Evaluation model "+model)
    correctPickNum = 0
    vegasCorrectPickNum = 0
    hometeamWins = 0
    awayteamWins = 0

    for id, prediction in enumerate(predictions):
        result = games[id]
        if float(prediction["Winner"]) < 0.5:
            predictedwinner = 0
        else:
            predictedwinner = 1

        actualwinner = result['Winner']
        actualspread = result['actualSpread']

        vegas = result["VegasLine"]
        vegas_parts = vegas.split(" -")
        vegas_spread = float(vegas_parts[1])
        vegas_fav = vegas_parts[0]

        if actualwinner == 1:
            hometeamWins += 1
        elif actualwinner == 0:
            awayteamWins += 1

        # the away team was predicted to win
        if actualwinner == predictedwinner:
            correctPickNum += 1

        if vegas_fav == prediction['homeTeam']:
            vegas_spread = -vegas_spread

        if vegas_spread > 0 and actualspread > 0:
            vegasCorrectPickNum += 1
        if vegas_spread < 0 and actualspread < 0:
            vegasCorrectPickNum += 1

        print("Predicted Winner:", predictedwinner, "Actual Winner", actualwinner)

    return {"correctPickNum": correctPickNum, "vegasCorrectPickNum": vegasCorrectPickNum, "homeWins": hometeamWins, "awayWins": awayteamWins, "totalgames": len(games)}

def get_team_recent_stats_from_db(season, week, team, recency = 8):
    # now that we have the base data, lets go calculate some historic stuff

    #go get the target date for the week we are currently processing
    mycursor = mydb.cursor()
    sql = """
    SELECT
    isoDate
    FROM season_data_by_team
    WHERE Season = %s
    AND Week = %s
    ORDER BY isoDate DESC
    LIMIT 1;"""
    val = (season, week)

    mycursor.execute(sql, val)
    columns = mycursor.description
    result = [{columns[index][0]: column for index, column in enumerate(value)} for value in mycursor.fetchall()]

    if not result:
        # this means that we may be processing the current week, for which we cannot go get the targetDate so we can probably just use today's date for the targetDate
        targetDate = datetime.date.today().isoformat()
    else:
        # this means we are currently processing some week that already exists in the database, probably for historical reasons
        targetDate = result[0]['isoDate']

    mycursor = mydb.cursor()
    sql = """
    SELECT
    Team,
    ROUND(AVG(Tm), 2) as avgScore,
    ROUND(AVG(O1stD), 2) as avgFirstDowns,
    ROUND(AVG(OTO), 2) as avgTurnoversLost,
    ROUND(AVG(OPassY), 2) as avgPassingYards,
    ROUND(AVG(ORushY), 2) as avgRushingYards,
    ROUND(AVG(OTotYd), 2) as avgOffensiveYards,
    ROUND(AVG(DPassY), 2) as avgPassingYardsAllowed,
    ROUND(AVG(DRushY), 2) as avgRushingYardsAllowed,
    ROUND(AVG(DTO), 2) as avgTurnoversForced,
    ROUND(AVG(DTotYd), 2) as avgYardsAllowed,
    ROUND(AVG(Opp), 2) as avgOppScore,
    ROUND(AVG(IF(`W/L` = 'W', 1, 0)), 2) as Wins,
    ROUND(AVG(SOS), 2) as SOS
    FROM (
        SELECT * FROM season_data_by_team
        WHERE Team = %s
        AND isoDate < %s
        ORDER BY isoDate DESC
    ) as recent
    GROUP BY Team
    LIMIT %s;"""
    val = (team, targetDate, recency)

    mycursor.execute(sql, val)
    columns = mycursor.description
    result = [{columns[index][0]: column for index, column in enumerate(value)} for value in mycursor.fetchall()]
    if not result:
        #we couldn't find any games before this, so lets set all the avgs to just the most recent game we know abouut
        mycursor = mydb.cursor()
        sql = """
            SELECT
            Team,
            ROUND(AVG(Tm), 2) as avgScore,
            ROUND(AVG(O1stD), 2) as avgFirstDowns,
            ROUND(AVG(OTO), 2) as avgTurnoversLost,
            ROUND(AVG(OPassY), 2) as avgPassingYards,
            ROUND(AVG(ORushY), 2) as avgRushingYards,
            ROUND(AVG(OTotYd), 2) as avgOffensiveYards,
            ROUND(AVG(DPassY), 2) as avgPassingYardsAllowed,
            ROUND(AVG(DRushY), 2) as avgRushingYardsAllowed,
            ROUND(AVG(DTO), 2) as avgTurnoversForced,
            ROUND(AVG(DTotYd), 2) as avgYardsAllowed,
            ROUND(AVG(Opp), 2) as avgOppScore,
            ROUND(AVG(IF(`W/L` = 'W', 1, 0)), 2) as Wins,
            ROUND(AVG(SOS), 2) as SOS
            FROM (
                SELECT * FROM season_data_by_team
                WHERE Team = %s
                AND isoDate <= %s
                ORDER BY isoDate DESC
            ) as recent
            GROUP BY Team
            LIMIT %s;"""
        val = (team, targetDate, recency)

        mycursor.execute(sql, val)
        columns = mycursor.description
        result = [{columns[index][0]: column for index, column in enumerate(value)} for value in mycursor.fetchall()]
        if not result:
            # something is wrong/weird here, so lets just return some zeros
            return {
                'Team': team,
                'avgScore': 0,
                'avgFirstDowns': 0,
                'avgTurnoversLost': 0,
                'avgPassingYards': 0,
                'avgRushingYards': 0,
                'avgOffensiveYards': 0,
                'avgPassingYardsAllowed': 0,
                'avgRushingYardsAllowed': 0,
                'avgTurnoversForced': 0,
                'avgYardsAllowed': 0,
                'avgOppScore': 0,
                'Wins': 0,
                'SOS': 0,
                'Streak': 0
            }
        else:
            avgs = result[0]
    else:
        avgs = result[0]

    mycursor = mydb.cursor()
    sql = """
    SELECT `Week`,
    `W/L`
    FROM season_data_by_team 
    WHERE  Team = %s 
    AND isoDate < %s 
    ORDER BY `Week`"""

    val = (team, targetDate)
    mycursor.execute(sql, val)
    columns = mycursor.description
    fullseason = [{columns[index][0]: column for index, column in enumerate(value)} for value in mycursor.fetchall()]

    streak = 0
    for game in fullseason:
        if game['W/L'] == 'W':
            streak += 1
        else:
            streak = 0

    avgs['Streak'] = streak
    return avgs

def get_team_recent_stats(season, week, teamName):

    if teamName == "":
        print("Empty team name for recent stats!!!")
        return False
    else:
        teamAvg = get_team_recent_stats_from_db(season, week, teamName, 8)
        return teamAvg

def insert_past_week_into_game_log(season, week):
    #we should have already updated this by this point so we don't need to overwrite the cache
    games = get_weekly_results(season, week - 1, False)
    # now that we have the games, we need to go calculate the averages from the data we are now storing in the database
    rows = []
    # print("Processing... ", season, week)
    for game in games:
        row = {}
        if game["homeTeam"] != "" and game["awayTeam"] != "":

            HomeAvgs = get_team_recent_stats(season=season, teamName=game["homeTeam"], week=week - 1)
            AwayAvgs = get_team_recent_stats(season=season, teamName=game["awayTeam"], week=week - 1)

            if HomeAvgs != False and AwayAvgs != False:
                for key in AwayAvgs.keys():
                    row["away" + key] = AwayAvgs[key]
                for key in HomeAvgs.keys():
                    row["home" + key] = HomeAvgs[key]
                row['week'] = int(week - 1)
                row['season'] = int(season)
                # date_format = "%B %d %Y"
                # row['Date'] = datetime.datetime.strptime(game['Date']+" 2024", date_format).strftime("%Y-%m-%d")
                row['Date'] = game['Date']
                row["awayTeamShort"] = ProFootballReferenceService.teamMap[game["awayTeam"]]
                row["homeTeamShort"] = ProFootballReferenceService.teamMap[game["homeTeam"]]
                row["AwayScore"] = game["AwayScore"]
                row["HomeScore"] = game["HomeScore"]
                row["Winner"] = game["Winner"]
                row["actualSpread"] = game["actualSpread"]
                row["VegasLine"] = game["VegasLine"]
                rows.append(row)

    games = calculate_previous_opp_record(rows, 4)
    # write it and reload it so it converts Decimal types to floats
    convertedjson = json.dumps(games, indent=4, default=float)
    newdataforinserting = json.loads(convertedjson)
    insert_games_into_db(newdataforinserting)

def predict_past_week(season, week, model, overwrite=False, modeltype='Classification'):
    games = get_past_weekly_games(season, week, overwrite)
    #now that we have the games, we need to go calculate the averages from the data we are now storing in the database
    rows = []
    for game in games:
        row = {}
        if game["HomeTm"] != "" and game["VisTm"] != "":

            HomeAvgs = get_team_recent_stats(season=season, teamName=game["HomeTm"], week=week)
            AwayAvgs = get_team_recent_stats(season=season, teamName=game["VisTm"], week=week)

            if HomeAvgs != False and AwayAvgs != False:
                for key in AwayAvgs.keys():
                    row["away" + key] = AwayAvgs[key]
                for key in HomeAvgs.keys():
                    row["home" + key] = HomeAvgs[key]
                row['week'] = int(week)
                # date_format = "%B %d %Y"
                # row['Date'] = datetime.datetime.strptime(game['Date']+" 2024", date_format).strftime("%Y-%m-%d")
                row['Date'] = game['Date']
                row["awayTeamShort"] = ProFootballReferenceService.teamMap[game["VisTm"]]
                row["homeTeamShort"] = ProFootballReferenceService.teamMap[game["HomeTm"]]
                rows.append(row)

    games = calculate_previous_opp_record(rows, 4)

    #write it and reload it so it converts Decimal types to floats
    convertedjson =  json.dumps(games, indent=4, default=float)
    games = json.loads(convertedjson)
    weekPredictions = load_and_predict(games, model, modeltype)
    return weekPredictions

def predict_upcoming_week(season, week, model, overwrite=True, modeltype='Classification'):
    #warning! this will upset the caches so evaluating the last week will no longer be possible after running this
    games = get_weekly_games(season, week, overwrite)

    #now that we have the games, we need to go calculate the averages from the data we are now storing in the database
    rows = []
    # print("Processing... ", season, week)
    for game in games:
        row = {}
        if game["HomeTm"] != "" and game["VisTm"] != "":

            HomeAvgs = get_team_recent_stats(season=season, teamName=game["HomeTm"], week=week)
            AwayAvgs = get_team_recent_stats(season=season, teamName=game["VisTm"], week=week)

            if HomeAvgs != False and AwayAvgs != False:
                for key in AwayAvgs.keys():
                    row["away" + key] = AwayAvgs[key]
                for key in HomeAvgs.keys():
                    row["home" + key] = HomeAvgs[key]
                row['week'] = int(week)
                # date_format = "%B %d %Y"
                # row['Date'] = datetime.datetime.strptime(game['Date']+" 2024", date_format).strftime("%Y-%m-%d")
                row['Date'] = game['Date']
                row["awayTeamShort"] = ProFootballReferenceService.teamMap[game["VisTm"]]
                row["homeTeamShort"] = ProFootballReferenceService.teamMap[game["HomeTm"]]
                rows.append(row)

    games = calculate_previous_opp_record(rows, 4)

    #write it and reload it so it converts Decimal types to floats
    f = open("week" + str(week) + "games.json", "w")
    f.write(json.dumps(games, indent=4, default=float))
    f.close()

    f = open("week" + str(week) + "games.json", "r")
    games = json.load(f)
    f.close()



    weekPredictions = load_and_predict(games, model, modeltype)
    return weekPredictions

def insert_games_into_db(games):
    mycursor = mydb.cursor()
    for game in games:
        sql = """INSERT INTO game_log (
        awayTeam,
        awayTeamShort,
        awayavgScore,
        awayavgFirstDowns,
        awayavgTurnoversLost,
        awayavgPassingYards,
        awayavgRushingYards,
        awayavgOffensiveYards,
        awayavgPassingYardsAllowed,
        awayavgRushingYardsAllowed,
        awayavgTurnoversForced,
        awayavgYardsAllowed,
        awayavgOppScore,
        awayWins,
        awayStreak,
        awaySOS,
        homeTeam,
        homeTeamShort,
        homeavgScore,
        homeavgFirstDowns,
        homeavgTurnoversLost,
        homeavgPassingYards,
        homeavgRushingYards,
        homeavgOffensiveYards,
        homeavgPassingYardsAllowed,
        homeavgRushingYardsAllowed,
        homeavgTurnoversForced,
        homeavgYardsAllowed,
        homeavgOppScore,
        homeWins,
        homeStreak,
        homeSOS,
        AwayScore,
        HomeScore,
        Winner,
        actualSpread,
        Date,
        VegasLine,
        week,
        season,
        awayRecordAgainstOpp,
        homeRecordAgainstOpp
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        game["awayTeamShort"] = ProFootballReferenceService.teamMap[game["awayTeam"]]
        game["homeTeamShort"] = ProFootballReferenceService.teamMap[game["homeTeam"]]
        val = (
            game["awayTeam"],
            game["awayTeamShort"],
            game["awayavgScore"],
            game["awayavgFirstDowns"],
            game["awayavgTurnoversLost"],
            game["awayavgPassingYards"],
            game["awayavgRushingYards"],
            game["awayavgOffensiveYards"],
            game["awayavgPassingYardsAllowed"],
            game["awayavgRushingYardsAllowed"],
            game["awayavgTurnoversForced"],
            game["awayavgYardsAllowed"],
            game["awayavgOppScore"],
            game["awayWins"],
            game["awayStreak"],
            game["awaySOS"],
            game["homeTeam"],
            game["homeTeamShort"],
            game["homeavgScore"],
            game["homeavgFirstDowns"],
            game["homeavgTurnoversLost"],
            game["homeavgPassingYards"],
            game["homeavgRushingYards"],
            game["homeavgOffensiveYards"],
            game["homeavgPassingYardsAllowed"],
            game["homeavgRushingYardsAllowed"],
            game["homeavgTurnoversForced"],
            game["homeavgYardsAllowed"],
            game["homeavgOppScore"],
            game["homeWins"],
            game["homeStreak"],
            game["homeSOS"],
            game["AwayScore"],
            game["HomeScore"],
            game["Winner"],
            game["actualSpread"],
            game["Date"],
            game["VegasLine"],
            game["week"],
            game["season"],
            game["awayRecordAgainstOpp"],
            game["homeRecordAgainstOpp"]
        )
        mycursor.execute(sql, val)
        mydb.commit()

def insert_team_season_into_db(games):
    mycursor = mydb.cursor()
    for game in games:
        date_format = "%B %d %Y"
        formatted_date = datetime.datetime.strptime(game['Date']+" "+str(game['Season']), date_format).strftime("%Y-%m-%d")

        sql = """INSERT IGNORE INTO season_data_by_team (Team,
            Season,
            Week, 
            Day, 
            Date, 
            Time, 
            boxlink, 
            `W/L`, 
            OT, 
            Rec, 
            at, 
            Opponent, 
            Tm, 
            Opp, 
            O1stD, 
            OTotYd, 
            OPassY, 
            ORushY, 
            OTO, 
            D1stD, 
            DTotYd, 
            DPassY, 
            DRushY, 
            DTO, 
            Offense, 
            Defense, 
            `Sp. Tms`,
            SOS,
            isoDate
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        val = (
            game['Team'],
            game['Season'],
            game['Week'],
            game['Day'],
            game['Date'],
            game['Time'],
            game['boxlink'],
            game['W/L'],
            game['OT'],
            game['Rec'],
            game['at'],
            game['Opponent'],
            game['Tm'],
            game['Opp'],
            game['O1stD'],
            game['OTotYd'],
            game['OPassY'],
            game['ORushY'],
            game['OTO'],
            game['D1stD'],
            game['DTotYd'],
            game['DPassY'],
            game['DRushY'],
            game['DTO'],
            game['Offense'],
            game['Defense'],
            game['Sp. Tms'],
            game['SOS'],
            formatted_date
        )
        mycursor.execute(sql, val)
        mydb.commit()

def populate_db_from_file(filename='data/alldata.json'):
    f = open("data/alldata.json", "r")
    alldata = json.load(f)
    f.close()
    insert_games_into_db(alldata)

def evaluate_past_week_and_update_running_totales(season, week):
    # evaluate past weeks predictions
    week = week - 1
    evaluations = {}
    first = True
    models = ["trainedClassifier.keras", "trainedRegressor.keras"]
    for model_label in models:
        evaluations[model_label] = evaluate_past_week(season, week, model=model_label, overwrite=first)
        first = False
    f = open("week" + str(week) + "evaluations.json", "w")
    f.write(json.dumps(evaluations, indent=4))
    f.close()

    # evaluate models running accuracy
    startweek = 1
    endweek = week
    totals = {
        "startweek": startweek,
        "endweek": endweek,
        "trainedClassifier.keras": {
            "spreadDiff": 0,
            "correctPickNum": 0,
            "totalmoney": 0,
        },
        "trainedRegressor.keras": {
            "spreadDiff": 0,
            "correctPickNum": 0,
            "totalmoney": 0,
        }
    }

    for week in range(startweek, endweek + 1):
        f = open("week" + str(week) + "evaluations.json", "r")
        evaluations = json.load(f)
        f.close()
        for model in evaluations:
            totals[model]["spreadDiff"] += evaluations[model]["spreadDiff"]
            totals[model]["correctPickNum"] += evaluations[model]["correctPickNum"]
            totals[model]["totalmoney"] += evaluations[model]["totalmoney"]
    f = open("runningEvaluations.json", "w")
    f.write(json.dumps(totals, indent=4))
    f.close()

def supplement_all_data():
    f = open("data/alldata.json", "r")
    alldata = json.load(f)
    f.close()
    # now that we have the games, we need to go calculate the averages from the data we are now storing in the database
    rows = []
    # print("Processing... ", season, week)
    for game in alldata:
        row = {}
        if game["homeTeam"] != "" and game["awayTeam"] != "":

            HomeAvgs = get_team_recent_stats(season=game['season'], teamName=game["homeTeam"], week=game['week'])
            AwayAvgs = get_team_recent_stats(season=game['season'], teamName=game["awayTeam"], week=game['week'])

            if HomeAvgs != False and AwayAvgs != False:
                for key in AwayAvgs.keys():
                    row["away" + key] = AwayAvgs[key]
                for key in HomeAvgs.keys():
                    row["home" + key] = HomeAvgs[key]
                row['week'] = int(game['week'])
                row['season'] = int(game['season'])
                # date_format = "%B %d %Y"
                # row['Date'] = datetime.datetime.strptime(game['Date']+" 2024", date_format).strftime("%Y-%m-%d")
                row['Date'] = game['Date']
                row["awayTeamShort"] = ProFootballReferenceService.teamMap[game["awayTeam"]]
                row["homeTeamShort"] = ProFootballReferenceService.teamMap[game["homeTeam"]]
                row["AwayScore"] = game["AwayScore"]
                row["HomeScore"] = game["HomeScore"]
                row["Winner"] = game["Winner"]
                row["actualSpread"] = game["actualSpread"]
                row["VegasLine"] = game["VegasLine"]
                rows.append(row)

    games = calculate_previous_opp_record(rows, 4)
    f = open("data/alldata.json", "w")
    f.write(json.dumps(games, indent=4, default=float))
    f.close()

def populate_team_historic_data_to_db():
    seasons = [
        {
            "season": 2024,
        },
        {
            "season": 2023,
        },
        {
            "season": 2022,
        },
        {
            "season": 2021,
        },
        {
            "season": 2020,
        },
        {
            "season": 2019,
        },
        {
            "season": 2018,
        },
        {
            "season": 2017,
        },
        {
            "season": 2016,
        },
        {
            "season": 2015,
        },
        {
            "season": 2014,
        },
        {
            "season": 2013,
        },
        {
            "season": 2012,
        },
        {
            "season": 2011,
        },
        {
            "season": 2010,
        },
        {
            "season": 2009,
        },
        {
            "season": 2008,
        }
    ]

    footballservice = ProFootballReferenceService()
    for thisseason in seasons:
        for team in footballservice.teams:
            games = footballservice.get_team_season_data(thisseason['season'], team)
            insert_team_season_into_db(games)

def populate_team_season_data_to_db(season):
    footballservice = ProFootballReferenceService()
    for team in footballservice.teams:
        games = footballservice.get_team_season_data(season, team)
        insert_team_season_into_db(games)

if __name__ == '__main__':
    model_label = 'trainedRegressor.keras'
    modeltype = 'Regression'
    outputParam = 'actualSpread'

    #purge cache data for team games
    # for team in ProFootballReferenceService.teamMap.values():
    #     url = "https://www.pro-football-reference.com/years/2023/games.htm"
    #     hash = hashlib.md5(url.encode('UTF-8')).hexdigest()
    #     print(url, hash)
    #     if os.path.exists("caches/"+hash):
    #         print("removing 2023 cache data")
    #         os.remove("caches/"+hash)

    # overwrite a particular cache
    # service = ProFootballReferenceService()
    # data = service.get_or_fetch_from_cache(endpoint="years/2023/games.htm", overwrite=True)
    # print(data)

    #populate the team seasson database fully: this populates the season_data_by_team
    # populate_team_historic_data_to_db()
    # exit(1)

    # # generate the start of alldata file from the cached footballservice data
    # service = ProFootballReferenceService()
    # service.dump_historic_data()
    # exit(1)

    # #supplement the "alldata" with team averages
    # supplement_all_data()
    # exit(1)

    #insert the data from alldata into the database
    # populate_db_from_file()
    # exit(1)

    # might want to integrate sacks into inputs
    # train_and_evaluate_model(outlierspread=25, outputParam=outputParam, type=modeltype, modelLabel=model_label)

    # #now that we have a trained model, lets simluate its performance against the 2023 season
    # seasondata = generate_training_data_from_db(2023, 2023)
    # predictions = load_and_predict_from_training(seasondata, model_label, outputParams=outputParam)
    # if modeltype == 'Classification':
    #     evaluations = evaluate_full_season_classification(seasondata, predictions)
    # if modeltype == 'Regression':
    #     evaluations = evaluate_full_season(seasondata, predictions)
    #
    # print(evaluations)
    # exit(1)

    # predictions = predict_past_week(2024, 8, model_label, overwrite=False, modeltype=modeltype)
    # print(json.dumps(predictions, indent=4, default=float))
    # exit(1)

    season = 2024
    week = 10

    #start by going at getting the
    evaluate_past_week_and_update_running_totales(season, week)

    #now update the team season data table
    populate_team_season_data_to_db(season)

    #now updated the game_log_table
    insert_past_week_into_game_log(season, week)

    #generate this weeks predictions
    predictions = {}
    model_label = 'trainedRegressor.keras'
    modeltype = 'Regression'
    predictions[model_label] = predict_upcoming_week(season, week, model_label, overwrite=False, modeltype=modeltype)
    model_label = 'trainedClassifier.keras'
    modeltype = 'Classification'
    predictions[model_label] = predict_upcoming_week(season, week, model_label, overwrite=False, modeltype=modeltype)

    f = open("week" + str(week) + "predictions.json", "w")
    f.write(json.dumps(predictions, indent=4))
    f.close()