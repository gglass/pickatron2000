import json
from ProFootballReferenceService import ProFootballReferenceService
import numpy as np
import pandas as pd
from tensorflow import keras
import keras_tuner
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import hashlib

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

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 25])
  plt.xlabel('Epoch')
  plt.ylabel('Error [ActualSpread]')
  plt.legend()
  plt.grid(True)
  plt.show()

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
    output = calculate_previous_opp_record(games, 4)
    return output

def calculate_previous_opp_record(games, numMatchups=4):
    # now that we have the base data, lets go calculate some historic stuff
    mycursor = mydb.cursor()
    output = []
    for game in games:
        if type(game['Date']) is not str:
            game["Date"] = game["Date"].strftime("%Y-%m-%d")
        homeTeam = game["homeTeamShort"]
        awayTeam = game["awayTeamShort"]
        awayWins = 0
        homeWins = 0
        previousGames = 0

        # lets go get (up to) the last 4 matchups of these two teams to see how they did
        sql = "SELECT * FROM game_log WHERE date < %s AND (homeTeamShort = %s OR awayTeamShort = %s) AND (awayTeamShort = %s OR homeTeamShort = %s) ORDER BY Date desc LIMIT "+str(numMatchups)
        val = (game["Date"], homeTeam, homeTeam, awayTeam, awayTeam)
        mycursor.execute(sql, val)
        columns = mycursor.description

        pastMatchups = [{columns[index][0]: column for index, column in enumerate(value)} for value in
                        mycursor.fetchall()]

        for matchup in pastMatchups:
            previousGames += 1
            if matchup['awayTeamShort'] == awayTeam:
                if matchup['actualSpread'] < 0:
                    homeWins += 1
                else:
                    awayWins += 1
            else:
                if matchup['actualSpread'] < 0:
                    awayWins += 1
                else:
                    homeWins += 1

        if previousGames > 0:
            game["awayRecordAgainstOpp"] = awayWins / previousGames
            game["homeRecordAgainstOpp"] = homeWins / previousGames
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
            max_trials=25,
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

def get_past_weekly_games(season, week):
    service = ProFootballReferenceService()
    return service.get_weekly_inputs(season, week, overwrite=True)

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
    if overwrite == True:
        insert_games_into_db(results)

    spreadDiff = 0
    correctPicks = []
    correctPickNum = 0
    totalmoney = 0
    for id, prediction in enumerate(predictions):
        result = {}
        for game in results:
            if game['hometeam'] == prediction['homeTeam'] and game['awayteam'] == prediction['awayTeam']:
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

def predict_past_week(season, week, model):
    games = get_past_weekly_games(season, week)
    thispredictions = load_and_predict(games, model)
    print(predictions)

def predict_upcoming_week(season, week, model, overwrite=True, modeltype='Classification'):
    #warning! this will upset the caches so evaluating the last week will no longer be possible after running this
    games = get_weekly_games(season, week, overwrite)
    f = open("week" + str(week) + "games.json", "w")
    f.write(json.dumps(games, indent=4))
    f.close()
    games = calculate_previous_opp_record(games, 4)
    weekPredictions = load_and_predict(games, model, modeltype)
    return weekPredictions

def insert_games_into_db(games):
    mycursor = mydb.cursor()
    for game in games:
        sql = """INSERT INTO game_log (awayTeam,
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
        season
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        game["awayTeamShort"] = ProFootballReferenceService.teamMap[game["awayteam"]]
        game["homeTeamShort"] = ProFootballReferenceService.teamMap[game["hometeam"]]
        val = (
            game["awayteam"],
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
            game["hometeam"],
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
            game["season"]
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

    # generate the alldata file from the cached footballservice data
    # service = ProFootballReferenceService()
    # service.dump_historic_data()
    # exit(1)

    #insert the data from alldata into the database
    # populate_db_from_file()
    # exit(1)

    # might want to integrate sacks into inputs
    # generate_training_data_from_db(2008, 2022)
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

    season = 2024
    week = 6

    evaluate_past_week_and_update_running_totales(season, week)

    #generate this weeks predictions
    predictions = {}
    model_label = 'trainedRegressor.keras'
    modeltype = 'Regression'
    predictions[model_label] = predict_upcoming_week(season, week, model_label, overwrite=False, modeltype=modeltype)
    model_label = 'trainedClassifier.keras'
    modeltype = 'Classification'
    predictions[model_label] = predict_upcoming_week(season, week, model_label, overwrite=False, modeltype=modeltype)

    print(predictions)

    f = open("week" + str(week) + "predictions.json", "w")
    f.write(json.dumps(predictions, indent=4))
    f.close()