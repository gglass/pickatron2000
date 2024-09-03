import json
from ProFootballReferenceService import ProFootballReferenceService
import numpy as np
import pandas as pd
from tensorflow import keras
import keras_tuner
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

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
    dnnmodel.add(keras.layers.Input(shape=(29,)))
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

def preprocess_data(outlierspread=10):
    f = open("inputs/trainingdata.json", "r")
    data = json.load(f)
    f.close()
    # print("training data set size =", len(data))
    raw = pd.DataFrame(data)
    dataset = raw.copy()
    dataset = dataset.dropna()
    # clean up games where the spread was wildly out of the norm
    dataset['outlier'] = np.abs(dataset['actualSpread']) > outlierspread
    dataset = dataset[dataset['outlier'] == False]
    train_dataset = dataset.drop(columns=["AwayScore", "HomeScore", "hometeam", "awayteam", "Date", 'outlier', 'VegasLine'])
    # print(train_dataset.describe().transpose()[['mean', 'std']])

    f = open("inputs/testdata.json", "r")
    data = json.load(f)
    f.close()
    # print("test data set size =", len(data))
    raw = pd.DataFrame(data)
    dataset = raw.copy()
    dataset = dataset.dropna()
    # clean up games where the spread was wildly out of the norm
    dataset['outlier'] = np.abs(dataset['actualSpread']) > outlierspread
    dataset = dataset[dataset['outlier'] == False]
    test_dataset = dataset.drop(columns=["AwayScore", "HomeScore", "hometeam", "awayteam", "Date", 'outlier', 'VegasLine'])
    return train_dataset, test_dataset

def train_and_evaluate_model(outlierspread=10):
    train_dataset, test_dataset, = preprocess_data(outlierspread)
    train_inputs = train_dataset.copy()
    eval_inputs = test_dataset.copy()
    train_outputs = train_inputs.pop("actualSpread")
    eval_outputs = eval_inputs.pop("actualSpread")

    train_inputs = train_inputs.to_numpy()
    eval_inputs = eval_inputs.to_numpy()
    train_outputs = train_outputs.to_numpy()
    eval_outputs = eval_outputs.to_numpy()

    label = 'trainedSimple.keras'

    # dnn = build_and_compile_model()
    # history = dnn.fit(train_inputs,
    #           train_outputs,
    #           validation_split=0.15,
    #           verbose=1,
    #           epochs=250,
    #           batch_size=60,
    #           shuffle=True
    #       )

    tuner = keras_tuner.RandomSearch(
        hypermodel=build_and_compile_model_hp,
        objective=keras_tuner.Objective("val_loss", direction="min"),
        max_trials=100,
        overwrite=True,
        executions_per_trial=5,
        directory="search_results",
        project_name="scratch",
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
    plt.xlabel('True Values [Spread]')
    plt.ylabel('Predictions [Spread]')
    lims = [-20, 20]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.show()

def load_and_predict_from_training(games, model='trained.keras'):
    dnn_model = keras.models.load_model(model)
    raw = pd.DataFrame(games)
    dataset = raw.copy()
    dataset = dataset.dropna()
    dataset = dataset.drop(columns=["hometeam", "awayteam", "AwayScore", "HomeScore", "actualSpread", "Date", "VegasLine"])

    predictions = dnn_model.predict(dataset)
    predicted_results = []
    for idx, game in enumerate(games):
        predicted_results.append({
            "awayteam": game['awayteam'],
            "hometeam": game['hometeam'],
            "spread": str(predictions[idx][0])
        })
        # print(game['awayteam'], "@", game['hometeam'], predictions[idx][0])
    return predicted_results

def load_and_predict(games, model='trained.keras'):
    dnn_model = keras.models.load_model(model)
    raw = pd.DataFrame(games)
    dataset = raw.copy()
    dataset = dataset.dropna()
    dataset = dataset.drop(columns=["hometeam", "awayteam"])

    predictions = dnn_model.predict(dataset)
    predicted_results = []
    for idx, game in enumerate(games):
        predicted_results.append({
            "awayteam": game['awayteam'],
            "hometeam": game['hometeam'],
            "spread": str(predictions[idx][0])
        })
        # print(game['awayteam'], "@", game['hometeam'], predictions[idx][0])
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
    spreadDiff = 0
    correctPicks = []
    correctPickNum = 0
    totalmoney = 0
    for id, prediction in enumerate(predictions):
        result = {}
        for game in results:
            if game['HomeTeam'] == prediction['hometeam'] and game['AwayTeam'] == prediction['awayteam']:
                result = game
        predictedspread = float(prediction['spread'])
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
        if vegas_fav == prediction['hometeam']:
            if actualspread == -vegas_spread:
                money = 0
            else:
                if predictedspread < -vegas_spread and actualspread < -vegas_spread:
                    money = 100
                if predictedspread > -vegas_spread and actualspread > -vegas_spread:
                    money = 100

        elif vegas_fav == prediction['awayteam']:
            if actualspread == vegas_spread:
                money = 0
            else:
                if predictedspread > vegas_spread and actualspread > vegas_spread:
                    money = 100
                if predictedspread < vegas_spread and actualspread < vegas_spread:
                    money = 100
        else:
            print("We couldn't find a match on teams... oh noes.")
            print(vegas_fav, prediction['awayteam'], prediction['hometeam'])

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
        # print(predictedspread, vegas, actualspread, money)

    return {"spreadDiff": spreadDiff, "correctPickNum": correctPickNum, "totalmoney": totalmoney}

def evaluate_full_season(games, predictions):
    # print("Evaluation model "+model)
    spreadDiff = 0
    correctPickNum = 0
    vegasCorrectPickNum = 0
    totalmoney = 0
    hometeamWins = 0
    awayteamWins = 0

    for id, prediction in enumerate(predictions):
        result = games[id]
        predictedspread = float(prediction['spread'])
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

        if vegas_fav == prediction['hometeam']:
            vegas_spread = -vegas_spread
            if actualspread == vegas_spread:
                money = 0
            else:
                if predictedspread < vegas_spread and actualspread < vegas_spread:
                    money = 100
                if predictedspread > vegas_spread and actualspread > vegas_spread:
                    money = 100

        elif vegas_fav == prediction['awayteam']:
            if actualspread == vegas_spread:
                money = 0
            else:
                if predictedspread > vegas_spread and actualspread > vegas_spread:
                    money = 100
                if predictedspread < vegas_spread and actualspread < vegas_spread:
                    money = 100
        else:
            print("We couldn't find a match on teams... oh noes.")
            print(vegas_fav, prediction['awayteam'], prediction['hometeam'])

        if vegas_spread > 0 and actualspread > 0:
            vegasCorrectPickNum += 1
        if vegas_spread < 0 and actualspread < 0:
            vegasCorrectPickNum += 1

        print("Predicted Spread:", predictedspread, "Actual Spread", actualspread, "Vegas Spread", vegas_spread, "Bet Result", money)
        totalmoney += money

    return {"spreadDiff": spreadDiff, "avgSpreadDiff": spreadDiff/len(games),"correctPickNum": correctPickNum, "vegasCorrectPickNum": vegasCorrectPickNum, "homeWins": hometeamWins, "awayWins": awayteamWins, "totalgames": len(games), "totalMoney": totalmoney}

def predict_past_week(season, week, model):
    games = get_past_weekly_games(season, week)
    predictions = load_and_predict(games, model)
    print(predictions)

def predict_upcoming_week(season, week, model, overwrite=True):
    #warning! this will upset the caches so evaluating the last week will no longer be possible after running this
    games = get_weekly_games(season, week, overwrite)
    predictions = load_and_predict(games, model)
    return predictions

if __name__ == '__main__':

    # might want to integrate sacks into inputs
    # generate_training_data()
    # exit(1)
    train_and_evaluate_model(outlierspread=25)

    #now that we have a trained model, lets simluate its performance against the 2023 season
    f = open("inputs/testdata.json", "r")
    seasondata = json.load(f)
    f.close()
    predictions = load_and_predict_from_training(seasondata, 'trainedSimple.keras')
    evaluations = evaluate_full_season(seasondata, predictions)
    print(evaluations)
    exit(1)

    season = 2023
    week = 18
    model_label = 'trainedSimple.keras'

    #evaluate past weeks predictions
    week = week - 1
    evaluations = {}
    first = True

    evaluations[model_label] = evaluate_past_week(season, week,model=model_label,overwrite=first)
    f = open("week" + str(week) + "evaluations.json", "w")
    f.write(json.dumps(evaluations, indent=4))
    f.close()

    #evaluate models running accuracy
    startweek = 9
    endweek = week
    totals = {
        "startweek": startweek,
        "endweek": endweek,
        model_label: {
            "spreadDiff": 0,
            "correctPickNum": 0,
            "totalmoney": 0,
        }
    }

    for week in range(startweek,endweek+1):
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

    #generate this weeks predictions
    week = week + 1
    predictions = {}
    predictions[model_label] = predict_upcoming_week(season, week, model_label, overwrite=first)
    f = open("week" + str(week) + "predictions.json", "w")
    f.write(json.dumps(predictions, indent=4))
    f.close()