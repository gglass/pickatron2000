import json
from ProFootballReferenceService import ProFootballReferenceService
import numpy as np
import pandas as pd
from tensorflow import keras
import keras_tuner
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

samplematchup = {
    "awayTeam": "Cincinnati Bengals",
    "awayavgScore": 28.25,
    "awayavgFirstDowns": 20.5,
    "awayavgTurnoversLost": 0.875,
    "awayavgPassingYards": 246.625,
    "awayavgRushingYards": 97.25,
    "awayavgOffensiveYards": 343.875,
    "awayavgPassingYardsAllowed": 224.0,
    "awayavgRushingYardsAllowed": 125.5,
    "awayavgTurnoversForced": 1.25,
    "awayavgYardsAllowed": 349.5,
    "awayavgOppScore": 26.375,
    "awayWins": 0.5,
    "awayStreak": 1,
    "awaySOS": -2.64,
    "homeTeam": "Baltimore Ravens",
    "homeavgScore": 32.875,
    "homeavgFirstDowns": 24.25,
    "homeavgTurnoversLost": 0.625,
    "homeavgPassingYards": 252.375,
    "homeavgRushingYards": 192.75,
    "homeavgOffensiveYards": 445.125,
    "homeavgPassingYardsAllowed": 280.875,
    "homeavgRushingYardsAllowed": 76.125,
    "homeavgTurnoversForced": 1.0,
    "homeavgYardsAllowed": 357.0,
    "homeavgOppScore": 24.0,
    "homeWins": 0.75,
    "homeStreak": 1,
    "homeSOS": 1.03,
    "week": 10,
    "Date": "2024-11-07",
    "awayTeamShort": "cin",
    "homeTeamShort": "rav",
    "awayRecordAgainstOpp": 0.5,
    "homeRecordAgainstOpp": 0.5
}

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


if __name__ == '__main__':
    model_label = 'trainedRegressor.keras'
    modeltype = 'Regression'

    plots = []

    print("Modifying weeks")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["week"] = i
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    print("Modifying awayavgScore")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["awayavgScore"] = 10 + (2 * i)
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    print("Modifying awayavgFirstDowns")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["awayavgFirstDowns"] = 10 + (2 * i)
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    print("Modifying awayavgTurnoversLost")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["awayavgTurnoversLost"] = 0 + (0.1 * i)
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    print("Modifying awayavgPassingYards")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["awayavgPassingYards"] = 200 + (15 * i)
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    print("Modifying awayavgRushingYards")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["awayavgRushingYards"] = 100 + (15 * i)
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    print("Modifying awayavgOffensiveYards")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["awayavgOffensiveYards"] = 200 + (30 * i)
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    print("Modifying awayavgPassingYardsAllowed")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["awayavgPassingYardsAllowed"] = 200 + (15 * i)
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    print("Modifying awayavgRushingYardsAllowed")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["awayavgRushingYardsAllowed"] = 100 + (15 * i)
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    print("Modifying awayavgTurnoversForced")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["awayavgTurnoversForced"] = 0 + (0.25 * i)
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    print("Modifying awayavgYardsAllowed")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["awayavgYardsAllowed"] = 200 + (30 * i)
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    print("Modifying awayavgOppScore")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["awayavgOppScore"] = 10 + (2 * i)
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    print("Modifying awayWins")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["awayWins"] = 0 + (0.1 * (i - 1))
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    print("Modifying awayStreak")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["awayWins"] = 0 + i
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    print("Modifying awaySOS")
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        sample["awaySOS"] = -2.5 + (0.5 * i)
        samplematchups.append(sample)
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))
    plots.append([xdata, ydata])

    plt.plot(plots[0][0], plots[0][1], label="Week")
    plt.plot(plots[1][0], plots[1][1], label="awayAvgScore")
    plt.plot(plots[2][0], plots[2][1], label="awayavgFirstDowns")
    plt.plot(plots[3][0], plots[3][1], label="awayavgTurnoversLost")
    plt.plot(plots[4][0], plots[4][1], label="awayavgPassingYards")
    plt.plot(plots[5][0], plots[5][1], label="awayavgRushingYards")
    plt.plot(plots[6][0], plots[6][1], label="awayavgOffensiveYards")
    plt.plot(plots[7][0], plots[7][1], label="awayavgPassingYardsAllowed")
    plt.plot(plots[8][0], plots[8][1], label="awayavgRushingYardsAllowed")
    plt.plot(plots[9][0], plots[9][1], label="awayavgTurnoversForced")
    plt.plot(plots[10][0], plots[10][1], label="awayavgYardsAllowed")
    plt.plot(plots[11][0], plots[11][1], label="awayavgOppScore")
    plt.plot(plots[12][0], plots[12][1], label="awayWins")
    plt.plot(plots[13][0], plots[13][1], label="awayStreak")
    plt.plot(plots[14][0], plots[14][1], label="awaySOS")
    plt.ylim([-5, 30])
    plt.xlabel('Input')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True)
    plt.show()