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
    "awayavgScore": 30.0,
    "awayavgFirstDowns": 20.0,
    "awayavgTurnoversLost": 0.5,
    "awayavgPassingYards": 250.0,
    "awayavgRushingYards": 125.0,
    "awayavgOffensiveYards": 375.0,
    "awayavgPassingYardsAllowed": 280.0,
    "awayavgRushingYardsAllowed": 75.0,
    "awayavgTurnoversForced": 1.0,
    "awayavgYardsAllowed": 350.0,
    "awayavgOppScore": 24.0,
    "awayWins": 0.5,
    "awayStreak": 1,
    "awaySOS": 0.0,
    "homeTeam": "Baltimore Ravens",
    "homeavgScore": 30.0,
    "homeavgFirstDowns": 20.0,
    "homeavgTurnoversLost": 0.5,
    "homeavgPassingYards": 250.0,
    "homeavgRushingYards": 125.0,
    "homeavgOffensiveYards": 375.0,
    "homeavgPassingYardsAllowed": 280.0,
    "homeavgRushingYardsAllowed": 75.0,
    "homeavgTurnoversForced": 1.0,
    "homeavgYardsAllowed": 350.0,
    "homeavgOppScore": 24.0,
    "homeWins": 0.5,
    "homeStreak": 1,
    "homeSOS": 0.0,
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

def modify_inputs(inputs, start_vals, coefficient):
    for input in inputs:
        print("Modifying " + input)
    samplematchups = []
    xdata = []
    for i in range(1,11):
        xdata.append(i)
        sample = samplematchup.copy()
        idx = 0
        for input in inputs:
            sample[input] = start_vals[idx] + (coefficient[idx] * i)
            samplematchups.append(sample)
            idx += 1
    predictions = load_and_predict(samplematchups, model=model_label, modeltype=modeltype)
    ydata = []
    for prediction in predictions:
        ydata.append(round(float(prediction['prediction']),2))

    print([xdata, ydata])
    return [xdata, ydata]

def show_plots(axes, label):
    plt.figure(figsize=(16,9))
    plt.plot(axes[0], axes[1], label=label)
    plt.xlabel(label)
    plt.ylabel('Spread')
    plt.grid(True)
    plt.savefig('weightplots/'+label+'.png')
    plt.close()

if __name__ == '__main__':
    model_label = 'trainedRegressor.keras'
    modeltype = 'Regression'

    # plots = modify_inputs(["week"], [0], [1])
    # show_plots(plots, "week")
    #
    # plots = modify_inputs(["awayavgScore"], [10], [2])
    # show_plots(plots, "awayavgScore")
    #
    # plots = modify_inputs(["awayavgFirstDowns"], [10], [2])
    # show_plots(plots, "awayavgFirstDowns")
    #
    # plots = modify_inputs(["awayavgTurnoversLost"], [0], [0.2])
    # show_plots(plots, "awayavgTurnoversLost")
    #
    # plots = modify_inputs(["awayavgPassingYards"], [200], [15])
    # show_plots(plots, "awayavgPassingYards")
    #
    # plots = modify_inputs(["awayavgRushingYards"], [100], [15])
    # show_plots(plots, "awayavgRushingYards")
    #
    # plots = modify_inputs(["awayavgOffensiveYards"], [200], [30])
    # show_plots(plots, "awayavgOffensiveYards")
    #
    # plots = modify_inputs(["awayavgPassingYardsAllowed"], [200], [15])
    # show_plots(plots, "awayavgPassingYardsAllowed")
    #
    # plots = modify_inputs(["awayavgRushingYardsAllowed"], [100], [15])
    # show_plots(plots, "awayavgRushingYardsAllowed")
    #
    # plots = modify_inputs(["awayavgTurnoversForced"], [0], [0.4])
    # show_plots(plots, "awayavgTurnoversForced")
    #
    # plots = modify_inputs(["awayavgYardsAllowed"], [200], [30])
    # show_plots(plots, "awayavgYardsAllowed")
    #
    # plots = modify_inputs(["awayavgOppScore"], [10], [2])
    # show_plots(plots, "awayavgOppScore")
    #
    # plots = modify_inputs(["awayWins"], [0], [0.1])
    # show_plots(plots, "awayWins")
    #
    # plots = modify_inputs(["awayStreak"], [0], [1])
    # show_plots(plots, "awayStreak")
    #
    # plots = modify_inputs(["awaySOS"], [-2.5], [0.5])
    # show_plots(plots, "awaySOS")
    #
    # plots = modify_inputs(["homeavgScore"], [10], [2])
    # show_plots(plots, "homeavgScore")
    #
    # plots = modify_inputs(["homeavgFirstDowns"], [10], [2])
    # show_plots(plots, "homeavgFirstDowns")
    #
    # plots = modify_inputs(["homeavgTurnoversLost"], [0], [0.2])
    # show_plots(plots, "homeavgTurnoversLost")
    #
    # plots = modify_inputs(["homeavgPassingYards"], [200], [15])
    # show_plots(plots, "homeavgPassingYards")
    #
    # plots = modify_inputs(["homeavgRushingYards"], [100], [15])
    # show_plots(plots, "homeavgRushingYards")
    #
    # plots = modify_inputs(["homeavgOffensiveYards"], [200], [30])
    # show_plots(plots, "homeavgOffensiveYards")
    #
    # plots = modify_inputs(["homeavgPassingYardsAllowed"], [200], [15])
    # show_plots(plots, "homeavgPassingYardsAllowed")
    #
    # plots = modify_inputs(["homeavgRushingYardsAllowed"], [100], [15])
    # show_plots(plots, "homeavgRushingYardsAllowed")
    #
    # plots = modify_inputs(["homeavgTurnoversForced"], [0], [0.2])
    # show_plots(plots, "homeavgTurnoversForced")
    #
    # plots = modify_inputs(["homeavgYardsAllowed"], [200], [30])
    # show_plots(plots, "homeavgYardsAllowed")
    #
    # plots = modify_inputs(["homeavgOppScore"], [10], [2])
    # show_plots(plots, "homeavgOppScore")
    #
    # plots = modify_inputs(["homeWins"], [0], [0.1])
    # show_plots(plots, "homeWins")
    #
    # plots = modify_inputs(["homeStreak"], [0], [1])
    # show_plots(plots, "homeStreak")
    #
    # plots = modify_inputs(["homeSOS"], [-2.5], [0.5])
    # show_plots(plots, "homeSOS")
    #
    # plots = modify_inputs(["awayRecordAgainstOpp"], [0], [0.1])
    # show_plots(plots, "awayRecordAgainstOpp")
    #
    # plots = modify_inputs(["homeRecordAgainstOpp"], [0], [0.1])
    # show_plots(plots, "homeRecordAgainstOpp")

    plots = modify_inputs(["homeavgTurnoversForced", "awayavgTurnoversLost"], [0, 0], [0.2, 0.2])
    show_plots(plots, "homeavgTurnoversForced_with_awayavgTurnoversLost")

