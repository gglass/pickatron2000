import json
from ProFootballReferenceService import ProFootballReferenceService
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import autokeras as ak
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import random

nn_sizes = [
    [52, 52, 13],
    [52, 26, 13],
    [52, 13, 13],
    [48, 13, 13],
    [26, 26, 13],
    [26, 13, 13],
    [13, 13, 4],
    [13, 8, 4],
    [52, 52],
    [52, 26],
    [52, 13],
    [52, 8],
    [26, 13],
    [26, 8],
    [13, 13],
    [13, 8],
    [13, 4]
]

import warnings
warnings.filterwarnings("ignore")
# best candidates so far [48->8]

def build_and_compile_model(norm,sizes):
  # model = keras.Sequential([
  #     # norm,
  #     keras.layers.Input(shape=(24,)),
  #     keras.layers.Dense(24, activation='relu'),
  #     keras.layers.Dense(12, activation='relu'),
  #     keras.layers.Dense(1)
  # ])

  model = keras.Sequential([keras.layers.Input(shape=(26,))])
  for neurons in sizes:
      model.add(keras.layers.Dense(neurons, activation='relu'))
  model.add(keras.layers.Dense(1))

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

def auto_regress_model():
    reg = ak.StructuredDataRegressor(overwrite=True, loss="mean_absolute_error")
    return reg

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

def train_and_evaluate_model(auto = False, max_iterations=50, outlierspread=10):
    f = open("inputs/trainingdata.json", "r")
    data = json.load(f)
    f.close()
    print("input data set size =", len(data))
    raw = pd.DataFrame(data)
    dataset = raw.copy()
    dataset = dataset.dropna()
    #clean up games where the spread was wildly out of the norm
    dataset['outlier'] = np.abs(dataset['actualSpread']) > outlierspread
    dataset = dataset[dataset['outlier'] == False]
    train_dataset = dataset.drop(columns=["AwayScore", "HomeScore", "hometeam", "awayteam", "Date", 'outlier'])
    print(train_dataset.describe().transpose()[['mean', 'std']])

    f = open("inputs/testdata.json", "r")
    data = json.load(f)
    f.close()
    print("input data set size =", len(data))
    raw = pd.DataFrame(data)
    dataset = raw.copy()
    dataset = dataset.dropna()
    #clean up games where the spread was wildly out of the norm
    dataset['outlier'] = np.abs(dataset['actualSpread']) > 20
    dataset = dataset[dataset['outlier'] == False]
    test_dataset = dataset.drop(columns=["AwayScore", "HomeScore", "hometeam", "awayteam", "Date", 'outlier'])

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop("actualSpread")
    test_labels = test_features.pop("actualSpread")

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    # first = np.array(train_features[:1])
    # with np.printoptions(precision=2, suppress=True):
    #     print('First example:', first)
    #     print()
    #     print('Normalized:', normalizer(first).numpy())
    # exit()



    results = []

    for nnsize in nn_sizes:
        model_label = 'trained'
        for layer in nnsize:
            model_label = model_label+str(layer)
        model_label = model_label + '.keras'
        iterations = 1
        rval = -1
        rvals = []
        while iterations < max_iterations + 1:
            if auto:
                reg = auto_regress_model()
                history = reg.fit(
                    train_features,
                    train_labels,
                    validation_split=0.15,
                    verbose=1,
                    epochs=300,
                    # batch_size=240,
                )

                dnn_model = reg.export_model()

            else:
                dnn_model = build_and_compile_model(normalizer, nnsize)
                history = dnn_model.fit(
                    train_features,
                    train_labels,
                    validation_split=0.15,
                    verbose=0,
                    epochs=100,
                    batch_size=64,
                    shuffle=True
                )

                # plot_loss(history)

            print(dnn_model.summary())
            print(dnn_model.evaluate(test_features, test_labels, verbose=0))
            test_predictions = dnn_model.predict(test_features).flatten()


            # error = test_predictions - test_labels
            # plt.hist(error, bins=25)
            # plt.xlabel('Prediction Error [Spread]')
            # plt.ylabel('Count')
            # plt.show()

            thisr2 = r2_score(test_labels, test_predictions)
            print("Training ", model_label)
            print("Training Set R-Square=", thisr2)
            print("Best so far=", rval)
            print("Iterations count=", iterations)
            iterations += 1
            if thisr2 > rval:
                rval = thisr2
                print("Saving incremental model", rval)
                dnn_model.save(model_label)
                a = plt.axes(aspect='equal')
                plt.scatter(test_labels, test_predictions)
                plt.xlabel('True Values [Spread]')
                plt.ylabel('Predictions [Spread]')
                lims = [-20, 20]
                plt.xlim(lims)
                plt.ylim(lims)
                plt.plot(lims, lims)
                plt.show()

            rvals.append(rval)

        plt.xlabel(model_label)
        plt.ylabel('Best r2')
        plt.plot(range(1,max_iterations+1), rvals)
        plt.show()
        results.append({model_label:rval})

    print(results)

def load_and_predict(games, model='trained.keras'):
    dnn_model = tf.keras.models.load_model(model)
    # for layer in dnn_model.layers:
    #     print(layer.get_config(), layer.get_weights())
    raw = pd.DataFrame(games)
    dataset = raw.copy()
    dataset = dataset.dropna()
    # dataset = dataset.drop(columns=["hometeam", "awayteam", "AwayScore", "HomeScore", "actualSpread", "Date"])
    dataset = dataset.drop(columns=["hometeam", "awayteam"])

    predictions = dnn_model.predict(dataset)
    predicted_results = []
    for idx, game in enumerate(games):
        predicted_results.append({
            "awayteam": game['awayteam'],
            "hometeam": game['hometeam'],
            "spread": predictions[idx][0]
        })
        # print(game['awayteam'], "@", game['hometeam'], predictions[idx][0])
    return predicted_results

def load_and_eval(predictions, results):

        return 1

def load_and_sum(games, model='trained.keras'):
    dnn_model = tf.keras.models.load_model(model)
    # for layer in dnn_model.layers:
    #     print(layer.get_config(), layer.get_weights())
    raw = pd.DataFrame(games)
    dataset = raw.copy()
    dataset = dataset.dropna()
    dataset = dataset.drop(columns=["hometeam", "awayteam", "AwayScore", "HomeScore", "actualSpread", "Date"])

    predictions = dnn_model.predict(dataset)
    sums = 0
    for idx, game in enumerate(games):
        sums += predictions[idx][0]
    print(model, sums)

def get_weekly_games(season, week):
    service = ProFootballReferenceService()
    return service.get_upcoming_inputs(season, week, overwrite=False)

def get_past_weekly_games(season, week):
    service = ProFootballReferenceService()
    return service.get_weekly_inputs(season, week, overwrite=True)

def get_weekly_results(season, week):
    service = ProFootballReferenceService()
    return service.get_weekly_results(season, week)

def evaluate_past_week(season, week, model):
    games = get_past_weekly_games(season, week)
    predictions = load_and_predict(games, model)
    results = get_weekly_results(season, week)
    spreadDiff = 0
    correctPicks = []
    correctPickNum = 0
    for id, prediction in enumerate(predictions):
        predictedspread = prediction['spread']
        actualspread = results[id]['actualSpread']
        diff = actualspread - predictedspread
        spreadDiff += abs(diff)
        correctPick = 'false'
        if predictedspread > 0 and actualspread > 0:
            correctPick = 'true'
            correctPickNum += 1
        if predictedspread < 0 and actualspread < 0:
            correctPick = 'true'
            correctPickNum += 1
        correctPicks.append(correctPick)

    return {model: {"spreadDiff": spreadDiff, "correctPickNum": correctPickNum}}

def predict_past_week(season, week, model):
    games = get_past_weekly_games(season, week)
    predictions = load_and_predict(games, model)
    print(predictions)

def predict_upcoming_week(season, week, model):
    #warning! this will upset the caches so evaluating the last week will no longer be possible after running this
    games = get_weekly_games(season, week)
    predictions = load_and_predict(games, model)
    print(predictions)

if __name__ == '__main__':
    # might want to integrate sacks into inputs
    # generate_training_data()
    # train_and_evaluate_model(auto=False,outlierspread=20,max_iterations=1)

    model_evaluations = []
    season = 2023
    week = 9

    # print(predict_past_week(season, week, 'trained521313.keras'))
    # print(evaluate_past_week(season, week, 'trained521313.keras'))

    # evaluations = []
    # model_label = ''
    # for nnsize in nn_sizes:
    #     model_label = 'trained'
    #     for layer in nnsize:
    #         model_label = model_label+str(layer)
    #     model_label = model_label + '.keras'
    #     evaluations.append(evaluate_past_week(season, week,model=model_label))
    # print(evaluations)

    # print(evaluate_past_week(season, week, 'trained521313.keras'))
    print(predict_upcoming_week(season, week, 'trained521313.keras'))