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

  model = keras.Sequential([keras.layers.Input(shape=(24,))])
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

    nn_sizes = [
        [48,48,12],
        [48,24,12],
        [48,12,12],
        [24,24,12],
        [24,12,12],
        [12,12,4],
        [12,8,4],
        [48,48],
        [48,24],
        [48,12],
        [48,8],
        [24,12],
        [24,8],
        [12,12],
        [12,8],
        [12,4]
    ]

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

            # print(dnn_model.summary())
            print(dnn_model.evaluate(test_features, test_labels, verbose=0))
            test_predictions = dnn_model.predict(test_features).flatten()

            # a = plt.axes(aspect='equal')
            # plt.scatter(test_labels, test_predictions)
            # plt.xlabel('True Values [Spread]')
            # plt.ylabel('Predictions [Spread]')
            # lims = [-20, 20]
            # plt.xlim(lims)
            # plt.ylim(lims)
            # plt.plot(lims, lims)
            # plt.show()

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
            rvals.append(rval)

        plt.xlabel(model_label)
        plt.ylabel('Best r2')
        plt.plot(range(1,max_iterations+1), rvals)
        plt.show()
        results.append({model_label:rval})

    print(results)

def load_and_predict(games, model='trained.keras'):
    dnn_model = tf.keras.models.load_model(model)
    raw = pd.DataFrame(games)
    dataset = raw.copy()
    dataset = dataset.dropna()
    dataset = dataset.drop(columns=["hometeam", "awayteam"])

    predictions = dnn_model.predict(dataset)

    for idx, game in enumerate(games):
        print(game['awayteam'], "@", game['hometeam'], predictions[idx][0])


def get_weekly_games(season, week):
    service = ProFootballReferenceService()
    return service.get_upcoming_inputs(season, week, overwrite=False)


if __name__ == '__main__':
    # generate_training_data()
    # train_and_evaluate_model(auto=False,outlierspread=20,max_iterations=35)
    season = 2023
    week = 2
    games = get_weekly_games(season, week)
    load_and_predict(games,model='trained242412.keras')