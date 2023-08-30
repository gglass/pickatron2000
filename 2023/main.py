import json
from ProFootballReferenceService import ProFootballReferenceService
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import autokeras as ak
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def build_and_compile_model(norm):
  model = keras.Sequential([
      # norm,
      # keras.layers.Dense(64, activation='relu'),
      keras.layers.Input(shape=(24,)),
      # keras.layers.Dense(32, activation='relu'),
      keras.layers.Dense(16, activation='relu'),
      keras.layers.Dense(8, activation='relu'),
      keras.layers.Dense(1)
  ])

  #this is pulled straight from the Clarity RiskScoring
  # model = tf.keras.Sequential([
  #     keras.layers.Input(shape=(24, 1)),
  #     keras.layers.Conv1D(128, kernel_size=1, activation=tf.nn.relu),
  #     keras.layers.Conv1D(64, kernel_size=1, activation=tf.nn.relu),
  #     keras.layers.Conv1D(32, kernel_size=1, activation=tf.nn.relu),
  #     keras.layers.MaxPool1D(pool_size=2),
  #     keras.layers.Dense(32, activation='relu'),
  #     keras.layers.Dense(1)
  # ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

def auto_regress_model():
    reg = ak.StructuredDataRegressor(max_trials=15, overwrite=True)
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

def train_and_evaluate_model():
    f = open("inputs/historicalmatchups.json", "r")
    data = json.load(f)
    f.close()
    print("input data set size =", len(data))
    raw = pd.DataFrame(data)
    dataset = raw.copy()
    dataset = dataset.dropna()
    dataset = dataset.drop(columns=["AwayScore", "HomeScore", "hometeam", "awayteam", "Date"])
    train_dataset = dataset.sample(frac=0.9, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop("actualSpread")
    test_labels = test_features.pop("actualSpread")

    normalizer = tf.keras.layers.Normalization(axis=None)
    normalizer.adapt(np.array(train_features))

    # dnn_model = auto_regress_model(train_features, train_labels)
    dnn_model = build_and_compile_model(normalizer)

    print(dnn_model.summary())

    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=0.1,
        verbose=0, epochs=300)

    plot_loss(history)

    print(dnn_model.evaluate(test_features, test_labels, verbose=0))

    test_predictions = dnn_model.predict(test_features).flatten()

    # print(test_predictions, test_labels)
    # exit()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [Spread]')
    plt.ylabel('Predictions [Spread]')
    lims = [-40, 40]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.show()

    # error = test_predictions - test_labels
    # plt.hist(error, bins=25)
    # plt.xlabel('Prediction Error [Spread]')
    # plt.ylabel('Count')
    # plt.show()

    TestR2Value = r2_score(test_labels, test_predictions)
    print("Training Set R-Square=", TestR2Value)

    dnn_model.save("trained.keras")

def load_and_predict(games):
    dnn_model = tf.keras.models.load_model('trained.keras')
    raw = pd.DataFrame(games)
    dataset = raw.copy()
    dataset = dataset.dropna()
    dataset = dataset.drop(columns=["hometeam", "awayteam"])

    predictions = dnn_model.predict(dataset)

    for idx, game in enumerate(games):
        print(game['awayteam'], "@", game['hometeam'], predictions[idx][0])


def get_weekly_games(season, week):
    service = ProFootballReferenceService()
    return service.get_upcoming_inputs(season, week)


if __name__ == '__main__':
    season = 2023
    week = 1
    games = get_weekly_games(season, week)
    # generate_training_data()
    # train_and_evaluate_model()
    load_and_predict(games)