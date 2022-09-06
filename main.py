import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

import constants
import stats
import utils
import model


def perform_eda():
    stats.examine_target_distribution()
    stats.examine_stratified_folds()


def train():

    df_train, df_test = utils.read_train_test_data()

    # train_data = [str(t) for t in df_train[constants.EXCERPT].values]
    # train_targets = [float(t) for t in df_train[constants.BT_EASINESS].values]

    train_data = [str(t) for t in df_train.iloc[:2000][constants.EXCERPT].values]
    train_targets = [float(t) for t in df_train.iloc[:2000][constants.BT_EASINESS].values]

    validation_data = [str(t) for t in df_train.iloc[2000:][constants.EXCERPT].values]
    validation_targets = [float(t) for t in df_train.iloc[2000:][constants.BT_EASINESS].values]

    hyperparams = {constants.BATCH_SIZE: 10,
                   constants.LEARNING_RATE: 9e-6,
                   constants.WEIGHT_DECAY: 0.01,
                   constants.NUM_EPOCH: 50,
                   constants.BIAS: True}
    config = {constants.NUM_LABELS: 1,
              constants.IS_MULTI_LABEL: False,
              constants.LOGGING_STEPS: 43}

    model.train(input_model_name=constants.ALBERT_BASE_V2,
                output_model_name='albert_deneme',
                train_data=train_data, train_targets=train_targets,
                hyperparams=hyperparams, config=config,
                validation_data=validation_data, validation_targets=validation_targets)

    dummy = -32


def test():
    df_train, df_test = utils.read_train_test_data()
    rmse = mean_squared_error(y_true=df_test[constants.BT_EASINESS],
                              y_pred=df_test[constants.BT_EASINESS],
                              squared=False)

    print(f'RMSE: {rmse}')

    number_of_bins = 10
    df_train.loc[:, constants.BIN] = pd.cut(df_train[constants.BT_EASINESS], bins=number_of_bins, labels=False)
    plt.figure(), sns.displot(data=df_train, x=constants.BIN, kind='hist', hue=constants.BIN, bins=number_of_bins,
                              binrange=(-0.5, number_of_bins - 0.5), stat='percent'), plt.show()

    n_folds = 5

    df_train = utils.split_stratified_folds(df=df_train, n_folds=n_folds, label=constants.BIN)

    dummy = -32


def main(name):
    print(name)

    perform_eda()
    # train()
    # test()

    dummy = -32


if __name__ == '__main__':
    main('CommonLit Readibility')
