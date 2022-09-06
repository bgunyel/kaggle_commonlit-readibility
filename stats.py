import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import constants
import utils


def examine_target_distribution():
    df = pd.read_csv(os.path.join(constants.DATA_FOLDER, constants.DATA_FILE))
    utils.plot_triplet(df=df, feature=constants.BT_EASINESS)
    dummy = -32


def examine_stratified_folds(n_folds=5, n_bins=10):
    df_train, df_test = utils.read_train_test_data()

    df_train.loc[:, constants.BIN] = pd.cut(df_train[constants.BT_EASINESS], bins=n_bins, labels=False)
    plt.figure()
    sns.displot(data=df_train,
                x=constants.BIN,
                kind='hist',
                hue=constants.BIN,
                bins=n_bins,
                binrange=(-0.5, n_bins - 0.5),
                stat='percent')
    plt.show()

    n_folds = 5

    df_train = utils.split_stratified_folds(df=df_train, n_folds=n_folds, label=constants.BIN)

    n_rows, n_cols = utils.compute_grid_dimensions(number_of_elements=n_folds+1)

    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    grid = gridspec.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)

    axes = []

    for i in range(n_folds+1):
        axes.append(fig.add_subplot(grid[i // n_cols, i % n_cols]))

        if i == 0:
            # sns.displot(data=df_train, x=constants.BIN, kind='hist', hue=constants.BIN,
            #             bins=n_bins, binrange=(-0.5, n_bins - 0.5), stat='percent', ax=axes[i])

            sns.histplot(data=df_train, x=constants.BIN, hue=constants.BIN,
                         bins=n_bins, binrange=(-0.5, n_bins - 0.5), stat='percent', ax=axes[i])

            axes[i].set_title('Train Set - All')
        else:
            axes[i].set_title(f'Train Set - Fold {i}')

            sns.histplot(data=df_train.loc[df_train[constants.FOLD] == i-1], x=constants.BIN,
                         hue=constants.BIN,
                         bins=n_bins, binrange=(-0.5, n_bins - 0.5), stat='percent', ax=axes[i])

    plt.show()

    dummy = -32
