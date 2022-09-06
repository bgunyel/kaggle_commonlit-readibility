import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedKFold

import constants


def read_data():
    df = pd.read_csv(os.path.join(constants.DATA_FOLDER, constants.DATA_FILE))
    return df


def separate_train_test(df):
    df_train = df.loc[df[constants.KAGGLE_SPLIT] == constants.TRAIN]
    df_test = df.loc[df[constants.KAGGLE_SPLIT] == constants.TEST]

    df_train.reset_index(inplace=True)
    df_test.reset_index(inplace=True)
    return df_train, df_test


def read_train_test_data():
    df = read_data()
    df_train, df_test = separate_train_test(df=df)
    return df_train, df_test


def generate_local_model_folder_path(model_name):
    out = os.path.join(constants.MODELS_FOLDER, model_name)
    return out


def plot_triplet(df, feature):
    df_train, df_test = separate_train_test(df=df)

    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    grid = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
    ax1 = fig.add_subplot(grid[0, :2])
    ax2 = fig.add_subplot(grid[1, :2])
    ax3 = fig.add_subplot(grid[:, 2])

    ax1.set_title(f'{feature} - Train')
    ax2.set_title(f'{feature} - Test')
    ax3.set_title('Box Plots')

    sns.distplot(df_train.loc[:, feature], hist=True, kde=True, ax=ax1,
                 hist_kws={'rwidth': 0.85, 'edgecolor': 'black', 'alpha': 0.8})
    sns.distplot(df_test.loc[:, feature], hist=True, kde=True, ax=ax2,
                 hist_kws={'rwidth': 0.85, 'edgecolor': 'black', 'alpha': 0.8})
    sns.boxplot(x=constants.KAGGLE_SPLIT, y=feature, data=df, ax=ax3)
    plt.show()


def split_stratified_folds(df, n_folds, label):
    out_df = df.copy(deep=True)
    out_df[constants.FOLD] = -1

    stratified_fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=constants.RANDOM_SEED)
    for k, (train_idx, valid_idx) in enumerate(stratified_fold.split(X=out_df, y=out_df[label])):
        out_df.loc[valid_idx, constants.FOLD] = k

    return out_df


def compute_grid_dimensions(number_of_elements):

    n_rows = 1
    n_cols = 1

    while n_cols * n_rows < number_of_elements:
        if n_cols == n_rows:
            n_cols += 1
        else:
            n_rows += 1
    return n_rows, n_cols
