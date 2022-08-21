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
