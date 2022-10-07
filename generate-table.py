
import pathlib
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from pyparsing import Dict
from tqdm import tqdm
import sys
import gc
import multiprocessing as mp
from utils import *
import seaborn as sns

RESULTS_FOLDER = '30-60-90-120'

results_path = pathlib.Path(f'./test-results/{RESULTS_FOLDER}')

if __name__ == '__main__':
    data = pd.read_csv('./test-results/results_summary.csv', sep=';')
    data["Prediction position"] = data["Prediction position"] + 1
    model_names1 = {
        "CNN-LSTM_network_absolute" : "CNN-LSTM (normal)",
        "CNN-LSTM_network_diffs_" : "CNN-LSTM (differential)",
        "feed-forward_network_absolute" : "Densely connected (normal)",
        "feed-forward_network_diffs" : "Densely connected (differential)",
        "LSTM_network_absolute" : "LSTM (normal)",
        "LSTM_network_diffs" : "LSTM (differential)"
    }

    model_namesEpochs = {
        "LSTM_network_diffs" : "LSTM (50 epochs)",
        "LSTM_network_diffs_200epochs" : "LSTM (200 epochs)",
        "LSTM_network_diffs_300epochs" : "LSTM (300 epochs)",
    }

    model_names2 = {
        "LSTM_network_diffs_200epochs" : "LSTM",
        "weighted_avg_diffs" : "Averaging baseline",
        "spline_diffs" : "Spline baseline"
    }

    model_names3 = {
        "LSTM_network_diffs_200epochs" : "LSTM (all trajectories)",
        "LSTM_network_diffs_trueturn_200epochs" : "LSTM (turns only)",
        "LSTM_network_diffs_trueturnTest_200epochs" : "LSTM (all training, turns test)",
        "weighted_avg_diffs_trueturn" : "Averaging baseline (turns only)"
    }

    model_names4 = {
        "LSTM_network_diffs_200epochs" : "LSTM (all trajectories)",
        "LSTM_network_diffs_phase_Climb_200epochs" : "LSTM (Climb)",
        "LSTM_network_diffs_phase_Cruise_200epochs" : "LSTM (Cruise)",
        "LSTM_network_diffs_phase_Descent_200epochs" : "LSTM (Descent)",
        "LSTM_network_diffs_phase_Level_200epochs" : "LSTM (Level)",
        "LSTM_network_diffs_phaseTest_Climb_200epochs" : "LSTM (all training, Climb)",
        "LSTM_network_diffs_phaseTest_Cruise_200epochs" : "LSTM (all training, Cruise)",
        "LSTM_network_diffs_phaseTest_Descent_200epochs" : "LSTM (all training, Descent)",
        "LSTM_network_diffs_phaseTest_Level_200epochs" : "LSTM (all training, Level)",
        "weighted_avg_diffs_phase_Climb" : "Averaging baseline (Climb)",
        "weighted_avg_diffs_phase_Cruise" : "Averaging baseline (Cruise)",
        "weighted_avg_diffs_phase_Descent" : "Averaging baseline (Descent)",
        "weighted_avg_diffs_phase_Level" : "Averaging baseline (Level)"
    }

    model_names5 = {
        "LSTM_network_diffs_200epochs" : "LSTM",
        "LSTM_network_diffs_5th_prediction_200epochs" : "LSTM (5th prediction)",
        "LSTM_network_diffs_trueturn_200epochs" : "LSTM (turns only)",
        "LSTM_network_diffs_5th_prediction_trueturn_200epochs" : "LSTM (5th prediciton, turns only)"
    }

    model_names6 = {
        "LSTM_network_diffs_200epochs" : "LSTM",
        "LSTM_network_diffs_extra_200epochs" : "LSTM + traj.-wide",
        "LSTM_network_diffs_additional_features_200epochs" : "LSTM + extra",
        "LSTM_network_diffs_extra_additional_features_200epochs" : "LSTM + traj.-wide + extra"
    }


    model_names = model_names6
    rows = []
    for model, pretty_model in model_names.items():
        row = [pretty_model,]
        for nop in (10,):
            for pred_pos in (1,3,5):
                print(model)
                value = data[(data["Model"]==model) & (data["Number of points"]==nop) & (data["Prediction position"]==pred_pos)]["ECEF Difference"]
                value = float(value)
                row.append(round(value,2))
        rows.append(row)

    df = pd.DataFrame(rows, columns = ["\textbf{Model}", "\textbf{Error 1}", "\textbf{Error 3}", "\textbf{Error 5}"])

    print(df.to_latex(index=False, escape=False))
    




