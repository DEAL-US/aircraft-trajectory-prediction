
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

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)
    
def add_results_as_rows(gts_per_case, inputs_per_case, results_per_case, res=[]):
    for case, results_per_tech in tqdm(results_per_case.items(), total=len(results_per_case)):
        time_interval, nop, _ = case.split('-')
        if case not in gts_per_case:
            print(results_per_tech.keys())
        gts = gts_per_case[case]
        for tech, results in results_per_tech.items():
            for pred_position, pred in enumerate(results):
                gt = gts[pred_position]
                difference = np.abs(gt[0:3] - pred[0:3])
                gt_ecef = np.array(lla_to_ecef(gt[0], gt[1], gt[2]))
                pred_ecef = np.array(lla_to_ecef(pred[0], pred[1], pred[2]))
                difference_ecef = np.linalg.norm(gt_ecef - pred_ecef)
                row = [case, pred_position, nop, tech, *difference, difference_ecef]
                res.append(row)
    return res

def draw_results(results_path:str, xs:np.ndarray, gt:np.ndarray, results:dict[str,np.ndarray], xs_color:str="grey", gt_color:str="black"):
    num_features = xs.shape[-1]
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 1, len(results))]
    results_dfs = {}
    if(num_features == 2):
        columns = ["latitude", "longitude"]
    else:
        columns = ["latitude", "longitude", "baro_altitude"]
    xs = pd.DataFrame(xs, columns=columns)
    gt = pd.DataFrame(gt, columns=columns)
    for tech, result in results.items():
        results_dfs[tech] = pd.DataFrame(result, columns=columns)

    concat_dfs = pd.concat((gt, xs, *results_dfs.values()))
    if(num_features == 2):
        max_lat, max_lon = concat_dfs.max()
        min_lat, min_lon = concat_dfs.min()
    if(num_features == 3):
        max_lat, max_lon, _ = concat_dfs.max()
        min_lat, min_lon, _ = concat_dfs.min()

    bbox = (min_lon, max_lon, min_lat, max_lat)

    plt.scatter(xs["longitude"], xs["latitude"], color=xs_color)
    plt.scatter(gt["longitude"], gt["latitude"], color=gt_color)
    for i, (tech, df) in enumerate(results_dfs.items()):
        plt.scatter(df["longitude"], df["latitude"], color=colors[i])

    plt.savefig(results_path)
    plt.clf()

RESULTS_FOLDER = '30-60-90-120'

results_path = pathlib.Path(f'./test-results/{RESULTS_FOLDER}')

def compute_metrics(nop_path, tech_path):
    nop = nop_path.name
    tech_name = tech_path.name.split('.')[0]
    summary_path = pathlib.Path(f'{nop_path.absolute()}/results_summary')
    summary_path.mkdir(parents=True, exist_ok=True)
    results = {}
    ground_truth = np.load(f'{nop_path.absolute()}/ground_truth.npy', allow_pickle=True).item()
    inputs = np.load(f'{nop_path.absolute()}/xs.npy', allow_pickle=True).item()
    # Loop used to store the results of each case for each model
    
    tech_results = np.load(tech_path.absolute(), allow_pickle=True).item()
    for case, predictions in tech_results.items():
        if(case not in results): 
            results[case] = {}
        results[case][tech_name] = predictions
    results_df = add_results_as_rows(ground_truth, inputs, results)
    col_names = ["case", "pred_position", "NoP", "model", "lat_difference", "lon_difference", "alt_difference", "ecef_difference"]
    results_df = pd.DataFrame(results_df, columns=col_names)
    results_df['difference_mean'] = results_df.loc[:,["lat_difference", "lon_difference", "alt_difference"]].mean(axis=1)
    results_df['latlon_difference_mean'] = results_df.loc[:,["lat_difference", "lon_difference"]].mean(axis=1)
    results_df = results_df.groupby("pred_position")
    rows = []
    columns = ["Model", "Number of points", "Prediction position", "Latitude difference", "Longitude difference", "Altitude difference", "LatLon difference", "Difference", "ECEF Difference"]

    for j in range(10):
        group = results_df.get_group(j)
        latlon_difference_mean = group["latlon_difference_mean"].mean()
        difference_mean = group["difference_mean"].mean()
        lat_difference_mean = group["lat_difference"].mean()
        lon_difference_mean = group["lon_difference"].mean()
        alt_difference_mean = group["alt_difference"].mean()
        ecef_difference_mean = group["ecef_difference"].mean()
        rows.append([tech_name, nop, j, lat_difference_mean, lon_difference_mean, alt_difference_mean, latlon_difference_mean, difference_mean, ecef_difference_mean])
        del group
    summary_df = pd.DataFrame(rows, columns=columns)
    summary_df.to_csv(f'{summary_path.absolute()}/{tech_name}.csv', sep=';')

if __name__ == '__main__':
    for nop_path in tqdm(results_path.iterdir()):
        nop = nop_path.name
        for tech_path in nop_path.iterdir():
            tech_name = tech_path.name.split('.')[0]
            if tech_name not in ("ground_truth", "xs", "results", "results_summary") and "LSTM_network_diffs_phaseTest_Cruise_200epochs" in tech_name:
                proc = mp.Process(target=compute_metrics, args=(nop_path,tech_path))
                proc.start()
                proc.join()

