
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
    dfs = []

    for nop_path in tqdm(results_path.iterdir()):
        nop = nop_path.name
        summaries_path = pathlib.Path(f'{nop_path.absolute()}/results_summary')
        print(summaries_path)
        for tech_path in summaries_path.iterdir():
            df = pd.read_csv(tech_path.absolute(), sep=';')
            dfs.append(df)
    full_df = pd.concat(dfs)
    full_df.to_csv('./test-results/results_summary.csv', sep=';')
    

