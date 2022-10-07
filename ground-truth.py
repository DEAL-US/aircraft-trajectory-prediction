import numpy as np
from numpy.core.fromnumeric import reshape
import torch
from torch._C import Value
from tqdm import tqdm
from sklearn import preprocessing
from torch import nn
from torch.utils.data import dataloader
import architectures
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

np.set_printoptions(edgeitems=30, linewidth=100000)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
    def __len__(self):
            return len(self.labels)
    def __getitem__(self, idx):
            labels = self.labels[idx]
            inputs = self.inputs[idx]
            sample = {"inputs": inputs, "labels": labels}
            return sample

class Scaler():
    def fit(self, data):
        shape = data.shape
        data = np.reshape(data, newshape=(-1, shape[-1]))
        self.mean = np.mean(data, axis=0)
        self.sd = np.std(data, axis=0)
    def scale(self, data):
        return np.copy((data - self.mean) / self.sd)
    def scale_back(self, data):
        return np.copy(data * self.sd + self.mean)


model_name = "ground_truth"
number_points = [5,6,7,8,9,10]
prediction_position = 0
features_x = [0,1,3]
features_y = [0,1,3]
f_diff = [0,1,3] 
time_intervals = [30, 60, 90, 120]
remove_all_zero_ys = True
num_predictions = 10

if prediction_position>0 and num_predictions>1:
    raise ValueError("Can't make several consecutive non-immediate predictions")

for nop in number_points:

    seq_size = nop
    if(len(f_diff)>0):
        seq_size -= 1

    all_xs = []
    all_ys = []
    all_ids = []

    for time_interval in time_intervals:
        folder_path = f"./training-data/training-data-{time_interval}/{nop}"
        xs_path = f"{folder_path}/xs.npy"
        ys_path = f"{folder_path}/ys.npy"

        xs = np.load(xs_path)
        ys = np.load(ys_path)
        ids = [f"{time_interval}-{nop}-{i}" for i in range(len(xs))]

        all_xs.append(xs)
        all_ys.append(ys)
        all_ids.extend(ids)

    xs:np.ndarray = np.concatenate(all_xs)
    ys:np.ndarray = np.concatenate(all_ys)
    ids = np.array(all_ids)
    
    xs = xs[:,:,features_x]
    ys = ys[:,:,features_y]

    model_predictions = ys[:,prediction_position:prediction_position+num_predictions,:]

    results_folder = f"./test-results/{'-'.join([str(ti) for ti in time_intervals])}/{nop}"
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    results = {id:model_predictions[i] for i, id in enumerate(ids)}
    inputs = {id:xs[i] for i, id in enumerate(ids)}
    np.save(f"{results_folder}/{model_name}.npy",results)
    np.save(f"{results_folder}/xs.npy",inputs)