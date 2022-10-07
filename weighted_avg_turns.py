import numpy as np
from numpy.core.fromnumeric import reshape
import torch
import architectures
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

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


model_name = "weighted_avg_diffs_trueturn"
number_points = [5,6,7,8,9,10]
prediction_position = 0
features_x = [0,1,3]
features_y = [0,1,3]
f_diff = [0,1,3] 
time_intervals = [30, 60, 90, 120]
training_fraction = 0.5
remove_all_zero_ys = True
num_predictions = 10

if prediction_position>0 and num_predictions>1:
    raise ValueError("Can't make several consecutive non-immediate predictions")


def to_diff(xs:np.ndarray, ys:np.ndarray, return_diffs_only:bool=False):
    xs = np.copy(xs)
    ys = np.copy(ys)
    originals = np.expand_dims(xs[:,0,:], 1)
    if ys is not None:
        ys[:,1:,f_diff] = ys[:,1:,f_diff] - ys[:,:-1:,f_diff]
        ys[:,0,f_diff] = ys[:,0,f_diff] - xs[:,-1,f_diff]
    xs[:,1:,f_diff] = xs[:,1:,f_diff] - xs[:,:-1,f_diff]
    xs = xs[:,1:,:]

    if return_diffs_only:
        xs = xs[:,:,f_diff]
        ys = ys[:,:,f_diff]
        originals = originals[:,:,f_diff]

    return (originals, xs, ys) 

def from_diff(originals, xs, ys, diff_indices:list=[]):
    xs = np.copy(np.concatenate([originals, xs], 1))
    if(len(diff_indices)==0):
        xs[:,1:,:] = xs.cumsum(1)[:,1:,:]
    else:
        xs[:,1:,diff_indices] = xs.cumsum(1)[:,1:,diff_indices]
    if ys is not None:
        ys = np.copy(np.concatenate([np.expand_dims(xs[:,-1,:], 1), ys], 1))
        if(len(diff_indices) == 0):
            ys[:,1:,:] = ys.cumsum(1)[:,1:,:]
        else:
            ys[:,1:,diff_indices] = ys.cumsum(1)[:,1:,diff_indices]
        ys = ys[:,1:,:]
    return (xs, ys)

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


    # Removal of samples with -999999 (no value) for any of the features.
    keep_positions = []
    for pos, (x, y) in enumerate(zip(xs, ys)):
        if -999999 in x[:,features_x] or -999999 in y[prediction_position:prediction_position+num_predictions,features_y]:
            keep_positions.append(False)
        else:
            keep_positions.append(True)

    xs = xs[keep_positions]
    ys = ys[keep_positions]
    ids = ids[keep_positions]

    if(len(f_diff)>0):
        originals, xs, ys = to_diff(xs, ys)

    xs = xs[:,:,features_x]
    ys = ys[:,:,features_y]
    if(len(f_diff)>0):
        originals = originals[:,:,features_x]


    # Removal of samples with all features equal to 0
    if remove_all_zero_ys:
        keep_positions = []
        for x, y in zip(xs, ys):
            if (~y[prediction_position:(prediction_position+num_predictions)].any(axis=1)).any() or (~x.any(axis=1)).any():
                keep_positions.append(False)
            else:
                keep_positions.append(True)
        xs = xs[keep_positions]
        ys = ys[keep_positions]
        ids = ids[keep_positions]
        if(len(f_diff)>0):
            originals = originals[keep_positions]
        
    scaler = Scaler()
    scaler.fit(xs)

    # Removing anomalous values
    keep_positions = []
    for x, y in zip(xs, ys):
        x_ratios = abs((x-scaler.mean)/scaler.sd)
        y_ratios = abs((y[prediction_position:(prediction_position+num_predictions)]-scaler.mean)/scaler.sd)
        keep_positions.append(not ((x_ratios > 2).any() or (y_ratios > 2).any()))

    xs = xs[keep_positions]
    ys = ys[keep_positions]
    ids = ids[keep_positions]

    if(len(f_diff)>0):
        originals = originals[keep_positions]

    # Keeping only cases with turns
    with open(f'./data-with-extra-phases/training-data-turns/{nop}', 'rb') as f:
        turns_ids = set(pickle.load(f))
    keep_positions = []
    for id in ids:
        keep_positions.append(id in turns_ids)
    
    xs = xs[keep_positions]
    ys = ys[keep_positions]
    ids = ids[keep_positions]
    
    if(len(f_diff)>0):
        originals = originals[keep_positions]


    rnd = np.random.default_rng(1337)
    positions = rnd.permutation(len(xs))
    num_training = int(len(xs)*training_fraction)
    pos_testing = positions[num_training:]

    xs_test = xs[pos_testing]
    ys_test = ys[pos_testing]
    ids_testing = ids[pos_testing]

    if(len(f_diff)>0):
        originals_testing = originals[pos_testing]


    if(len(f_diff)>0):
        xs_test_dediff, ys_test_dediff = from_diff(originals_testing, xs_test, ys_test)

    def lf(data):
        out = data[1][0,[0,1]]
        ys = data[2][[0,1]]
        loss = np.mean(abs((out - ys)))
        return loss

    model_predictions = np.zeros((len(xs_test), num_predictions, len(features_y)))
    for i in range(num_predictions):
        out = architectures.predict_average(xs_test)
        if(len(f_diff)>0):
            xs_test_dediff, out_dediff = from_diff(originals_testing, xs_test, out)
        predictions = list(zip(xs_test_dediff, out_dediff, ys_test_dediff[:,prediction_position+i,:])) 
        model_predictions[:,i,:] = np.squeeze(out_dediff)
        total_loss = 0
        for prediction in predictions:
            total_loss += lf(prediction)

        avg_loss = total_loss / len(predictions)
        print(f"Average latlon loss for prediction {i}: {avg_loss}")
        xs_test = np.concatenate([xs_test[:,1:,:], out], 1)
        originals_testing = np.expand_dims(xs_test_dediff[:,1,:], 1)


    results_folder = f"./test-results/{'-'.join([str(ti) for ti in time_intervals])}/{nop}"
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    results = {id:model_predictions[i] for i, id in enumerate(ids_testing)}
    np.save(f"{results_folder}/{model_name}.npy",results)