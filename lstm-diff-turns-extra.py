import numpy as np
from numpy.core.fromnumeric import reshape
import torch
from torch import nn
from torch.utils.data import dataloader
import architectures
import random
import pandas as pd
from pathlib import Path
import pickle

np.set_printoptions(edgeitems=30, linewidth=100000)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, extras, labels):
        self.inputs = inputs
        self.extras = extras
        self.labels = labels
    def __len__(self):
            return len(self.labels)
    def __getitem__(self, idx):
            labels = self.labels[idx]
            inputs = self.inputs[idx]
            extras = self.extras[idx]
            sample = {"inputs": inputs, "extras": extras, "labels": labels}
            return sample

class Scaler():
    def fit(self, data, is_sequential=True):
        if(is_sequential):
            shape = data.shape
            data = np.reshape(data, newshape=(-1, shape[-1]))
        self.mean = np.mean(data, axis=0)
        self.sd = np.std(data, axis=0)
        self.max = np.max(data, axis=0)
        self.min = np.min(data, axis=0)
    def scale(self, data):
        # scaled = np.copy(((data-self.min)/(self.max-self.min)-0.5)*2)
        scaled = np.copy((data - self.mean) / self.sd)
        return scaled
    def scale_back(self, data):
        # return np.copy((data/2+0.5)*(self.max-self.min)+self.min)
        return np.copy(data * self.sd + self.mean)


model_name = "LSTM_network_diffs_trueturn_extra_200epochs"
number_points = [5,6,7,8,9,10]
prediction_position = 0
features_x = [0,1,3]
features_y = [0,1,3]
f_diff = [0,1,3] 
time_intervals = [30, 60, 90, 120]
training_fraction = 0.5
device = torch.device("cuda")
batch_size = 256
learning_rate = 1e-3
num_epochs = 200
remove_samples_with_0_movement_row = True
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
    all_extras = []
    all_ids = []

    for time_interval in time_intervals:
        folder_path = f"./training-data/training-data-{time_interval}/{nop}"
        xs_path = f"{folder_path}/xs.npy"
        ys_path = f"{folder_path}/ys.npy"
        extras_path = f"{folder_path}/extra.npy"

        xs = np.load(xs_path)
        ys = np.load(ys_path)
        extras = np.load(extras_path)
        extras = np.concatenate((xs[:,0,0:2], extras), axis=1)
        ids = [f"{time_interval}-{nop}-{i}" for i in range(len(xs))]

        all_xs.append(xs)
        all_ys.append(ys)
        all_extras.append(extras)
        all_ids.extend(ids)

    xs:np.ndarray = np.concatenate(all_xs)
    ys:np.ndarray = np.concatenate(all_ys)
    extras:np.ndarray = np.concatenate(all_extras)
    ids = np.array(all_ids)

    # Removal of samples with -999999 (no value) for any of the features, or -1 (used formerly).
    keep_positions = []
    for pos, (x, y) in enumerate(zip(xs, ys)):
        if -999999 in x[:,features_x] or -1 in y[prediction_position:(prediction_position+num_predictions),features_y]:
            keep_positions.append(False)
        else:
            keep_positions.append(True)

    xs = xs[keep_positions]
    ys = ys[keep_positions]
    extras = extras[keep_positions]
    ids = ids[keep_positions]

    if(len(f_diff)>0):
        originals, xs, ys = to_diff(xs, ys)

    xs = xs[:,:,features_x]
    ys = ys[:,:,features_y]
    if(len(f_diff)>0):
        originals = originals[:,:,features_x]

    # Removal of samples with one vector where all features had no increment
    if remove_samples_with_0_movement_row:
        keep_positions = []
        for x, y in zip(xs, ys):
            if (~y[prediction_position:(prediction_position+num_predictions)].any(axis=1)).any() or (~x.any(axis=1)).any():
                keep_positions.append(False)
            else:
                keep_positions.append(True)
        xs = xs[keep_positions]
        ys = ys[keep_positions]
        extras = extras[keep_positions]
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
    extras = extras[keep_positions]
    ids = ids[keep_positions]
    if(len(f_diff)>0):
        originals = originals[keep_positions]

    scaler.fit(xs)
    scaler_extras = Scaler()
    scaler_extras.fit(extras, is_sequential=False)

    # Keeping only cases with turns
    with open(f'./training-data/training-data-turns/{nop}', 'rb') as f:
        turns_ids = set(pickle.load(f))
    keep_positions = []
    for id in ids:
        keep_positions.append(id in turns_ids)
    
    xs = xs[keep_positions]
    ys = ys[keep_positions]
    extras = extras[keep_positions]
    ids = ids[keep_positions]
    
    if(len(f_diff)>0):
        originals = originals[keep_positions]

    rnd = np.random.default_rng(1337)
    positions = rnd.permutation(len(xs))
    num_training = int(len(xs)*training_fraction)
    pos_training = positions[:num_training]
    pos_testing = positions[num_training:]

    xs_training = xs[pos_training]
    ys_training = ys[pos_training]
    extras_training = extras[pos_training]
    ids_training = ids[pos_training]
    xs_test = xs[pos_testing]
    ys_test = ys[pos_testing]
    extras_test = extras[pos_testing]
    ids_testing = ids[pos_testing]

    if(len(f_diff)>0):
        originals_training = originals[pos_training]
        originals_testing = originals[pos_testing]

    xs_training = scaler.scale(xs_training)
    ys_training = scaler.scale(ys_training)
    extras_training = scaler_extras.scale(extras_training)
    xs_test = scaler.scale(xs_test)
    ys_test = scaler.scale(ys_test)
    extras_test = scaler_extras.scale(extras_test)
    good_positions_training = np.where(ys_training)


    dataset_training = CustomDataset(xs_training, extras_training, np.copy(ys_training[:,prediction_position,:]))
    loader_training = dataloader.DataLoader(dataset_training, batch_size)

    model = architectures.LSTMNetworkExtras(len(features_x), len(extras_training[0]))

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, threshold=1e-4)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 75], 0.1)
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        counter = 0
        avg_train_loss = 0
        for batch in loader_training:
            xs_batch:torch.Tensor = batch["inputs"]
            extras_batch:torch.Tensor = batch["extras"]
            ys_batch:torch.Tensor = batch["labels"]
            xs_batch = xs_batch.to(device).float()
            ys_batch = ys_batch.to(device).float()
            extras_batch = extras_batch.to(device).float()
            counter += 1
            model.zero_grad()
            out = model(xs_batch, extras_batch)
            #loss = loss_function(out, ys_batch[:,prediction_position])
            loss = loss_function(out, ys_batch)
            if(loss.item() > 5):
                print("ruh roh")
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item() / len(loader_training)
            if counter%100 == 0:
                print(f"======== Epoch {epoch}......Step: {counter}/{len(loader_training)}....... Loss: {loss.item()} =========")
                """ print("x: ")
                print(xs_batch[0])
                print("prediction: ")
                print(out[0])
                print("y: ")
                print(ys_batch[0]) """
                
        print(f"AVG LOSS: {avg_train_loss}")
        scheduler.step(metrics=avg_train_loss)
        # scheduler.step()
    def lf(data):
        out = data[1]
        ys = data[2]
        loss = np.mean((out - ys)**2)
        return loss

    model.eval()
    with torch.no_grad():
        
        ys_test_descaled = scaler.scale_back(ys_test)
        xs_test_descaled = scaler.scale_back(xs_test)

        if(len(f_diff)>0):
            xs_test_descaled_dediff, ys_test_descaled_dediff = from_diff(originals_testing, xs_test_descaled, ys_test_descaled)


        model_predictions = np.zeros((len(xs_test), num_predictions, len(features_y)))
        for i in range(num_predictions):
            out = []
            for j in range(0, len(xs_test), batch_size):
                input = xs_test[j:j+batch_size]
                input =  torch.Tensor(input).to(device).float()
                extra = extras_test[j:j+batch_size]
                extra =  torch.Tensor(extra).to(device).float()
                out.append(np.expand_dims(model(input, extra).cpu().detach().numpy(), 1))
            out = np.concatenate(out)
            xs_test_descaled = scaler.scale_back(xs_test)
            out_descaled = scaler.scale_back(out)

            if(len(f_diff)>0):
                xs_test_descaled_dediff, out_descaled_dediff = from_diff(originals_testing, xs_test_descaled, out_descaled)
            predictions = list(zip(xs_test_descaled_dediff, out_descaled_dediff, ys_test_descaled_dediff[:,prediction_position+i,:])) 
            model_predictions[:,i,:] = np.squeeze(out_descaled_dediff)
            total_loss = 0
            for prediction in predictions:
                total_loss += lf(prediction)

            avg_loss = total_loss / len(predictions)
            print(f"Average loss for prediction {i}: {avg_loss}")
            xs_test = np.concatenate([xs_test[:,1:,:], out], 1)
            originals_testing = np.expand_dims(xs_test_descaled_dediff[:,1,:], 1)


        results_folder = f"./test-results/{'-'.join([str(ti) for ti in time_intervals])}/{nop}"
        Path(results_folder).mkdir(parents=True, exist_ok=True)
        results = {id:model_predictions[i] for i, id in enumerate(ids_testing)}
        np.save(f"{results_folder}/{model_name}.npy",results)
