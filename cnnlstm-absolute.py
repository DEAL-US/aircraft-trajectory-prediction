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
        self.max = np.max(data, axis=0)
        self.min = np.min(data, axis=0)
    def scale(self, data):
        # scaled = np.copy(((data-self.min)/(self.max-self.min)-0.5)*2)
        scaled = np.copy((data - self.mean) / self.sd)
        return scaled
    def scale_back(self, data):
        # return np.copy((data/2+0.5)*(self.max-self.min)+self.min)
        return np.copy(data * self.sd + self.mean)


model_name = "CNN-LSTM_network_absolute"
number_points = [6,7,8,9,10]
prediction_position = 0
features_x = [0,1,3]
features_y = [0,1,3]
time_intervals = [30, 60, 90, 120]
training_fraction = 0.5
device = torch.device("cuda")
batch_size = 256
learning_rate = 1e-3
num_epochs = 50
remove_samples_with_0_movement_row = True
num_predictions = 10

if prediction_position>0 and num_predictions>1:
    raise ValueError("Can't make several consecutive non-immediate predictions")

for nop in number_points:

    seq_size = nop

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
        if -999999 in x[:,features_x] or -999999 in y[prediction_position:(prediction_position+num_predictions),features_y]:
            keep_positions.append(False)
        else:
            keep_positions.append(True)

    xs = xs[keep_positions]
    ys = ys[keep_positions]
    ids = ids[keep_positions]

    xs = xs[:,:,features_x]
    ys = ys[:,:,features_y]
    
    # Removal of samples with one vector where all features had no increment
    if remove_samples_with_0_movement_row:
        keep_positions = []
        for x, y in zip(xs, ys):
            if (~(y[prediction_position+1:(prediction_position+num_predictions)] - y[prediction_position:(prediction_position+num_predictions-1)]).any(axis=1)).any() or (~(x[1:]-x[:-1]).any(axis=1)).any():
                keep_positions.append(False)
            else:
                keep_positions.append(True)
        xs = xs[keep_positions]
        ys = ys[keep_positions]
        ids = ids[keep_positions]
        
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

    scaler.fit(xs)

    rnd = np.random.default_rng(1337)
    positions = rnd.permutation(len(xs))
    num_training = int(len(xs)*training_fraction)
    pos_training = positions[:num_training]
    pos_testing = positions[num_training:]

    xs_training = xs[pos_training]
    ys_training = ys[pos_training]
    ids_training = ids[pos_training]
    xs_test = xs[pos_testing]
    ys_test = ys[pos_testing]
    ids_testing = ids[pos_testing]

    xs_training = scaler.scale(xs_training)
    ys_training = scaler.scale(ys_training)
    xs_test = scaler.scale(xs_test)
    ys_test = scaler.scale(ys_test)
    good_positions_training = np.where(ys_training)


    dataset_training = CustomDataset(xs_training, np.copy(ys_training[:,prediction_position,:]))
    loader_training = dataloader.DataLoader(dataset_training, batch_size)

    model = architectures.CNNLSTMNetwork(len(features_x))

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, threshold=1e-4)
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        counter = 0
        avg_train_loss = 0
        for batch in loader_training:
            xs_batch:torch.Tensor = batch["inputs"]
            ys_batch:torch.Tensor = batch["labels"]
            xs_batch = xs_batch.to(device).float()
            ys_batch = ys_batch.to(device).float()
            counter += 1
            model.zero_grad()
            out = model(xs_batch)
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

    def lf(data):
        out = data[1]
        ys = data[2]
        loss = np.mean((out - ys)**2)
        return loss

    model.eval()
    with torch.no_grad():
        
        ys_test_descaled = scaler.scale_back(ys_test)
        xs_test_descaled = scaler.scale_back(xs_test)

        model_predictions = np.zeros((len(xs_test), num_predictions, len(features_y)))
        for i in range(num_predictions):
            out = []
            for j in range(0, len(xs_test), batch_size):
                input = xs_test[j:j+batch_size]
                out.append(np.expand_dims(model(torch.Tensor(input).to(device).float()).cpu().detach().numpy(), 1))
            out = np.concatenate(out)
            xs_test_descaled = scaler.scale_back(xs_test)
            out_descaled = scaler.scale_back(out)

            predictions = list(zip(xs_test_descaled, out_descaled, ys_test_descaled[:,prediction_position+i,:])) 
            model_predictions[:,i,:] = np.squeeze(out_descaled)
            total_loss = 0
            for prediction in predictions:
                total_loss += lf(prediction)

            avg_loss = total_loss / len(predictions)
            print(f"Average loss for prediction {i}: {avg_loss}")
            xs_test = np.concatenate([xs_test[:,1:,:], out], 1)


        results_folder = f"./test-results/{'-'.join([str(ti) for ti in time_intervals])}/{nop}"
        Path(results_folder).mkdir(parents=True, exist_ok=True)
        results = {id:model_predictions[i] for i, id in enumerate(ids_testing)}
        np.save(f"{results_folder}/{model_name}.npy",results)
