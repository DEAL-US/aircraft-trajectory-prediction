import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import termplotlib as tpl

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

def to_diff(xs:np.ndarray, ys:np.ndarray, f_diff, return_diffs_only:bool=False):
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

def compute_flight_sd(flight:np.ndarray, absolute_input:bool=True, feat_indices:list=[], compute_mean=True):
    if len(feat_indices)==0:
        feat_indices = range(flight.shape[-1])
    flight = flight[:,feat_indices]
    if absolute_input:
        flight = np.diff(flight, axis=0)
    sds = np.std(flight, axis=0)
    if compute_mean:
        return np.mean(sds)
    else:
        return sds

def compute_flight_accumulated_second_differential(flight:np.ndarray, absolute_input:bool=True, feat_indices:list=[], compute_mean=True, normalise=True):
    if len(feat_indices)==0:
        feat_indices = range(flight.shape[-1])
    flight = flight[:,feat_indices]
    if absolute_input:
        flight = np.diff(flight, axis=0)
    if normalise:
        flight = flight / abs(flight).max()
    flight = np.diff(flight, axis=0)
    acc = abs(flight).sum(axis=0)
    if compute_mean:
        return np.mean(acc)
    else:
        return acc

def angle_between(v1, v2):
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

def compute_flight_accumulated_rotation(flight:np.ndarray, absolute_input:bool=True, feat_indices:list=[]):
    if len(feat_indices)==0:
        feat_indices = range(flight.shape[-1])
    flight = flight[:,feat_indices]
    if absolute_input:
        flight = np.diff(flight, axis=0)
    flight = flight / np.expand_dims(np.linalg.norm(flight, axis=1), 1)
    acc = 0
    for i in range(len(flight)-1):
        angle = angle_between(flight[i], flight[i+1])   
        # Check to not to include anomalous 180 degrees turns
        if angle < 3:
            acc += angle
    return acc

def draw_console_flight(lats:np.ndarray, lons:np.ndarray, width=60, height=25):
    fig = tpl.figure()
    fig.plot(lons, lats, width=width, height=height)
    fig.show()


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

def find_positions_without_absent_values(xs, ys, features=[]):
    if len(features)==0:
        features = range(xs.shape[-1])
    keep_positions = []
    for pos, (x, y) in enumerate(zip(xs, ys)):
        if -1 in x[:,features] or -1 in y[:,features] or -999999 in x[:,features] or -999999 in y[:,features] :
            keep_positions.append(False)
        else:
            keep_positions.append(True)
    return keep_positions

def find_positions_without_redoudant_vectors(xs, ys, features=[], compute_diffs=True):
    if len(features)==0:
        features = range(xs.shape[-1])
    if compute_diffs:
        _, xs, ys = to_diff(xs, ys, features, True, ) 
    keep_positions = []
    for x, y in zip(xs, ys):
        if (~y.any(axis=1)).any() or (~x.any(axis=1)).any():
            keep_positions.append(False)
        else:
            keep_positions.append(True)
    return keep_positions

def find_positions_without_anomalous_values(xs, ys, features=[], ratio_threshold=2):
    if len(features)==0:
        features = range(xs.shape[-1])
    xs = xs[:,:,features]
    ys = ys[:,:,features]
    concat_data = np.concatenate((xs, ys), axis=1)
    concat_data = np.reshape(concat_data, (-1, concat_data.shape[-1]))
    sd = np.std(concat_data, axis=0)
    mean = np.mean(concat_data, axis=0)
    keep_positions = []
    for x, y in zip(xs, ys):
        x_ratios = abs((x-mean)/sd)
        y_ratios = abs((y-mean)/sd)
        keep_positions.append(not ((x_ratios > 2).any() or (y_ratios > 2).any()))
    return keep_positions

def lla_to_ecef(lat, lon, alt, input_in_radians=False):
    if(not input_in_radians):
        lat = lat*2*np.pi/360
        lon = lon*2*np.pi/360
    rad = np.float64(6378137.0)        # Radius of the Earth (in meters)
    f = np.float64(1.0/298.257223563)  # Flattening factor WGS84 Model
    np.cosLat = np.cos(lat)
    np.sinLat = np.sin(lat)
    FF     = (1.0-f)**2
    C      = 1/np.sqrt(np.cosLat**2 + FF * np.sinLat**2)
    S      = C * FF

    x = (rad * C + alt)*np.cosLat * np.cos(lon)
    y = (rad * C + alt)*np.cosLat * np.sin(lon)
    z = (rad * S + alt)*np.sinLat
    return x, y, z
