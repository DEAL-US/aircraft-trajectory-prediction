import torch
from torch import nn
from scipy.interpolate import UnivariateSpline
import numpy as np

def predict_extrapolate(observations, prediction_position, num_predictions):
    predictions = np.zeros((len(observations), num_predictions, observations.shape[-1]))
    for i, observation in enumerate(observations):
        for feature in range(observations.shape[-1]):
            ys = observation[:,feature]
            xs = np.arange(len(observation))
            spl = UnivariateSpline(xs, ys, k=2)
            out = spl(np.arange(prediction_position, prediction_position+num_predictions))
            predictions[i,:,feature] = out
    return predictions

def predict_average(observations):
    predictions = observations[:,-1,:]*0.5 + observations[:,-2,:]*0.3 + observations[:,-3,:]*0.2
    predictions = np.expand_dims(predictions, axis=1)
    return predictions



def create_dense_block(input_size, output_size, hidden_sizes:list, dropout_rate=0.1):
    layers = []
    input_sizes = [input_size,] + hidden_sizes
    output_sizes = hidden_sizes + [output_size,]

    for iz, oz in zip(input_sizes, output_sizes):
        # layers.append(nn.BatchNorm1d(input_size))
        if(dropout_rate > 0):
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(iz, oz))
        layers.append(nn.Tanh())
    
    return nn.Sequential(*layers)

class DenseNetwork(nn.Module): 
    def __init__(self, number_features, sequence_size):
        super(DenseNetwork, self).__init__()
        self.fc1 = create_dense_block(number_features*sequence_size, 100, [300,200,100], dropout_rate=0)
        self.fc2 = nn.Linear(100, number_features)

    def forward(self, x:torch.Tensor):
        shape = x.shape
        # reshaping to turn sequences of x elements with y features into x*y features
        x = x.reshape((shape[0], -1))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class AttentionNetwork(nn.Module):
    def __init__(self, number_features, sequence_size, encoder_features=80):
        super(AttentionNetwork, self).__init__()
        self.nf = number_features
        self.fc1 = create_dense_block(number_features, encoder_features, [100], dropout_rate=0)
        encoder_layer = nn.TransformerEncoderLayer(encoder_features, nhead=8, batch_first=True, dim_feedforward=80)
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=3)
        # Using the entire sequence from att layer: self.fc2 = create_dense_block(encoder_features*sequence_size, number_features, [64,64,32])
        self.fc2 = nn.Linear(encoder_features, number_features)

    def forward(self, x:torch.Tensor):
        shape = x.shape
        x = x.reshape((-1, self.nf))
        x = self.fc1(x)
        x = x.reshape((shape[0],shape[1],-1))
        x = self.attention(x)
        x = x[:,-1,:]
        # Using the entire sequence from attention layer: x = x.reshape((shape[0],-1))
        x = self.fc2(x)
        return x

class LSTMNetwork(nn.Module):
    def __init__(self, number_features, encoder_features=80):
        super(LSTMNetwork, self).__init__()
        self.nf = number_features
        self.fc1 = create_dense_block(number_features, encoder_features, [100], dropout_rate=0)
        self.lstm = nn.LSTM(input_size=80, hidden_size=80, num_layers=3, batch_first=True)
        self.fc2 = nn.Linear(encoder_features, number_features)

    def forward(self, x:torch.Tensor):
        shape = x.shape
        x = x.reshape((-1, self.nf))
        x = self.fc1(x)
        x = x.reshape((shape[0],shape[1],-1))
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.fc2(x)
        return x

class CNNLSTMNetwork(nn.Module):
    def __init__(self, number_features):
        super(CNNLSTMNetwork, self).__init__()
        self.nf = number_features
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(number_features, 32, 3, padding="same")
        self.max_pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 32, 3, padding="same")
        self.max_pool2 = nn.MaxPool1d(2)
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=50)
        self.drop = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50)
        self.dense = nn.Linear(50, number_features)

    def forward(self, x:torch.Tensor):
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = x.permute(0,2,1)
        x, _ = self.lstm1(x)
        x = self.drop(x)
        x, _ = self.lstm2(x)
        x = x[:,-1,:]
        x = self.drop(x)
        x = self.dense(x)
        return x

class LSTMNetworkExtras(nn.Module):
    def __init__(self, number_features, number_extra_features, encoder_features=80, extra_encoding_features=30, all_encoding_features=30):
        super(LSTMNetworkExtras, self).__init__()
        self.nf = number_features
        self.fc1 = create_dense_block(number_features, encoder_features, [100], dropout_rate=0)
        self.fc_extra1 = create_dense_block(number_extra_features, extra_encoding_features, [100, 75, 50], dropout_rate=0)
        self.lstm = nn.LSTM(input_size=80, hidden_size=80, num_layers=3, batch_first=True)
        self.fc_union = create_dense_block(encoder_features+extra_encoding_features, all_encoding_features, [75, 50])
        self.fc2 = nn.Linear(all_encoding_features, number_features)

    def forward(self, x:torch.Tensor, extras:torch.Tensor):
        shape = x.shape
        x = x.reshape((-1, self.nf))
        x = self.fc1(x)
        x = x.reshape((shape[0],shape[1],-1))
        extras = self.fc_extra1(extras)
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        union = torch.cat((x, extras), axis=1)
        union = self.fc_union(union)
        union = self.fc2(union)
        return union
