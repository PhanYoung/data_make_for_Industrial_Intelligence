import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.interpolate import interp1d
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr

import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from torch import optim

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import random
random.seed(715)



class Network(object):
    '''
        general nueral network framework
    '''
    def __init__(self, mod, op_method=optim.Adam, loss_method=nn.MSELoss,
                 device='cpu', solo_output=False, with_scaler=True,
                 predict_trans=None, squeeze_1st_dim=False, **optim_params):
        self.device = device
        self.optimizer = op_method(mod.parameters(), **optim_params)
        self.mod = mod.to(device)
        self.loss_func = loss_method().to(device)
        self.reshape_y = solo_output
        self.predict_trans = predict_trans
        self.with_scaler = with_scaler
        if with_scaler:
            self.in_scaler = StandardScaler()
            self.out_scaler = StandardScaler()
        self.squeeze_1st_dim = squeeze_1st_dim


    def fit(self, x, y, batch_size=16, max_epoch=1000, gossip=False):
        from torch.nn.init import xavier_uniform
        self.mod.train()
        self.optimizer.zero_grad() #梯度清零，不叠加
        if self.reshape_y:
            y = y.reshape(-1, 1)
        if self.with_scaler:
            x = self.in_scaler.fit_transform(x)
            y = self.out_scaler.fit_transform(y)

        for epoch in range(max_epoch):
            for batch_x, batch_y in self.batch_iter(x, y, batch_size):
                if self.squeeze_1st_dim:
                    batch_x, batch_y = batch_x[0], batch_y[0]
                self.optimizer.zero_grad() #梯度清零，不叠加
                pred_y = self.mod(batch_x)
                loss = self.loss_func(pred_y, batch_y)
                loss.backward()
                self.optimizer.step()
            if gossip and epoch % 100 == 0:
                print("epoch-", epoch, " : ", loss)


    def predict(self, x):
        self.mod.eval()
        x = self.in_scaler.transform(x)
        x = t.Tensor(x).to(self.device)
        y_ = self.mod(x)
        y_ = y_.detach().cpu().numpy()

        if self.reshape_y:
            y_ = y_.squeeze()
        if self.out_scaler:
            y_ = self.out_scaler.inverse_transform(y_)
        if self.predict_trans:
            y_ = self.predict_trans(y_)

        return y_


    def batch_iter(self, x, y, batch_size):
        x = t.Tensor(x).to(self.device)
        y = t.Tensor(y).to(self.device)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size)



class LstmNetwork(Network):
    '''Framework for LSTM
    '''
    def __init__(self, mod, op_method=optim.Adam, loss_method=nn.MSELoss,
            device='cpu', solo_output=False, with_scaler=True, seq_len=50,
            predict_trans=None, **optim_params):
        super(LstmNetwork, self).__init__(mod, op_method, loss_method, device, solo_output, with_scaler,
                                          squeeze_1st_dim=True, **optim_params)
        self.seq_len = seq_len


    def roll_tile_array(self, a, length, batch_size):
        return np.array([[a[i+j:i+j+length] for j in range(batch_size)]
                         for i in range(0, len(a) + 1 - length - batch_size, length)])


    def batch_iter(self, x, y, batch_size):
        newx = t.Tensor(self.roll_tile_array(x, self.seq_len, batch_size)).to(self.device)
        newy = t.Tensor(self.roll_tile_array(y, self.seq_len, batch_size)).to(self.device)
        return DataLoader(TensorDataset(newx, newy))
    
    
    def fit(self, x, y, batch_size=16, max_epoch=1000, gossip=False):
        from torch.nn.init import xavier_uniform
        self.mod.train()
        self.optimizer.zero_grad() #梯度清零，不叠加
        if self.reshape_y:
            y = y.reshape(-1, 1)
        if self.with_scaler:
            x = self.in_scaler.fit_transform(x)
            y = self.out_scaler.fit_transform(y)

        hidden = None
        i = 0
        for epoch in range(max_epoch):
            print("epoch:", i)
            i += 1
            for batch_x, batch_y in self.batch_iter(x, y, batch_size):
                if self.squeeze_1st_dim:
                    batch_x, batch_y = batch_x[0], batch_y[0]
                self.optimizer.zero_grad() #梯度清零，不叠加
                pred_y, hidden = self.mod(batch_x, hidden)
                loss = self.loss_func(pred_y, batch_y)
                loss.backward(retain_graph=True)
                self.optimizer.step()
            if gossip and epoch % 100 == 0:
                print("epoch-", epoch, " : ", loss)


    def predict(self, x):
        self.mod.eval()
        x = self.in_scaler.transform(x)
        x = t.Tensor(x).to(self.device)
        y_ = self.mod(x, None)
        y_ = y_.detach().cpu().numpy()

        if self.reshape_y:
            y_ = y_.squeeze()
        if self.out_scaler:
            y_ = self.out_scaler.inverse_transform(y_)
        if self.predict_trans:
            y_ = self.predict_trans(y_)

        return y_



class FcLinear(nn.Module):
    def __init__(self, sz_input, sz_output, sz_hidden):
        super(FcLinear, self).__init__()
        self.fc1 = nn.Linear(sz_input, sz_hidden)
        self.fc2 = nn.Linear(sz_hidden, sz_output)

    def forward(self, input_ftrs):
        _ = self.fc1(input_ftrs)
        output = self.fc2(_)
        return output



from itertools import chain

class FcBlock(nn.Module):
    def __init__(self, sz_input, sz_output, sz_hiddens, drop_rates=None):
        super(FcBlock, self).__init__()
        fc_list = [nn.Linear(sz_in, sz_out) \
                   for sz_in, sz_out in zip([sz_input] + sz_hiddens,
                                            sz_hiddens + [sz_output])]

        relu_list = [nn.ReLU() for i in range(len(fc_list)+1)]

        if drop_rates:
            drop_list = [nn.Dropout(rt_drop) for rt_drop in drop_rates]
            layer_chain = chain(*zip(fc_list[:-1], drop_list, relu_list))
        else:
            layer_chain = chain(*zip(fc_list[:-1], relu_list))

        layers = list(layer_chain) + [fc_list[-1]]
        self.seq = nn.Sequential(*layers)


    def forward(self, input_ftrs):
        return self.seq(input_ftrs)


class LstmBlock(nn.Module):
    def __init__(self, sz_input, sz_output, sz_hidden1, sz_hidden2, sz_lstm_in, sz_lstm_out, layers=1, drop_rates=None):
        super(LstmBlock, self).__init__()
        self.fc1 = FcBlock(sz_input, sz_lstm_in, sz_hidden1)
        self.fc2 = FcBlock(sz_lstm_out, sz_output, sz_hidden2)
        self.rnn = nn.LSTM(input_size=sz_lstm_in, hidden_size=sz_lstm_out, num_layers=layers, batch_first=True)


    def forward(self, input_ftrs, hidden):
        #input_ftrs: batch_size * seq_size * ftrs_szie
        #b, q, f = input_ftrs.size()
        #x = input_ftrs.view(b*q, f)
        x = self.fc1(input_ftrs)
        #x = x.view(b, q, -1)
        x, h = self.rnn(x, hidden)
        x = self.fc2(x)
        return x, h




class MultiTaskBlock(nn.Module):
    def __init__(self, sz_input, sz_output):
        super(MultiTaskBlock, self).__init__()
        self.in_block = FcBlock(sz_input, 12, [12])
        self.to_obs = FcBlock(12, 2, [6])
        self.to_result = FcBlock(12, 1, [8])

    def forward(self, input_ftrs):
        hid_1 = self.in_block(input_ftrs)
        obs = self.to_obs(hid_1)
        rslt = self.to_result(hid_1)
        return obs, rslt



from .evaluate import evaluate

def lr_test(x_train, y_train, x_test, y_test):
    l = LinearRegression().fit(x_train, y_train)
    y_pred = l.predict(x_test)
    evaluate(y_test, y_pred)
    return l


def nn_test(x_train, y_train, x_test, y_test, device='cuda:2', sz_hiddens=[12, 12, 12],
            batch_size=128, max_epoch=500):
    sz_input = x_train.shape[-1]
    if len(y_train.shape) == 1:
        solo_output = True
        sz_output = 1
    else:
        sz_output = y_train.shape[-1]

    n = Network(mod=FcBlock(sz_input=sz_input, sz_output=sz_output, sz_hiddens=sz_hiddens), 
                            device=device, solo_output=solo_output)
    n.fit(x_train, y_train, batch_size=batch_size, max_epoch=max_epoch)

    y_pred = n.predict(x_test)
    evaluate(y_test, y_pred)
    return n





