import os
import time

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from dataset.scaler import DefinedMinMaxScaler

import warnings
warnings.filterwarnings('ignore')


class TimeSeriesListDataset(object):
    """
    Time Series Dataset Class
    """
    def __init__(self, data_path: str, status: str, lag: int, 
                 output_dim: int, target: str, weat_feat=[]):
        """
        Args:
            data_path (str): Data path of training and test data.
            status (str): Status of engine.
                          Choices: ['train', 'validate', 'test']
            lag (int): Number of previous features used for training
            output_dim (int): Time of prediction output.
            data_list (list, optional): System id list to construct as training dataset.
                                          Defaults to None.
            weat_feat (list, optional): Weather feature that append to input time-series features.
                                        Defaults to [].
        """
        self.data_path = data_path
        self.lag = lag
        self.target = target
        self.output_dim = output_dim
        self.status = status
        self.weat_feat = weat_feat

        self.__dataX, self.__dataY = self._preprocess()

    def _preprocess(self):
        data = pd.read_csv(self.data_path)
        for feat in self.weat_feat:
            scaler = MinMaxScaler()
            data.loc[:, feat] = scaler.fit_transform(data.loc[:, feat].values.reshape(-1, 1))

        data.loc[:, self.target] = self.scaler.transform(data.loc[:, self.target])

        trainX, trainY = self._make_dataset(data, self.lag, self.output_dim)
        trainX = np.array(trainX)
        trainY = np.array(trainY)
        print("dataX shape: ", trainX.shape)
        print("dataY shape: ", trainY.shape)
        return trainX, trainY

    def _make_dataset(self, df, lag, outdim):
        dataX, dataY = [], []
        for i in range(0, len(df)-lag-outdim, outdim):
            tmp = []
            tmp.extend(df.loc[i:i+lag-1, self.target].values)
            for f in self.weat_feat:
                tmp.extend(df.loc[i:i+lag+outdim-1, f].values)
            dataX.append(tmp)
            dataY.append(df.loc[i+lag:i+lag+outdim-1, self.target])
        return dataX, dataY

    @property
    def scaler(self):
        return DefinedMinMaxScaler(min_num=0, max_num=5000)

    @property
    def data(self):
        return self.__dataX, self.__dataY

class TimeSeriesDataset(Dataset):
    def __init__(self, data_path: str, status: str, lag: int,
                 output_dim: int, target: str, weat_feat=[]):
        """
        Args:
            data_path (str): Data path of training and test data.
            status (str): Status of engine.
                          Choices: ['train', 'validate', 'test']
            lag (int): Number of previous features used for training
            output_dim (int): Time of prediction output.
            data_list (list, optional): System id list to construct as training dataset.
                                          Defaults to None.
            weat_feat (list, optional): Weather feature that append to input time-series features.
                                        Defaults to [].
        """
        self.dataset = TimeSeriesListDataset(
            data_path=data_path,
            status=status,
            lag=lag,
            output_dim=output_dim,
            target=target,
            weat_feat=weat_feat
        )

        self.dataX, self.dataY = self.dataset.data
        self.split_feat = lag
        self.feat_dim = len(weat_feat)

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, idx):
        ts_data = torch.FloatTensor(
            self.dataX[idx, :self.split_feat]
        ).unsqueeze(-1)
        add_data = torch.FloatTensor(
                self.dataX[idx, self.split_feat:].reshape(-1, self.feat_dim)
        )
        y_data = torch.FloatTensor(self.dataY[idx])
        return ts_data, add_data, y_data

    @property
    def scaler(self):
        return self.dataset.scaler

    @classmethod
    def constructor(cls, data_path, status, lag, output_dim, target, weat_feat=[]):
        return TimeSeriesDataset(
                data_path=data_path, status=status, lag=lag,
                output_dim=output_dim, target=target, weat_feat=weat_feat)