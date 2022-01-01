import os
import time

import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from dataset.scaler import DefinedMinMaxScaler

import warnings
warnings.filterwarnings('ignore')

to_date = lambda date: datetime.strptime(str(date), "%Y%m%d")

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
        self.one_day = 13

        self.__dataX, self.__dataY = self._preprocess()

    def _make_continuous_data(self, df, one_day):
        # make time list
        hhmm_list = []
        for h in range(6, 19):
            time_ = time(hour=h).isoformat(timespec='minutes')
            f_time = int("".join(time_.split(":")))
            hhmm_list.append(f_time)

        new_row_list = []
        index_list = list(df.index)
        for i in range(0, len(df)-one_day, one_day):
            d = index_list[i]
            cur = to_date(df.loc[d, 'yyyymmdd'])
            nxt = to_date(df.loc[d+one_day, 'yyyymmdd'])

            while nxt > cur + timedelta(days=1):
                for h in hhmm_list:
                    str_d = str(cur + timedelta(days=1)).split(" ")[0]
                    cur_date = str_d + " " + str(h)
                    int_d = int("".join(str_d.split("-")))
                    row_data = {'yyyymmdd': int_d, 'hhmm': h}

                    for c in df.columns:
                        if c not in row_data.keys():
                            row_data[c] = np.nan
                    new_row_list.append(row_data)
                cur = cur + timedelta(days=1)

        df = df.append(new_row_list, ignore_index=True)
        df = df.sort_values(by=['yyyymmdd', 'hhmm'])
        df.index = np.arange(0, len(df))
        return df

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

    # Sliding_window_view
    def _sliding_window_view(self, arr, window_shape, steps):
        """
        sliding window using different size of window and step
        Reference code:
            https://gist.github.com/meowklaski/4bda7c86c6168f3557657d5fb0b5395a
        """ 
        in_shape = np.array(arr.shape[-len(steps):])
        window_shape = np.array(window_shape)
        steps = np.array(steps)
        nbytes = arr.strides[-1]

        window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
        step_strides = tuple(window_strides[-len(steps):] * steps) 
        strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

        outshape = tuple((in_shape - window_shape) // steps + 1) 
        outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape) 
        return np.lib.stride_tricks.as_strided(arr, shape=outshape, strides=strides, writeable=False)

    def _make_dataset_vectorized(self, df, window, one_day, target_col = 'Active_Power', weat_feat=[]):
        size_of_arr = (len(df) - window) // one_day
        dataX, dataY = np.zeros((size_of_arr, 0)), np.zeros((0, one_day))

        # Previous target column feature
        if window != 0:
            series = df[target_col].values[:-one_day]
            tmpX = self._sliding_window_view(
                series, window_shape=(window, ), steps=(one_day, ))

            dataX = np.hstack((dataX, tmpX))

        # Real data to predict
        series = df[target_col].values[window:]
        dataY = self._sliding_window_view(
            series, window_shape=(one_day, ), steps=(one_day, ))

        window += one_day
        # Weather Feature
        for feat in weat_feat:
            series = df[feat].values
            tmpX = self._sliding_window_view(
                series, window_shape=(window, ), steps=(one_day, ))
            dataX = np.hstack((dataX, tmpX))

        # remove nan data
        remove_row = np.isnan(dataX).any(axis=1)
        dataX, dataY = dataX[~remove_row, :], dataY[~remove_row, :]
        remove_row = np.isnan(dataY).any(axis=1)
        dataX, dataY = dataX[~remove_row, :], dataY[~remove_row, :]

        return dataX, dataY

    def _preprocess(self, scale=False):
        data = pd.read_csv(self.data_path)
        data[self.target] = data[self.target] * 1000
        data = self._make_continuous_data(data, self.one_day)

        for feat in self.weat_feat:
            scaler = MinMaxScaler()
            data.loc[:, feat] = scaler.fit_transform(data.loc[:, feat].values.reshape(-1, 1))

        data.loc[:, self.target] = self.scaler.transform(data.loc[:, self.target])

        trainX, trainY = self._make_dataset_vectorized(
            df=data,
            window=self.lag,
            one_day=self.one_day,
            weat_feat=self.weat_feat)

        print("dataX shape: ", trainX.shape)
        print("dataY shape: ", trainY.shape)
        return trainX, trainY

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
