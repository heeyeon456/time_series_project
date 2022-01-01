import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm
import pickle
import json
import warnings
import shutil

from dataset.dataset import TimeSeriesListDataset
from models.skmodels import MLModels
from metrics import *
import utils
import configs


class MLEngine:
    def __init__(self, args):
        """[summary]

        Args:
            args (ArgumentParser): Arguments to train or test the model.
        """
        self.args = args

        self.train_days = args.train_days

        self.dataset = self._get_dataset()
        self.data_X, self.data_y = self.dataset.data

        self.scaler = self.dataset.scaler
        self.model = MLModels(model_name=self.args.model_to_use,
                              feat_size = self.data_X.shape[1])
        self.wrapper_model = None

    def _get_dataset(self):
        # Initiate dataset
        dataset = TimeSeriesListDataset(data_path=self.args.data_path,
                                        target=self.args.target,
                                        lag=self.args.lag,
                                        output_dim=self.args.outdim,
                                        status=self.args.phase,
                                        weat_feat=self.args.additional_feat)
        return dataset

    def train(self):
        # train model
        print("\nTrain '{}' model..".format(self.args.target))
        print("- Input: use features of past {} days (additional features: {})".format(
                        self.args.lag // 13, self.args.additional_feat))
        print("- Output: predict one-day head '{} outputs \n".format(
                        self.args.outdim))

        train_days = self.train_days
        model = self.model.model
        self.wrapper_model = self.model.get_multi_out_model(model)
        print(self.wrapper_model)
    
        predictedY, testY = [], []

        #for x in range(1, len(self.data_y)):
        for x in range(1, 3):
            if x < train_days:
                trainX, trainY = self.data_X[:x+1], self.data_y[:x+1]
            else:
                trainX, trainY = self.data_X[x-train_days:x+1], self.data_y[x-train_days:x+1]


            pred_y = self.scaler.inverse_transform(
                self.train_oneday(trainX, trainY))
            predictedY.append(pred_y)
            test_y = self.scaler.inverse_transform(
                self.data_y[x]
            )
            testY.append(test_y)

        self._write_result(predictedY, testY, out_path=self.args.model_dir)

    def train_oneday(self, dataX, dataY):
        """
        It trains the scikit-learn model.
        It has no batch (train at once).
        """
        trainX, trainY = dataX[:-1], dataY[:-1]
        testX = dataX[-1].reshape(1, -1)

        if len(trainY) == 0:
            warnings.warn("Pass training model")
            return

        self.model.forward(trainX, trainY)

        # test result
        predicted = self.model.predict(testX)
        return predicted

    def _write_result(self, pred, real, out_path="."):
        pred = np.array(pred)
        real = np.array(real)

        output_filename = f"{self.args.model_to_use}_result.out"

        with open(os.path.join(out_path, output_filename), 'a') as f:
            f.write("-----------------------------------\n")
            f.write("RMSE: %.4f\n" % (calcRMSE(real, pred)))
            f.write("MAE: %.4f\n" % (calcMAE(real, pred)))
            f.write("MAPE: %.4f %%\n" % (calcMAPE(real, pred)))
            f.write("PRMSE : %.4f %%\n" % (calcPRMSE(real, pred)))
            f.write("sMAPE: %.4f %%\n" % (calcSMAPE(real, pred)))
            f.write("Correlation: %.4f (p=%.2f) %%\n" % (calcCorr(real, pred)))
            f.write("-----------------------------------")

        print("-----------------------------------\n")
        print("RMSE: %.4f\n" % (calcRMSE(real, pred)))
        print("MAE: %.4f\n" % (calcMAE(real, pred)))
        print("PRMSE : %.4f %%\n" % (calcPRMSE(real, pred)))
        print("-----------------------------------")

        #with open(os.path.join(out_path, output_filename)) as f:
        #    for line in f.readlines():
        #        print(line)

    def _save_model(self):
        output_path = utils.numbered_modelname(
            self.args.model_dir, self.args.model_to_use)
        cur_model = output_path

        model_name = self.args.model_to_use

        pkl_filename = os.path.join(output_path, "{}.pkl".format(model_name))
        with open(pkl_filename, 'wb') as f:
            pickle.dump(self.wrapper_model, f)

        json_filename = os.path.join(output_path,
                                     "{}_best_param.json".format(self.args.model_to_use))

        param_data = str(self.model.get_params())
        jsonStr = json.dumps(param_data)
        with open(json_filename, 'w') as f:
            f.write(jsonStr)

        print("\nTrained model is successly saved at {}\n".format(output_path))
        return cur_model

    def _load_model(self):
        with open(self.args.ckpt_path, 'rb') as f:
            model = pickle.load(f)
        return model


if __name__ == '__main__':
    parser = configs.config_args()
    args = parser.parse_args()

    engine = MLEngine(args)
    engine.train()
