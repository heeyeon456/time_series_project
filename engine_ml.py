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

        self.dataset = self._get_dataset()
        if self.args.phase == "train":
            self.train_X, self.train_y = self.dataset.data
            if self.args.validation:
                self.train_X, self.train_y, self.valid_X, self.valid_y =self.dataset.split_valid(
                    self.train_X, self.train_y, ratio=0.9)

            # scale data
            self.train_y = self.dataset.scaler.transform(self.train_X)
            self.train_y = self.dataset.scaler.transform(self.train_X)

        elif self.args.phase == "test":
            self.test_X, self.test_y, _, _ = self.dataset.data
            self.test_X, self.test_y = self._scale_data(self.test_X, self.test_y)

        self.model = MLModels(model_name=self.args.model_to_use)

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
        """
        It trains the scikit-learn model.
        It has no batch (train at once).
        """
        # train model
        print("\nTrain '{}' model..".format(self.args.target))
        print("- Input: use features of past {} days (additional features: {})".format(
                        self.args.lag // 13, self.args.additional_feat))
        print("- Output: predict one-day head '{} outputs \n".format(
                        self.args.outdim))

        if len(self.train_y) == 0:
            warnings.warn("Pass training model")
            return
        model = self.model.model
        self.wrapper_model = self.model.get_multi_out_model(model)

        self.model.forward(self.train_X, self.train_y)
        print(self.wrapper_model)

        if self.args.save_model:
            out_path = self._save_model()

        # check validation result
        if self.args.validation:
            predicted_data = self.model.predict(self.valid_X)

            y_data = self.dataset.scaler.inverse_transform(self.valid_y)
            inverse_pred = self.dataset.get_inverse(np.array(predicted_data))
            self._write_result(inverse_pred, y_data, "validate", out_path)

        # test result
        self.test(out_path)

    def test(self, out_path: str = None):
        """
        It tests the trained scikit-learn model.

        Args:
            out_path (str, optional): Output path that saves the model weights and results.
                                      Defaults to None.
        """
        if out_path is None:
            self.model = self._load_model()
            out_path = "/".join(self.args.ckpt_path.split("/")[:-1])
        predicted_data = self.model.predict(self.test_X)

        y_data = self.dataset.scaler.inverse_transform(self.test_y)
        inverse_pred = self.dataset.scaler.inverse_transform(np.array(predicted_data))
        pred, real = inverse_pred, y_data

        print("Total Mean of Multi Output Results")
        self._write_result(pred, real, "test", out_path)
        if self.args.show_each_output:
            for i in range(self.args.time_of_pred):
                print("Each Result of {}th Output Results".format(str(i)))
                self._write_result(pred[:, i], real[:, i],
                                   status="each", out_path=out_path)

    def _write_result(self, pred, real, status, out_path):
        pred = np.array(pred)
        real = np.array(real)

        if self.system_id is None:
            output_filename = "total_{}_result.out".format(status)
        else:
            output_filename = "{}_result.out".format(status)

        utils.draw_plot(pred, real, length=196, save_path=out_path, filename=status)

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
