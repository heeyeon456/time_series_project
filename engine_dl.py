import os
import sys

import numpy as np
from tqdm import tqdm
import pickle
import json
import warnings
import random

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import MSELoss
import torchnet as tnt

from dataset.dataset import TimeSeriesDataset

from models.mlp import MLPNetwork, RegressorNetwork
from models.tcn import TemporalCNNNetwork
from models.lstm import LSTMNetwork
from models.cnn_lstm import CNNLSTMNetwork

from metrics import *
import utils
import configs

random_seed = 123
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

class DLEngine:
    """
    This class is used to train and infer the pytorch based deep learning model.
    """
    def __init__(self, args):
        """
        Constructor method

        Args:
            args (ArgumentParser): Arguments to train or test the model
        """
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("current device: {}\n".format(self.device))

        # Define model

        input_dim = self.args.lag
        one_day = 13
        out_dim = self.args.outdim
        add_input_dim = (input_dim+out_dim) * len(self.args.additional_feat)

        self.dataset = self._get_dataset("train")

        self.train_loader, self.test_loader, self.valid_loader = \
            self.make_loader(self.dataset, status="train")

        print("ts model:")
        num_feat = len(self.args.additional_feat) + 1
        input_dim = one_day
        self.ts_model = LSTMNetwork(input_dim=input_dim,
                                    hidden_size=self.args.hidden_layer_num,
                                    output_dim=self.args.hidden_layer_num,
                                    num_feat=1,
                                    device=self.device,
                                    layer_dim=1,
                                    dropout=0.1)

        print("add model:")
        if self.args.model_to_use == "mlp":
            self.add_model = MLPNetwork(input_dim=add_input_dim,
                                        hidden_size=self.args.add_hidden_layer_num//2,
                                        num_layers=3,
                                        output_dim=self.args.add_hidden_layer_num)

        elif self.args.model_to_use == "tcn":
            input_dim = input_dim + 24
            channel_list = [input_dim, input_dim // 2, input_dim // 4]
            self.args.add_hidden_layer_num = (input_dim // 4) * num_feat
            self.add_model = TemporalCNNNetwork(input_dim=input_dim,
                                                num_channels=channel_list,
                                                kernel_size=4,
                                                dropout=0.1)

        elif self.args.model_to_use == "lstm":
            input_dim = one_day // 4
            self.add_model = LSTMNetwork(input_dim=input_dim,
                                        hidden_size=self.args.add_hidden_layer_num,
                                        output_dim=self.args.add_hidden_layer_num,
                                        num_feat=num_feat,
                                        device=self.device,
                                        layer_dim=2,
                                        dropout=0.1)

        elif self.args.model_to_use == "cnn_lstm":
            input_dim = one_day // 4
            self.add_model = CNNLSTMNetwork(input_dim=input_dim,
                                            hidden_size=self.args.add_hidden_layer_num,
                                            output_dim=self.args.add_hidden_layer_num,
                                            num_feat=num_feat,
                                            device=self.device,
                                            layer_dim=1,
                                            dropout=0.2)
            print(self.add_model)
        else:
            warnings.warn("There is no model type {}".format(self.args.model_to_use))

        self.regressor_model = RegressorNetwork(input1_dim=self.args.hidden_layer_num,
                                                input2_dim=self.args.add_hidden_layer_num,
                                                output_dim=out_dim)

        self.ts_model.to(self.device)
        self.add_model.to(self.device)
        self.regressor_model.to(self.device)

        self.criterion = MSELoss(reduction='sum')
        self.criterion.to(self.device)

        # Define optimizer
        total_parameters = list(self.ts_model.parameters()) + list(self.add_model.parameters()) \
                + list(self.regressor_model.parameters())

        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(total_parameters,
                                             lr=self.args.lr,
                                             momentum=0.9,
                                             weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(total_parameters,
                                              weight_decay=self.args.weight_decay,
                                              lr=self.args.lr)
        else:
            print("no optimizer name ", self.args.optimizer)

        if self.args.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.ts_model.parameters(), max_norm=self.args.clip_max_norm, norm_type=2)

        if self.args.model_to_use in ["tcn", "cnn_lstm"]:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 50, eta_min=1e-05)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=20, gamma=0.5)


    def make_loader(self, dataset, status="train"):
        """
        Make data loader to train and test model

        Args:
            data_list (list): System id list to construct as training dataset
            status

        Returns:
            train_loader(Iterator) : Training data loader
            test_loader(Iterator) : Test data loader
            valid_loader(Iterator) : Validation data loader
        """

        train_loader, valid_loader, test_loader = None, None, None
        valid_ratio = 0.1 if self.args.validation else 0
        if status == "train":
            train_dataset, test_dataset, valid_dataset = \
                    self._split_dataset(dataset, valid_ratio=valid_ratio, test_ratio=0.2)

            train_loader = DataLoader(train_dataset,
                                      batch_size=self.args.batch_size,
                                      shuffle=True,
                                      num_workers=4)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.args.batch_size,
                                     shuffle=False,
                                     num_workers=4)
            if self.args.validation:
                valid_loader = DataLoader(valid_dataset,
                                          batch_size=self.args.batch_size,
                                          shuffle=False,
                                          num_workers=4)
        else:
            test_loader = DataLoader(dataset,
                                     batch_size=self.args.batch_size,
                                     shuffle=False,
                                     num_workers=4)

        return train_loader, test_loader, valid_loader

    def _get_dataset(self, status):
        param_set = {"data_path" : self.args.data_path, "target" : self.args.target,
                    "lag" : self.args.lag, "output_dim" : self.args.outdim, "status": status,
                    "weat_feat" : self.args.additional_feat}
        dataset = TimeSeriesDataset.constructor(**param_set)

        return dataset

    def _split_dataset(self, dataset, valid_ratio, test_ratio):
        indices = set(range(dataset.__len__()))
        test_idx = round(test_ratio * dataset.__len__())
        # dataset_train = torch.utils.data.Subset(dataset, indices[:split_train_idx])
        # dataset_val = torch.utils.data.Subset(dataset, indices[split_train_idx:])
        test_indices = set(random.sample(indices, test_idx))
        train_indices = indices.difference(test_indices)
        dataset_train = torch.utils.data.Subset(dataset, list(train_indices))
        dataset_test = torch.utils.data.Subset(dataset, list(test_indices))
        dataset_val = None

        if valid_ratio != 0:
            valid_idx = round(valid_ratio * len(train_indices))
            valid_indices = set(random.sample(train_indices, valid_idx))
            train_indices_ = train_indices.difference(valid_indices)
            dataset_train = torch.utils.data.Subset(dataset, list(train_indices_))
            dataset_val = torch.utils.data.Subset(dataset, list(valid_indices))

        print("training examples: {}".format(len(dataset_train)))
        print("test examples: {}".format(len(dataset_test)))
        if dataset_val is not None:
            print("validating examples: {}".format(len(dataset_val)))
        return dataset_train, dataset_val, dataset_test

    def train_one_epoch(self, epoch):
        """
        Training the model one epoch.

        Args:
            epoch (int): current epoch
        """
        # Set as train mode
        self.ts_model.train()
        self.add_model.train()
        self.regressor_model.train()

        logger = utils.MetricLogger(epoch)
        total_train_loss, total_pred_loss = 0., 0.
        total_num = 0

        logger.on_start_epoch(self.lr_scheduler.get_lr()[0])
        total_iter = len(self.train_loader)
        for idx, (data1_x, data2_x, data_y) in enumerate(self.train_loader):
            data1_x = data1_x.to(self.device)
            data2_x = data2_x.to(self.device)
            data_y = data_y.to(self.device)
            total_num += 1
            #total_num += len(data_y)

            ts_embd = self.ts_model.forward(data1_x)
            add_embd = self.add_model.forward(data2_x)
            pred = self.regressor_model.forward(ts_embd, add_embd)
            batch_loss = self.criterion(pred, data_y)

            self.optimizer.zero_grad()
            batch_loss.backward()

            total_train_loss += batch_loss

            if idx % 10 == 0:
                logger.on_forward(pred, data_y, idx, total_iter)
            self.optimizer.step()

        total_pred_loss = total_train_loss / total_num
        print("total prediction loss : %.4f" % (total_pred_loss))

    def validate(self, epoch):
        """
        Validating the model during training.
        It activates if args.validation is True.

        Args:
            epoch (int): Current epoch
        """
        # Set as eval mode
        self.ts_model.eval()
        self.add_model.eval()
        self.regressor_model.eval()

        pred_result, real_result = [], []
        total_num = 0
        total_valid_loss, total_pred_loss = 0., 0.
        with torch.no_grad():
            for idx, (data1_x, data2_x, data_y) in enumerate(self.valid_loader):
                data1_x = data1_x.to(self.device)
                data2_x = data2_x.to(self.device)
                data_y = data_y.to(self.device)
                #total_num += len(data_y)
                total_num += 1

                ts_embd = self.ts_model.forward(data1_x)
                add_embd = self.add_model.forward(data2_x)
                pred = self.regressor_model.forward(ts_embd, add_embd)
                batch_loss = self.criterion(pred, data_y)

                reverse_pred = self.dataset.scaler.get_inverse(
                        pred.cpu().detach().numpy())
                reverse_real = self.dataset.scaler.get_inverse(
                        data_y.cpu().detach().numpy())

                total_valid_loss += batch_loss

                pred_result.extend(reverse_pred)
                real_result.extend(reverse_real)

        total_pred_loss = (total_valid_loss / total_num)
        #print("---------- Validation Result(Epoch: [{}]) -----------".format(epoch))
        print("Validation prediction loss: %.4f" % (total_pred_loss))

        self._write_result(pred_result, real_result, status="validate")

    def train(self):
        """
        It trains the model.
        """
        print("\nTrain '{}' model..".format(self.args.target))
        print("- Input: use features of past {} days (additional features: {})".format(
                        self.args.lag // 13, self.args.additional_feat))
        print("- Output: predict one-day head '{} data\n".format(
                        self.args.outdim))

        for epoch in range(self.args.num_epochs):
            self.train_one_epoch(epoch)
            if self.args.validation:
                self.validate(epoch)
            self.lr_scheduler.step()

        if self.args.save_model:
            out_path = self._save_model()
        else:
            out_path = "."
        self.test(out_path)

    def test(self, out_path=None):
        """
        It tests the model and write performance of model to out_path.

        Args:
            out_path (str, optional): Output path that saves the model weights and results.
                                      Defaults to None.
        """

        self.ts_model.eval()
        self.add_model.eval()
        self.regressor_model.eval()

        print("\nTest model..")
        pred_result, real_result = [], []
        total_num = 0
        scaler = None

        if out_path is None:
            self._load_model()
            out_path = "/".join(self.args.ckpt_path.split("/")[:-1])

        with torch.no_grad():
            for idx, (data1_x, data2_x, data_y) in enumerate(self.test_loader):
                data1_x = data1_x.to(self.device)
                data2_x = data2_x.to(self.device)
                total_num += len(data_y)

                ts_embd = self.ts_model.forward(data1_x)
                add_embd = self.add_model.forward(data2_x)
                pred = self.regressor_model.forward(ts_embd, add_embd)

                reverse_pred = self.dataset.scaler.get_inverse(
                        pred.cpu().detach().numpy())
                reverse_real = self.dataset.scaler.get_inverse(
                        data_y.numpy())

                pred_result.extend(reverse_pred)
                real_result.extend(reverse_real)

        self._write_result(pred_result, real_result,
                           status="total", out_path=out_path)

        if self.type != "long" and self.args.show_each_output:
            for i in range(self.args.time_of_pred):
                print("Each Result of {}th Output Results".format(str(i)))
                self._write_result(np.array(pred_result)[:, i], np.array(real_result)[:, i],
                                   status="each", out_path=out_path)

    def _write_result(self, pred, real, status, out_path=None):
        pred = np.array(pred)
        real = np.array(real)
        if self.system_id is None:
            output_filename = "total_{}_result.out".format(status)
        else:
            output_filename = "{}_result.out".format(status)

        utils.draw_plot(pred, real, length=196, save_path=out_path, filename=status)
        if status != "validate":
            with open(os.path.join(out_path, output_filename), 'a') as f:
                f.write("-----------------------------------\n")
                f.write("RMSE: %.4f\n" % (calcRMSE(real, pred)))
                f.write("MAE: %.4f\n" % (calcMAE(real, pred)))
                f.write("PRMSE : %.4f %%\n" % (calcPRMSE(real, pred)))
                f.write("PRMSE_MEAN : %.4f %%\n" % (calcPRMSE_mean(real, pred)))
                f.write("MAPE: %.4f %%\n" % (calcMAPE(real, pred)))
                f.write("sMAPE: %.4f %%\n" % (calcSMAPE(real, pred)))
                f.write("Correlation: %.4f (p=%.2f) %%\n" % (calcCorr(real, pred)))
                f.write("-----------------------------------")

            np.savetxt(os.path.join(out_path, "pred.txt"), pred)
            np.savetxt(os.path.join(out_path, "real.txt"), real)


        print("-----------------------------------\n")
        print("RMSE: %.4f\n" % (calcRMSE(real, pred)))
        print("MAE: %.4f\n" % (calcMAE(real, pred)))
        print("PRMSE : %.4f %%\n" % (calcPRMSE(real, pred)))
        print("PRMSE_MEAN : %.4f %%\n" % (calcPRMSE_mean(real, pred)))
        print("-----------------------------------")


    def _save_model(self):
        output_path = utils.numbered_modelname(
            self.args.model_dir, self.args.model_to_use, self.args.model_name)
        model_dict = {"ts": self.ts_model, "add": self.add_model, "reg": self.regressor_model}
        ckpt_dict = {}
        if self.seasonal is not None:
            model_name = "{}_{}.pt".format(self.args.model_to_use, self.seasonal)
        elif self.holiday is not None:
            model_name = "{}_{}.pt".format(self.args.model_to_use, self.holiday)
        else:
            model_name = "{}.pt".format(self.args.model_to_use)

        for k, v in model_dict.items():
            ckpt_dict[k] = v.state_dict()

        ckpt_filename = os.path.join(output_path, model_name)
        torch.save({'epoch': self.args.num_epochs,
                    'model_state_dict': ckpt_dict,
                    'optimizer_state_dict': self.optimizer.state_dict()
                    }, ckpt_filename)

        #scaler_filename = os.path.join(output_path, "scaler.gz")
        #joblib.dump(self.dataset.data_scaler, scaler_filename)

        print("\nTrained model is successly saved at {}".format(output_path))
        return output_path

    def _load_model(self):
        model_dict = {"ts": self.ts_model, "add": self.add_model, "reg": self.regressor_model}
        ckpt_dict = torch.load(self.args.ckpt_path)
        for k, v in model_dict.items():
            v.load_state_dict(ckpt_dict['model_state_dict'][k])

        #scaler_path = "/".join(self.args.ckpt_path.split("/")[:-1])
        #scaler = joblib.load(os.path.join(scaler_path, "scaler.gz"))

if __name__ == "__main__":
    parser = configs.config_args()
    args = parser.parse_known_args()[0]

    engine = DLEngine(args)
    engine.train()