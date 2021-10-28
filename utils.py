import os
import tqdm

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchnet as tnt


def numbered_modelname(output_path: str, model_type: str, defined_name:str = ""):
    """
    It finds the current model number by finding the output folder,
    and returns updated model number to save the model name.

    Args:
        output_path (str): Output path
        model_type (str) : Type of model to use
        defined_name (str): Model name to save

    Returns:
        logdir: Directory name to save the output
    """
    model_name = "{}_{}".format(model_type, defined_name) \
                    if defined_name != '' else model_type
    if os.path.exists(output_path):
        run_number = [int(f.split("-")[0]) for f in os.listdir(output_path) if
                        os.path.isdir(os.path.join(output_path, f)) and model_name in f]
        run_number = max(run_number) + 1 if len(run_number) > 0 else 0
    else:
        run_number = 0

    logdir = os.path.expanduser(os.path.join(output_path,
                                            '{}-{}'.format(run_number, model_name)))
    os.makedirs(logdir, exist_ok=True)
    return logdir


def get_region_data(data_path: str, system_profile_file: str, region: str):
    """
    Returns the system id list that located in certain region.
    It finds the system id in same region by searching system_profile_info.csv

    Args:
        data_path (str): Data path of system id files
        system_profile_file (str): Data path of system profile info file
        region (str): Region name to select

    Returns:
        filelist[list]: Selected system id file list
    """
    filelist = []
    meta_data = pd.read_csv(system_profile_file)
    id_list = list(meta_data[meta_data["stateProv"] == region]["SystemID"].values)

    if len(id_list) == 0:
        raise ValueError("There is no region named {}".format(region))
    print("Length of region {}: {}".format(region, len(id_list)))

    for x in os.listdir(data_path):
        cur_id = int(x.split("_")[5])
        if cur_id in id_list:
            filelist.append(x)
    return filelist

def draw_plot(y_pred, y_real,
              length: int =0, save_path=None, filename="total"):
    """
    Draw plots of test results

    Args:
        y_pred (np.ndarray): Prediction results
        y_real (np.ndarray): Label results
        length (int, optional): Number of data that wants to draw plot. Defaults to 0.
        save_path (str, optional): Output path to the figure. Defaults to None.
        filename (str, optional): Output filename to the figure. Defaults to "total".
    """

    if not length:
        length = len(y_pred)

    fig = plt.figure(figsize=(10, 6))

    plt.plot(y_pred[-length:].flatten(), label='pred')
    plt.plot(y_real[-length:].flatten(), label='real')
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path,
                                 "{}_result_for_prediction.png".format(filename)),
                    bbox_inches='tight', pad_inches=0)


class MetricLogger(object):
    """
    Metric logger to see the log during training.
    """
    def __init__(self, epoch):
        self.metric_loss = tnt.meter.AverageValueMeter()
        self.mse_loss = tnt.meter.MSEMeter()
        self.epoch = epoch

    def _reset_meters(self):
        #self.metric_loss.reset()
        self.mse_loss.reset()

    def on_forward(self, pred, real, i, total):
        # self.metric_loss.add(pred, real)
        self.mse_loss.add(pred, real)

        print('[%d / %d] Training loss: %.4f ' % (i, total, self.mse_loss.value() * len(pred) ))

    def on_start_epoch(self, lr):
        self._reset_meters()
        print("======================= Epoch: [%d] (lr: %.4f) ======================= " % (self.epoch, lr))