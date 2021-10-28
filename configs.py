import argparse


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")


def config_args():
    """[summary]

    Returns:
        [type]: [description]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/Data/project/ess/DKASC/DKASC_sum_data.csv',
                        help='Path to root directory')
    parser.add_argument('--phase', type=str, default='train', required=True,
                        help='Phase to run: train / test / all',
                        choices=['train', 'test'])
    parser.add_argument('--target', type=str, default='Active_Power',
                        help='target time-series data to make prediction model')
    parser.add_argument('--model_to_use', type=str, default=None, required=True,
                        help='Model to use: svr, rf, mlp, tcn, lstm, cnn_lstm',
                        choices=['svr', 'rf', 'mlp', 'tcn', 'lstm', 'cnn_lstm'])
    parser.add_argument('--save_model', type=_str2bool, default=True,
                        help='Save trained model or not')
    parser.add_argument('--model_dir', type=str, default="./output_dir",
                        help='output dir that save the trained model weights')
    parser.add_argument('--ckpt_path', type=str, default="./output_dir",
                        help='path of model weight that used as inference')
    parser.add_argument('--validation', type=_str2bool, default=False,
                        help='Use validation for training')

    # Data Preprocessing
    parser.add_argument('--lag', type=int, default=13,
                        help='Number of day that use as lag to train the model')
    parser.add_argument('--outdim', type=int, default=1,
                        help='Number of times that model predicts')
    parser.add_argument('--additional_feat', nargs="+",
                        default=[])
    parser.add_argument('--num_cores', type=int, default=1,
                        help='Number of cores for preprocessing the input')


    # Deel Learning Model Training
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of batch for training deep learning model')
    parser.add_argument('--lr', type=float, default=1e-03,
                        help='Learning rate for training deep learning model')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay for optimizer')
    parser.add_argument('--hidden_layer_num', type=int, default=50,
                        help="Number of hidden_layer for MLP")
    parser.add_argument('--add_hidden_layer_num', type=int, default=4,
                        help="Number of additional model's hidden layer")
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs for training deep learning model')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to train deep learning model')
    parser.add_argument('--clip_grad_norm', type=_str2bool, default=False,
                        help="Whether to use clip_grad_norm or not")
    parser.add_argument('--clip_max_norm', type=float, default=5,
                        help="max grad norm of clip_grad_norm function")

    # etc
    parser.add_argument('--show_each_output', type=_str2bool, default=True,
                        help='Whether to show each output results of multi output regressor')

    return parser