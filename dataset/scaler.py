import numpy as np

class DefinedMinMaxScaler:
    """ MinMaxScaler for defined min and max value"""
    def __init__(self, min_num: int, max_num: int):
        """Constructor for minmaxscaler

        Args:
            min_num (int): Min value for scaling
            max_num (int): Max value for scaling
        """
        self.min_num = min_num
        self.max_num = max_num

    def transform(self, arr):
        return (arr - self.min_num) / (self.max_num - self.min_num)

    def inverse_transform(self, X):
        return X * (self.max_num - self.min_num) + self.min_num