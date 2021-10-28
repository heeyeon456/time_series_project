from models.skmodels import MLModels
from models.mlp import MLPNetwork

import numpy as np
from pyswarm import pso
import time

class EnsembleModel:
    """"""
    def __init__(self, model_to_use, ensemble_num, input_dim=None,
                 output_dim=None, use_pso=False, time_index=-1):
        self.model_to_use = model_to_use
        self.ensemble_num = ensemble_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_pso = use_pso

        if self.use_pso:
            assert self.ensemble_num ==2, "PSO ensemble method should have ensemble_num = 2"
            self.pso_ins = PSOEnsemble(self.ensemble_num)
            self.time_index = time_index

        else:
            self.weights = None

        if model_to_use in ['rf', 'svr', 'mlp']:
            self.model_list = self._make_ensemble_model()
        else:
            self.model_list = self._make_ensemble_network()

    def _make_ensemble_model(self):
        model_list = []
        for i in range(self.ensemble_num):
            model = MLModels(self.model_to_use, seed=None)
            model_list.append(model)
        return model_list

    def _make_ensemble_network(self):
        model_list = []
        for i in range(self.ensemble_num):
            network = MLPNetwork(input_dim=self.input_dim,
                                 hidden_size=0,
                                 output_dim=self.output_dim)
            model_list.append(network)

    def get_model(self):
        return [x.get_model() for x in self.model_list]

    def get_best_params(self):
        return [x.get_best_params() for x in self.model_list]

    def forward(self, X, y):
        for i in range(self.ensemble_num):
            self.model_list[i].forward(X, y)
        if self.use_pso:
            self.train_ensemble_weight(X, y)

    def train_ensemble_weight(self, X, y):
        predicted_arr = self.predict(X, ensemble=False)
        time_dict = self.pso_ins.make_dataset(X[:, self.time_index], predicted_arr, y)
        for k in time_dict.keys():
            self.pso_ins.weights[k] = self.pso_ins.initialize(k)
        self.pso_ins.execute(time_dict)

    def predict(self, X, ensemble=True):
        keys = ["alpha", "beta", "gamma"]
        predicted = []
        for i in range(self.ensemble_num):
            tmp_pred = self.model_list[i].predict(X)
            predicted.append(tmp_pred)

        if not ensemble:
            return predicted
        else:
            final_pred = []
            if not self.use_pso:
                final_pred.append(np.mean(np.array(predicted), axis=0))
            else:
                time_arr = X[:, self.time_index]
                self.weights = self.pso_ins.weights
                print(self.weights)
                for i in range(len(time_arr)):
                    if time_arr[i] not in self.weights.keys():
                        e_result = np.mean(predicted, axis=0)
                    else:
                        cur_weight = self.weights[time_arr[i]]
                        e_result = sum([predicted[x][i] * cur_weight[keys[x]] for x in range(self.ensemble_num)])
                        final_pred.append(e_result)
            final_pred = np.array(final_pred).reshape(-1, 1)
            return final_pred

class PSOEnsemble(object):
    """[summary]

    Args:
        object ([type]): [description]

    Returns:
        [type]: [description]
    """
    def __init__(self, model_num=0):
        self.model_num = model_num
        paramRange = [0, 1]
        self.optimizer = PSOModel(bounds=[paramRange, paramRange],
                                  object_function=self._pso_object_func,
                                  constraint_function=self._pso_constraints)
        self.weights = {}

    def _pso_object_func(self, x, **kwargs):
        timeData = kwargs['timeData']
        yHat = []
        for data in timeData:
            yHat.append(abs(data[-1] - sum(data[:-1] * x)))
        return np.mean(yHat)

    def _pso_constraints(self, x, timeData):
        return tuple([i/sum(x) for i in x])

    def make_dataset(self, time_arr, predicted_arr, real_arr):
        time_dict = {}
        predicted_arr = np.array(predicted_arr).reshape(-1, self.model_num)

        for i in range(len(time_arr)):
            data = tuple()
            for pred in predicted_arr[i]:
                data += (pred, )
            data += (real_arr[i][0], )

            t = time_arr[i]
            if t not in time_dict.keys():
                time_dict[t] = []
            time_dict[t].append(data)
        return time_dict

    def initialize(self, key):
        if key not in self.weights:
            if self.model_num == 2:
                return {"time": key, "alpha": 0.5, "beta": 0.5}
            elif self.model_num == 3:
                return{"time": key, "alpha": 1/3, "beta": 1/3, "gamma": 1/3}

    def execute(self, forecastData):
        start_time = time.time()
        if self.model_num == 2:
            key_list = ['alpha', 'beta']
        elif self.model_num == 3:
            key_list = ['alpha', 'beta', 'gamma']

        for t in forecastData.keys():
            timeData = forecastData[t]
            weights = self.train_weights(timeData)
            cur_dict = {'time': t}
            for i in range(len(key_list)):
                cur_dict[key_list[i]] = weights[i]
            self.weights[t] = cur_dict
        elapsed_time = time.time() - start_time
        #print(f'total run time {elapsed_time: 5.3f} sec')


    def train_weights(self, timeData):
        start_time = time.time()

        parameter = {"timeData": timeData}
        options = dict(swarmsize=30, omega=0.5, maxiter=30, minstep=1e-06)
        xopt, fopt = self.optimizer.optimize(debug=False, parameters=parameter, options=options)

        elapsed_time = time.time() - start_time
        #print(f'runtime: {elapsed_time:5.3f} sec')

        return xopt

class PSOModel(object):
    """[summary]

    Args:
        object ([type]): [description]
    """
    def __init__(self, bounds, object_function, constraint_function):
        if bounds is not None:
            self.lowLimit, self.highLimit = self.__makeBounds(bounds)
        self.object_function = object_function
        self.constraint_function = constraint_function

    def __makeBounds(self, bounds):
        lowLimit, highLimit = [], []
        for bound in bounds:
            low, high = bound
            lowLimit.append(low)
            highLimit.append(high)
        return lowLimit, highLimit

    def optimize(self, parameters=None, debug=False, options=None):
        default = dict(swarmsize=30, omega=0.5, maxiter=30, minstep=1e-06)
        if options is not None:
            default = dict(default, **options)
        xopt, fopt = pso(func=self.object_function, lb=self.lowLimit, ub=self.highLimit, debug=debug,
                         swarmsize=default['swarmsize'], omega=default['omega'], maxiter=default['maxiter'],
                         minstep=default['minstep'], f_ieqcons=self.constraint_function, kwargs=parameters)
        return xopt, fopt