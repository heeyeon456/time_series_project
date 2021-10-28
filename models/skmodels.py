from sklearn.svm import LinearSVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import uniform

import numpy as np
import random

class MLModels:
    """
    Consists of Scikit-learn model instances
    """
    def __init__(self, model_name, best_fit=False, seed=123):
        """
        Constructor for MLModels

        Args:
            model_name ([type]): Name of machine learning models.
                                 Choices = ['svr', 'rf', 'mlp']
            best_fit (bool, optional): Whether to search best parameters using Cross validation Random Search.
                                       Defaults to False.
            seed (int, optional): Random Seed. Defaults to 123.
        """
        np.random.seed(seed)
        self.seed = seed
        self.model_name = model_name
        self.random_seed = seed
        self.best_fit = best_fit
        self.param_dict, self.__model = self.get_skmodel()

    def _choice(self, param_dict):
        random.seed(self.random_seed)
        selected_param = {}
        for k, v in param_dict.items():
            sel = random.choice(param_dict[k])
            selected_param[k] = sel
        return selected_param

    def get_skmodel(self):
        """
        Get scikit-learn models
        if self.best_fit is true, parameters is fitted using RandomizedSearch
        else, parameters is randomly chosen by given parameter list dictionary.
        """
        def svr_model():
            param_dict = dict(kernel=['linear', 'rbf', 'poly'],
                              C=list(range(1, 100)),
                              epsilon=np.random.uniform(0.0, 10.0, 10))
            if self.best_fit:
                param = param_dict
                model = LinearSVR(gamma='scale')
            else:
                param = self._choice(param_dict)

                model = LinearSVR(C=param['C'],
                                  epsilon=param['epsilon'],
                                  random_state=self.seed)
            return param, model

        def rf_model():
            param_dict = dict(n_estimators=list(range(10, 100)),
                              max_depth=list(range(5, 50)))
            if self.best_fit:
                param = param_dict
                model = RandomForestRegressor(criterion='mse',
                                              n_jobs=4)
            else:
                param = self._choice(param_dict)
                model = RandomForestRegressor(n_estimators=50, #n_estimators=param['n_estimators'],
                                              max_depth=param['max_depth'],
                                              criterion='mse',
                                              n_jobs=4,
                                              random_state=self.seed)
            return param, model

        def mlp_model():
            param_dict = dict(
                hidden_layer_sizes=[(50, 50, 50), (50, 100), (100, 100), (20, 50, 20)],
                solver=['adam', 'sgd'],
                activation=['relu', 'tanh'],
                alpha=[1e-05, 5e-05, 1e-04, 5e-04, 1e-03, 5e-03, 1e-02],
                learning_rate=['constant', 'adaptive'],
                learning_rate_init=[0.05, 0.025, 0.01, 0.005, 0.001]
            )
            if self.best_fit:
                param = param_dict
                model = MLPRegressor(max_iter=30)
            else:
                param = self._choice(param_dict)
                model = MLPRegressor(hidden_layer_sizes=param['hidden_layer_sizes'],
                                     solver='adam',
                                     alpha=param['alpha'],
                                     learning_rate=param['learning_rate'],
                                     learning_rate_init=param['learning_rate_init'],
                                     activation=param['activation'],
                                     max_iter=50,
                                     random_state=self.seed)
            return param, model

        if self.model_name == 'svr':
            param_dict, model = svr_model()
        elif self.model_name == 'rf':
            param_dict, model = rf_model()
        elif self.model_name == 'mlp':
            param_dict, model = mlp_model()
        else:
            return
        return param_dict, model

    def get_multi_out_model(self, model):
        """
        To train the multi output model,
        it change the model as multiple one-output individual models.

        Args:
            model (class): Scikit-learn model class instance

        Returns:
            model : Scikit-learn multi output model.
        """
        self.__model = MultiOutputRegressor(model)
        return self.model

    def search_best_param(self, X, y):
        """[summary]

        Args:
            X ([type]): [description]
            y ([type]): [description]

        Returns:
            [type]: [description]
        """
        print("Find best parameters for {}..".format(self.model_name))
        clf = RandomizedSearchCV(self.model,  self.param_dict,
                                 cv=5, random_state=123)
        clf.fit(X, y[:, 0])
        self.params = clf.best_params_
        self.__model = clf
        self.param = clf.get_params()
        return clf

    def forward(self, X, y):
        """
        Train the model.
        Scikit-learn model don't use stochastic gradient descent.
        It trains the model at once (no batch size, every dataset is used as training)

        Args:
            X (np.ndarray): Training data features
            y (np.ndarray): Training data labels

        Returns:
            model: Trained model instance
        """
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """
        inference the model

        Args:
            X (np.ndarray): Query data features

        Returns:
            [type]: [description]
        """
        return self.model.predict(X)

    @property
    def model(self):
        return self.__model

    def get_params(self):
        return self.model.get_params()