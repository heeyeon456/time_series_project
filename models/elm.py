import numpy as np
from scipy.linalg import pinv2, inv
import time

class elm():
    """
    Function: elm class init
    Reference code: https://github.com/5663015/elm
    -------------------
    Parameters:
    shape: list, shape[hidden units, output units]
        numbers of hidden units and output units
    activation_function: str, 'sigmoid', 'relu', 'sin', 'tanh' or 'leaky_relu'
        Activation function of neurals
    C: float
        regularization parameter
    one_hot: bool, Ture or False, default True 
        The parameter is useful only when elm_type == 'clf'. If the labels need to transformed to
        one_hot, this parameter is set to be True
    random_type: str, 'uniform' or 'normal', default:'normal'
        Weight initialization method
    """
    def __init__(self, input_shape, hidden_units, activation_function, C,
                 one_day, random_type='normal', algorithm='solution1'):
        self.hidden_units = hidden_units
        self.input_shape = input_shape
        self.activation_function = activation_function
        self.random_type = random_type
        self.C = C
        self.class_num = one_day
        self.beta = np.zeros((self.hidden_units, self.class_num))   
        self.algorithm = algorithm

        # Randomly generate the weight matrix and bias vector from input to hidden layer
        # 'uniform': uniform distribution
        # 'normal': normal distribution
        if self.random_type == 'uniform':
            self.W = np.random.uniform(low=0, high=1, size=(self.hidden_units, self.input_shape))
            self.b = np.random.uniform(low=0, high=1, size=(self.hidden_units, 1))
        if self.random_type == 'normal':
            self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.input_shape))
            self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

    def __repr__(self):
        return (
            f"ELM(input_shape = {self.input_shape}, \n \
            class_num = {self.class_num}, \n \
            activation_function = {self.activation_function}, \n, \
            C = {self.C}, \n \
            hidden_units = {self.hidden_units})"
        )

    def __input2hidden(self, x):
        """
        compute the output of hidden layer according to different activation function
        """
        temH = np.dot(self.W, x.T) + self.b

        if self.activation_function == 'sigmoid':
            H = 1/(1 + np.exp(- temH))

        if self.activation_function == 'relu':
            H = temH * (temH > 0)

        if self.activation_function == 'sin':
            H = np.sin(temH)

        if self.activation_function == 'tanh':
            H = (np.exp(temH) - np.exp(-temH))/(np.exp(temH) + np.exp(-temH))

        if self.activation_function == 'leaky_relu':
            H = np.maximum(0, temH) + 0.1 * np.minimum(0, temH)

        return H

    def __hidden2output(self, H, beta):
        """
        compute the output
        """
        output = np.dot(H.T, beta)
        return output

    def fit_param(self, data, param):        
        trainX, trainY, testX, testY = data

        param_list = [v for v in param_dict.values()]
        all_comb = list(product(*param_list))
        rmse = []
        for x in all_comb:
            cur_param = {}
            i = 0
            for k in param_dict.keys():
                cur_param[k] = x[i]
                i += 1

            model = elm.elm(activation_function = 'relu',
                            **cur_param)

            model.fit(trainX, trainY)
            predicted = model.predict(testX)
            rmse.append(calcRMSE(corrected, testY))

        min_idx = rmse.index(min(rmse))
        selected_param = all_comb[min_idx]

        return selected_param

    def fit(self, X, y):
        """
        Function: Train the model, compute beta matrix, the weight matrix from hidden layer to output layer
        ------------------
        Parameter:
        algorithm: str, 'no_re', 'solution1' or 'solution2'
            The algorithm to compute beta matrix
        ------------------
        Return:
        beta: array
            the weight matrix from hidden layer to output layer
        train_score: float
            the accuracy or RMSE
        train_time: str
            time of computing beta
        """
 
        time1 = time.clock()   # compute running time
        H = self.__input2hidden(X)

        # no regularization
        if self.algorithm == 'no_re':
            self.beta = np.dot(pinv2(H.T), y)
        # faster algorithm 1
        elif self.algorithm == 'solution1':
            tmp1 = inv(np.eye(H.shape[0])/self.C + np.dot(H, H.T))
            tmp2 = np.dot(tmp1, H)
            self.beta = np.dot(tmp2, y)
        # faster algorithm 2
        elif self.algorithm == 'solution2':
            tmp1 = inv(np.eye(H.shape[0])/self.C + np.dot(H, H.T))
            tmp2 = np.dot(H.T, tmp1)
            self.beta = np.dot(tmp2.T, y_temp)
        else:
            print(f"{self.algorithm} is not defined")
        time2 = time.clock()

        # compute the results
        result = self.__hidden2output(H, self.beta)
        result = result * (result > 0)

        # Evaluate training results
        # If problem is regression, compute the RMSE
        train_score = np.sqrt(np.sum((result - y) * (result - y))/y.shape[0])
        train_time = str(time2 - time1)

    def predict(self, x):
        """
        Function: compute the result given data
        ---------------
        Parameters:
        x: array, shape[samples, features]
        ---------------
        Return:
        y_: array
            predicted results
        """
        H = self.__input2hidden(x)
        y_ = self.__hidden2output(H, self.beta)
        y_ = y_ * (y_ > 0)

        return y_
