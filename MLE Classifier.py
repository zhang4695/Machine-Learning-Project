import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats

# import pymc3 as pm3
# import numdifftools as ndt
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
import sklearn
import sklearn.metrics as metrics
# %matplotlib inline

train = pd.read_csv(r'C:\Users\Tianyi Zhang\Desktop\MachineLearning\propublicaTrain.csv')
test_data = pd.read_csv(r'C:\Users\Tianyi Zhang\Desktop\MachineLearning\propublicaTest.csv')
y_act, x_test = np.array(test_data)[:,0], np.array(test_data)[:,:9]


# ##Gaussian MLE classifier
class GaussianClassifier():
    def __init__(self, train):
        self.inpu = np.array(train)[:, :9]
        self.y_0_prior = np.array(train[train.two_year_recid == 0].two_year_recid).shape[0] / self.inpu.shape[0]
        self.y_1_prior = 1 - self.y_0_prior
        self.mu_0 = np.array(train[train.two_year_recid == 0].mean()[1:9]).reshape(-1, 1)
        self.mu_1 = np.array(train[train.two_year_recid == 1].mean()[1:9]).reshape(-1, 1)
        self.cov_0 = np.cov(np.array(train[train.two_year_recid == 0])[:, 1:9], rowvar=False)
        self.cov_1 = np.cov(np.array(train[train.two_year_recid == 1])[:, 1:9], rowvar=False)

    def train(self, data, mu, cov):
        data = data.reshape(-1, 1)
        cov_inv = np.linalg.inv(cov)
        denominator = np.sqrt(((2 * np.pi) ** 8) * np.linalg.det(cov))
        numerator = -0.5 * ((data - mu).T @ cov_inv @ (data - mu))
        res = float(np.exp(numerator) * 1. / denominator)
        return res

    def pred(self, input_set):
        pred = []
        for row in input_set[:, 1:]:
            if self.train(row, self.mu_0, self.cov_0) * self.y_0_prior >= self.train(row, self.mu_1,
                                                                                     self.cov_1) * self.y_1_prior:
                pred.append(0)
            else:
                pred.append(1)
        return pred

# ##KNN Classifier
class KNNClassifier():
    def __init__(self, train):
        self.train_result = np.array(train)[:, 0]
        self.train = np.array(train)[:, 1:9]
        self.scaler = sklearn.preprocessing.MinMaxScaler()
        self.scaler.fit(self.train)
        self.train = self.scaler.transform(self.train)

    def norm(self, row1, row2, L):
        return np.linalg.norm(row1 - row2, L)

    def get_neighbors(self, test_row, num_nei, L):
        distance = []
        for idx in range(len(self.train)):
            distance.append((self.train_result[idx], self.norm(test_row, self.train[idx], L)))
        distance.sort(key=lambda tup: tup[1])

        neighbors = []
        for idx in range(num_nei):
            neighbors.append(distance[idx][0])
        return max(neighbors, key=neighbors.count)

    def operate(self, test, num_nei=10, metric=None):
        test = test[:, 1:]
        self.scaler.fit(test)
        test = self.scaler.transform(test)
        pred_res = []
        if metric == 2 or metric == None:
            L = None
        elif metric == 1:
            L = 1
        else:
            L = np.inf
        for each, ind in zip(test, range(len(test))):
            pred_res.append(self.get_neighbors(each, num_nei, L))
            if ind % 500 == 0:
                print('in processing --->', ind, '\n')
        return pred_res

# ## Naive Bayes
class NaiveBayes():
    def __init__(self, train):

        self.result = np.array(train)[:, 0]
        self.y_0 = np.array(train[train.two_year_recid == 0])[:, 1:9]
        self.y_1 = np.array(train[train.two_year_recid == 1])[:, 1:9]
        self.y_0_prior = np.array(train[train.two_year_recid == 0].two_year_recid).shape[0] / train.shape[0]
        self.y_1_prior = 1 - self.y_0_prior

    def predict(self, test):
        test = test[:, 1:]

        pred = []
        for row in test:
            cond_0 = 1
            cond_1 = 1
            for ind in range(len(row)):
                cond_0 *= (list(self.y_0[:, ind]).count(row[ind]) + 1) / len(self.y_0)
                cond_1 *= (list(self.y_1[:, ind]).count(row[ind]) + 1) / len(self.y_1)
            if cond_0 * self.y_0_prior >= cond_1 * self.y_1_prior:
                pred.append(0)
            else:
                pred.append(1)

        return pred

# ## Run and calculate accuracy rate
gau_confusion_mat = metrics.confusion_matrix(y_act,gau_pred)
gau_score = metrics.accuracy_score(y_act, gau_pred)

knn_confusion_mat = metrics.confusion_matrix(y_act,knn_pred)
knn_score = metrics.accuracy_score(y_act, knn_pred)

nb_confusion_mat = metrics.confusion_matrix(y_act,nb_pred)
nb_score = metrics.accuracy_score(y_act, nb_pred)
print(gau_score,'        ',knn_score,'        ',nb_score,'\n')
