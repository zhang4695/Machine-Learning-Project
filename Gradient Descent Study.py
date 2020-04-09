import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import timeit
import random

eps = 1e-7


# ##make sure X (m *n) with m observations and n features and shape(m,n)
# ##make sure Y is one dimensional with shape(m,1)
# ##make sure input w has shape(n,1) with n feature weights
# ## create_data is built according to the paper here: https://towardsdatascience.com/logistic-regression-using-gradient-descent-optimizer-in-python-485148bd3ff2
def create_data():
    iris = pd.read_csv('iris.csv', header=None, names=('Sepal L', 'Sepal W', 'Petal L', 'Petal W', 'Class'))
    iris = iris.head(100)
    # iris.head()

    # Fitting it to our dataset
    LE = preprocessing.LabelEncoder()
    iris.Class = LE.fit_transform(iris.Class)
    iris.head()

    # Importing StandardScaler from scikit-learn
    sst = StandardScaler()
    # Standardizing the data apart from the Class column
    iris_scaled = pd.DataFrame(sst.fit_transform(iris.drop('Class', axis=1)))
    # Adding the Class column back to the DataFrame
    iris_scaled['Class'] = iris.Class

    x = np.array(iris_scaled.iloc[:, :4]).reshape((100, 4))
    y = np.array(iris_scaled.iloc[:, 4]).reshape((100, 1))
    w = np.random.normal(size=(np.shape(x)[1], 1))
    return x, y, w


def plot_function(x, y, w, Error, iteration, name, stop_p):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(Error)), Error)
    if stop_p < len(Error) - 1:
        ax.plot(stop_p, [0], '-r', marker='o', markersize=6)
    ax.set_title(name)
    plt.show()


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


# #part of the calculating function is quoted from https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))


def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(np.dot(x, theta))


def cal_cost(theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost


def gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, probability(theta, x) - y)


def get_minibatch(X, y, minibatch_size):
    minibatches = []
    idx = [i for i in range(len(X))]
    np.random.shuffle(idx)
    X_shuffle = X[idx]
    y_shuffle = y[idx]

    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X_shuffle[i:i + minibatch_size]
        y_mini = y_shuffle[i:i + minibatch_size]
        minibatches.append((X_mini, y_mini))
    return minibatches


def GD(X, y, theta, learning_rate=0.1, iteration=1000, tol=0.3):
    m = y.shape[0]
    cost_history = np.zeros(iteration)
    for it in range(iteration):
        theta -= learning_rate * gradient(theta, X, y)
        cost_history[it] = cal_cost(theta, X, y)
        if cost_history[it] <= tol:
            return theta, cost_history, it
    return theta, cost_history, iteration - 1


def SGD(X, y, theta, batch_size=1, learning_rate=0.01, iteration=100, tol=0.3):
    cost_history = np.zeros(iteration)
    for it in range(iteration):
        cost = 0.0
        batch = get_minibatch(X, y, batch_size)
        for data in batch:
            X_test = data[0]
            Y_test = data[1]
            grad = gradient(theta, X_test, Y_test)
            theta[:] -= learning_rate * grad
        cost_history[it] = cal_cost(theta, X, y)
        if cost_history[it] <= tol:
            return theta, cost_history, it
    return theta, cost_history, iteration - 1


def SGDM(X, y, theta, batch_size=10, gama=0.8, learning_rate=0.09, iteration=100, tol=0.3):
    cost_history = np.zeros(iteration)
    v = np.zeros((theta.shape[0], 1))
    for it in range(iteration):
        cost = 0.0
        batch = get_minibatch(X, y, batch_size)
        for data in batch:
            X_test = data[0]
            Y_test = data[1]
            grad = gradient(theta, X_test, Y_test)
            v[:] = gama * v + learning_rate * grad
            theta[:] -= v
        cost_history[it] = cal_cost(theta, X, y)
        if cost_history[it] <= tol:
            return theta, cost_history, it
    return theta, cost_history, iteration - 1


def AD_SGD(X, y, theta, batch_size=10, learning_rate=0.9, iteration=100, tol=0.3):
    cost_history = np.zeros(iteration)
    sq_sum = np.zeros((theta.shape[0], 1))
    for it in range(iteration):
        cost = 0.0
        batch = get_minibatch(X, y, batch_size)
        for data in batch:
            X_test = data[0]
            Y_test = data[1]
            prediction = np.dot(X_test, theta)
            grad = gradient(theta, X_test, Y_test)
            sq_sum[:] += np.power(grad, 2)
            theta[:] -= learning_rate / (np.sqrt(sq_sum + eps)) * grad
        cost_history[it] = cal_cost(theta, X, y)
        if cost_history[it] <= tol:
            return theta, cost_history, it
    return theta, cost_history, iteration - 1


def RMS(X, y, theta, batch_size=10, gama=0.9, learning_rate=0.01, iteration=100, tol=0.3):
    cost_history = np.zeros(iteration)
    v = np.zeros((theta.shape[0], 1))
    for it in range(iteration):
        cost = 0.0
        batch = get_minibatch(X, y, batch_size)
        for data in batch:
            X_test = data[0]
            Y_test = data[1]
            grad = gradient(theta, X_test, Y_test)
            v = gama * v + (1 - gama) * grad ** 2
            theta = theta - (learning_rate / (np.sqrt(v) + eps)) * grad
        cost_history[it] = cal_cost(theta, X, y)
        if cost_history[it] <= tol:
            return theta, cost_history, it
    return theta, cost_history, iteration - 1


def ADE(X, y, theta, batch_size=10, gama=0.9999, iteration=100, tol=0.3):
    cost_history = np.zeros(iteration)
    v_g = np.ones((theta.shape[0], 1)) * 0.1
    v_t = np.ones((theta.shape[0], 1)) * 2

    for it in range(iteration):
        cost = 0.0
        batch = get_minibatch(X, y, batch_size)
        for data in batch:
            X_test = data[0]
            Y_test = data[1]
            grad = gradient(theta, X_test, Y_test)
            v_g[:] = gama * v_g + (1 - gama) * grad ** 2
            change_theta = (np.sqrt(v_t + eps) / (np.sqrt(v_g) + eps)) * grad
            v_t[:] = gama * v_t + (1 - gama) * (change_theta ** 2)
            theta[:] -= change_theta
        cost_history[it] = cal_cost(theta, X, y)
        if cost_history[it] <= tol:
            return theta, cost_history, it
    return theta, cost_history, iteration - 1


def ADAM(X, y, theta, batch_size=10, gama_1=0.9, gama_2=0.999, learning_rate=0.2, iteration=100, tol=0.3):
    cost_history = np.zeros(iteration)
    t_p = 1
    v_m = np.zeros((theta.shape[0], 1))
    v_sqr = np.zeros((theta.shape[0], 1))
    for it in range(iteration):
        t_p += 1
        cost = 0.0
        batch = get_minibatch(X, y, batch_size)
        for data in batch:
            X_test = data[0]
            Y_test = data[1]
            grad = gradient(theta, X_test, Y_test)
            v_m[:] = gama_1 * v_m + (1 - gama_1) * grad
            v_m_2 = v_m / (1 - np.power(gama_1, t_p))
            v_sqr[:] = gama_2 * v_sqr + (1 - gama_2) * grad ** 2
            v_sqr_2 = v_sqr / (1 - np.power(gama_2, t_p))
            theta[:] -= (learning_rate / (np.sqrt(v_sqr_2) + eps)) * v_m_2
        cost_history[it] = cal_cost(theta, X, y)
        if cost_history[it] <= tol:
            return theta, cost_history, it
    return theta, cost_history, iteration - 1


def output_(nb=2000, tolerance=0.0005):
    x, y, w = create_data()
    gd_th, gd_ch, gd_idx = GD(x.copy(), y, w.copy(), iteration=nb, tol=tolerance)
    sgd_th, sgd_ch, sgd_idx = SGD(x.copy(), y, w.copy(), iteration=nb, tol=tolerance)
    sgdm_th, sgdm_ch, sgdm_idx = SGDM(x.copy(), y, w.copy(), iteration=nb, tol=tolerance)
    ad_sgd_th, ad_sgd_ch, ad_sgd_idx = AD_SGD(x.copy(), y, w.copy(), iteration=nb, tol=tolerance)
    rms_th, rms_ch, rms_idx = RMS(x.copy(), y, w.copy(), iteration=nb, tol=tolerance)
    ade_th, ade_ch, ade_idx = ADE(x.copy(), y, w.copy(), iteration=nb, tol=tolerance)
    adam_th, adam_ch, adam_idx = ADAM(x.copy(), y, w.copy(), iteration=nb, tol=tolerance)
    plot_function(x.copy(), y, gd_th, gd_ch, nb, 'Batch Gradient Descent', gd_idx)
    plot_function(x.copy(), y, sgd_th, sgd_ch, nb, 'Stochastic Gradient Descent', sgd_idx)
    plot_function(x.copy(), y, sgdm_th, sgdm_ch, nb, 'Stochastic Gradient Descent - Momentum', sgdm_idx)
    plot_function(x.copy(), y, ad_sgd_th, ad_sgd_ch, nb, 'AdaGrad', ad_sgd_idx)
    plot_function(x.copy(), y, rms_th, rms_ch, nb, 'RMSProp', rms_idx)
    plot_function(x.copy(), y, ade_th, ade_ch, nb, 'AdaDelta', ade_idx)
    plot_function(x.copy(), y, adam_th, adam_ch, nb, 'Adam', adam_idx)

    print('gd_ch:    ', gd_ch[gd_idx], '  convergence_time: ', gd_idx)
    print('sgd_ch:     ', sgd_ch[sgd_idx], '  convergence_time: ', sgd_idx)
    print('sgdm_ch:    ', sgdm_ch[sgdm_idx], '  convergence_time: ', sgdm_idx)
    print('ad_sgd_ch:    ', ad_sgd_ch[ad_sgd_idx], '  convergence_time: ', ad_sgd_idx)
    print('rms_ch:    ', rms_ch[rms_idx], '  convergence_time: ', rms_idx)
    print('ade_ch:    ', ade_ch[ade_idx], '  convergence_time: ', ade_idx)
    print('adam_ch:    ', adam_ch[adam_idx], '  convergence_time: ', adam_idx)


output_(300)