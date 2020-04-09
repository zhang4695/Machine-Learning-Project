# -*- coding: utf-8 -*-

"""
Part of the code and reasoning is based on the following articles:
    http://krasserm.github.io/2018/03/19/gaussian-processes/
    https://peterroelants.github.io/posts/gaussian-process-kernels/
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[], p=None):
    X = X.ravel()
    mu = mu.ravel()

    if p is not None:
        for i in range(-10, 10, 3):
            plt.axvline(x=i, ymin=0.15, ymax=0.85, ls='--', lw=2)

    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i + 1}')

    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
        for i in X_train:
            plt.axvline(x=i, ymin=0.05, ymax=0.95, ls='--', lw=2)

    plt.plot(X, mu, label='Mean')
    plt.legend(loc=4, fontsize=15)
    fig = plt.gcf()
    fig.set_size_inches((20, 10))
    plt.xlim((-10.9, 10.9))
    plt.ylim((-3.9, 3.9))
    plt.title('Random Draws from Gaussian Process', fontsize=15)
    plt.show()

def plot_mean(mu1, mu2, X, X_train=None, y_train=None, p=None):
    mu1 = mu1.ravel()
    mu2 = mu2.ravel()
    plt.plot(X, mu1, lw=2, ls=':', label='Mean_Squared Exponential Kernel')
    plt.plot(X, mu2, lw=2, ls='--', label='Mean_Periodic Kernel')
    if X_train is not None:
        plt.plot(X_train, y_train, 'rx')
        for i in X_train:
            plt.axvline(x=i, ymin=0.05, ymax=0.95, ls='--', lw=2, color='g')

    plt.legend(loc=4, fontsize=15)
    fig = plt.gcf()
    fig.set_size_inches((20, 10))
    plt.xlim((-10.9, 10.9))
    plt.ylim((-3.9, 3.9))
    plt.title('Mean Function of 2 Kernels', fontsize=15)
    plt.show()

def kernel_gau(X1,X2,h=5):
    diff = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return np.exp((-1/h)*diff)

def kernel_per(X1,X2,h=5,p=3):
    diff = (-2/h)*np.sin(dist.cdist(X1,X2,'cityblock')*np.pi*(1/p))**2
    return np.exp(diff)

def post_pred(x_test, x_train, y_train, method=1):
    if method == 1:
        cov_train = kernel_gau(x_train, x_train)
        cov_cross = kernel_gau(x_test, x_train)
        cov_test = kernel_gau(x_test, x_test)
    elif method == 2:
        cov_train = kernel_per(x_train, x_train, 2, 4)
        cov_cross = kernel_per(x_test, x_train, 2, 4)
        cov_test = kernel_per(x_test, x_test, 2, 4)

    b = cov_cross.dot(np.linalg.inv(cov_train)).dot(y_train)
    A = cov_test - cov_cross.dot(np.linalg.inv(cov_train)).dot(cov_cross.T)

    return b, A

# ---------------------------------------------------------------
X = np.arange(-10, 10, 0.04).reshape(-1, 1)
train = np.array(([-6,3],[0,-2],[7,2]))
X_train = train[:,0].reshape(-1,1)
y_train = train[:,1].reshape(-1,1)

##Squared Exponential Kernel,500-prior,zero-cov
mu = np.zeros(X.shape)
cov = kernel_gau(X, X)

samples = np.random.multivariate_normal(mu.ravel(), cov, 4)

plot_gp(mu, cov, X, samples=samples)

##Periodic Kernel,500-prior,zero-cov
mu = np.zeros(X.shape)
cov = kernel_per(X, X)

samples = np.random.multivariate_normal(mu.ravel(), cov, 4)

plot_gp(mu, cov, X, samples=samples, p=3)

##Squared Exponential Kernel,posterior, 3 training points, 500 testing points
b_gau, A_gau = post_pred(X,X_train, y_train,1)

samples = np.random.multivariate_normal(b_gau.ravel(), A_gau, 4)

plot_gp(b_gau, A_gau, X, X_train, y_train, samples=samples)

##Periodic Kernel, 3 training points, 500 testing points
b_per, A_per = post_pred(X,X_train, y_train,2)

samples = np.random.multivariate_normal(b_per.ravel(), A_per, 4)

plot_gp(b_per, A_per, X, X_train, y_train, samples=samples)

##Mean function plot
plot_mean(b_gau,b_per,X,X_train, y_train)