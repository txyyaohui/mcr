import numpy as np
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def neg_hscore(f, g):
    """pytorch version of maximal correlation"""
    f0 = f - torch.mean(f, 0)
    g0 = g - torch.mean(g, 0)
    corr = torch.mean(torch.sum(f0 * g0, 1))
    cov_f = torch.mm(torch.t(f0), f0) / (f0.size()[0] - 1.)
    cov_g = torch.mm(torch.t(g0), g0) / (g0.size()[0] - 1.)
    return - corr + torch.trace(torch.mm(cov_f, cov_g)) / 2.

def neg_hscore(x):
    """
    negative hscore calculation
    """
    f = x[0]
    g = x[1]
    f0 = f - K.mean(f, axis = 0)
    g0 = g - K.mean(g, axis = 0)
    corr = K.mean(K.sum(f0 * g0, axis = 1))
    cov_f = K.dot(K.transpose(f0), f0) / K.cast(K.shape(f0)[0] - 1, dtype = 'float32')
    cov_g = K.dot(K.transpose(g0), g0) / K.cast(K.shape(g0)[0] - 1, dtype = 'float32')
    return - corr + K.sum(cov_f * cov_g) / 2

def feature_f(input_x, fdim):
    conv1 = Conv2D(32, kernel_size=(3, 3),
                 activation='relu')(input_x)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    f = Dropout(0.25)(pool)
    f = Flatten()(f)
    # f = Dense(128, activation='relu')(f)
    # f = Dropout(0.5)(f)
    f = Dense(fdim)(f)
    # f = Dense(fdim, activation='relu')(f)
    return f

def mcr_paras(f_train, y_train):
    """
    compute the parameters of mcf
    Input:
       f_train: feature: n x k
       y_train: onehot label: n x |cY|
    
       py, mu_f, g_mat
       maximal correlation regression:
    """
    mu_f = np.mean(f_train, axis = 0) # E[f]
    f_train = f_train - mu_f
    py = np.mean(y_train, axis = 0)
    cov_f = np.cov(f_train.T)
    cy = y_train.shape[1] # cardinality of y
    k = f_train.shape[1]
    g_mat = np.zeros([cy, k])
    for i in range(cy):
        index = y_train[:, i] == 1
        cond_E_f = np.mean(f_train[index, :], axis = 0) # conditional expection E[f|Y = i]
        g_mat[i] = np.linalg.pinv(cov_f) @ cond_E_f
        # np.linalg.solve(cov_f, cond_E_f) # inv(cov(f)) * E[f|Y = i]
    return py, mu_f, g_mat

def mcr_pred(py, f_test, mu_f, g_mat):
    pygx = py * (1 + (f_test - mu_f) @ g_mat.T)
    return pygx


def pca_2(f):
    """
    compute the first 2 principal components
    """
    k = f.shape[1]
    f = f - np.mean(f)
    for i in range(k):
        f[:, i] = f[:, i] / np.std(f[:, i])
    cov_f = np.cov(f.T)
    U, _, _ = np.linalg.svd(cov_f)
    u0 = U[:, [0]]
    u1 = U[:, [1]]
    f0 = f @ u0
    f1 = f @ u1
    return f0, f1
