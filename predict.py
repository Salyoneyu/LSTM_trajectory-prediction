import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import Callback
import tensorflow as tf
# import tensorflow.compat.v1.keras.backend as KTF
import pandas as pd
import os
import keras.callbacks
import matplotlib.pyplot as plt
import copy

# 设定为自增长
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# KTF.set_session(session)

def NormalizeMult(data_new, set_range):
    '''
    返回归一化后的数据和最大最小值
    '''
    normalize = np.arange(2 * data_new.shape[1], dtype='float64')
    normalize = normalize.reshape(data_new.shape[1], 2)

    for i in range(0, data_new.shape[1]):
        if set_range == True:
            list = data_new[:, i]
            listlow, listhigh = np.percentile(list, [0, 100])
        else:
            if i == 0:
                listlow = -180
                listhigh = 180
            else:
                listlow = -90
                listhigh = 90

        normalize[i, 0] = listlow
        normalize[i, 1] = listhigh

        delta = listhigh - listlow
        if delta != 0:
            for j in range(0, data_new.shape[0]):
                data_new[j, i] = (data_new[j, i] - listlow) / delta
    normalize = np.array([[-180, 180.], [-90, 90.]])
    return data_new, normalize


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()


def create_dataset(data, n_predictions, n_next):
    '''
    对数据进行处理
    '''
    dim = data.shape[1]
    train_X, train_Y = [], []
    for i in range(data.shape[0] - n_predictions - n_next - 1):
        a = data[i:(i + n_predictions), :]
        train_X.append(a)
        tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
        b = []
        for j in range(len(tempb)):
            for k in range(dim):
                b.append(tempb[j, k])
        train_Y.append(b)
    train_X = np.array(train_X, dtype='float64')
    train_Y = np.array(train_Y, dtype='float64')

    test_X, test_Y = [], []
    i = data.shape[0] - n_predictions - n_next - 1
    a = data[i:(i + n_predictions), :]
    test_X.append(a)
    tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
    b = []
    for j in range(len(tempb)):
        for k in range(dim):
            b.append(tempb[j, k])
    test_Y.append(b)
    test_X = np.array(test_X, dtype='float64')
    test_Y = np.array(test_Y, dtype='float64')

    return train_X, train_Y, test_X, test_Y

def reshape_y_hat(y_hat, dim):
    re_y = []
    i = 0
    while i < len(y_hat):
        tmp = []
        for j in range(dim):
            tmp.append(y_hat[i + j])
        i = i + dim
        re_y.append(tmp)
    re_y = np.array(re_y, dtype='float64')
    return re_y


# 多维反归一化
def FNormalizeMult(data, normalize):
    data = np.array(data, dtype='float64')
    # 列
    for i in range(0, data.shape[1]):
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        print("listlow, listhigh, delta", listlow, listhigh, delta)
        # 行
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = data[j, i] * delta + listlow

    return data


# 使用训练数据的归一化
def NormalizeMultUseData(data, normalize):
    for i in range(0, data.shape[1]):

        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow

        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta

    return data


from math import sin, asin, cos, radians, fabs, sqrt

EARTH_RADIUS = 6371  # 地球平均半径，6371km


# 计算两个经纬度之间的直线距离
def hav(theta):
    s = sin(theta / 2)
    return s * s


def get_distance_hav(lat0, lng0, lat1, lng1):
    # "用haversine公式计算球面两点间的距离。"
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))
    return distance


if __name__ == '__main__':




    test_num = 6
    per_num = 1
    # series_idx = ['lat', 'lon', 'altitude']
    test_move_loc = 0
    data_all = pd.read_csv('predict.csv', sep=',').values.tolist()
    # data_all = data_all.loc[np.arange(len(data_all) - test_num - per_num - test_move_loc,
    #                                   len(data_all) - test_move_loc).tolist(), series_idx].values
    # print(data_all)
    data_all = np.array(data_all)
    data_all.dtype = 'float64'

    # 生成训练数据
    train_num = 6
    per_num = 1
    data_new = copy.deepcopy(data_all[:-per_num, :])
    # #归一化
    data, normalize = NormalizeMult(data_new, False)
    train_X, train_Y, test_X, test_Y = create_dataset(data, train_num, per_num)

    # normalize = np.load("./traj_model_trueNorm.npy")
    # data = NormalizeMultUseData(data, normalize)

    model = load_model("./traj_model_240_3layers_altitude.h5")

    y = data_all[-per_num:, :]
    x = data_all[-train_num-per_num:-per_num, :]
    x = x.reshape(1, x.shape[0], x.shape[1])
    # y_hat = model.predict(x)

    y_hat = model.predict(train_X)
    # y_hat = y_hat.reshape(y_hat.shape[1])
    # y_hat = reshape_y_hat(y_hat, y.shape[1])

    # 反归一化
    y_hat = FNormalizeMult(y_hat, normalize)
    train_Y = FNormalizeMult(train_Y, normalize)
    print("predict: {0}\ntrue：{1}".format(y_hat, train_Y))
    # print('预测均方误差：', mse(y_hat, y))

    print('预测均方误差：', mse(y_hat[-1, :], y[-1, :]))
    # print('预测直线距离：{:.4f} KM'.format(get_distance_hav(y_hat[0, 0], y_hat[0, 1], y[0, 0], y[0, 1])))
    print('预测直线距离：{:.4f} KM'.format(get_distance_hav(y_hat[-1, 0], y_hat[-1, 1], y[-1, 0], y[-1, 1])))
    #print('预测高度差：{:.4f} M'.format((y_hat[-1, 2] - y[-1, 2]) * 0.3047999995367))  # 1 feet = 0.3047999995367 m

    # 画测试样本数据库
    p1 = plt.scatter(y_hat[:, 1], y_hat[:, 0], c='r', marker='o', label='pre')
    p2 = plt.scatter(train_Y[:, 1], train_Y[:, 0], c='g', marker='o', label='pre_true')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()