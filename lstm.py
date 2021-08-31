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

# 设定为自增长
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# KTF.set_session(session)


def trainModel(train_X, train_Y):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    '''
    model = Sequential()  # 定义一个堆叠的顺序模型

    model.add(LSTM(128,input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(128,return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        train_Y.shape[1]))
    model.add(Activation("relu"))

    model.compile(loss='mse', optimizer='adam', metrics=['acc'])  # 配置模型学习过程
    return model


def create_dataset(data, n_predictions):
    '''
    对数据进行处理
    '''

    dim = data.shape[1]
    train_X, train_Y = [], []
    all_data_x = []
    all_data_y = []
    for i in range(data.shape[0] - 6):
        all_data_x.append(data[i:i+6,:])
        all_data_y.append(data[i+6,:])

    all_data_x = np.array(all_data_x, dtype='float64')
    all_data_y = np.array(all_data_y, dtype='float64')
    # 训练集和测试集 9：1
    train_index = int(len(all_data_x)/10)
    train_X = all_data_x[:train_index*9]
    train_Y = all_data_y[:train_index*9]

    test_X = all_data_x[train_index * 9:]
    test_Y = all_data_y[train_index * 9:]

    # 制作打乱顺序的下标
    indices = [ i for i in range(len(train_X))]  # indices = the number of images in the source data set
    np.random.shuffle(indices)
    # 把随机打乱的下标塞到训练数据里面
    train_X = train_X[indices]
    train_Y = train_Y[indices]

    # np.random.shuffle(train)

    # for i in range(data.shape[0] - (n_predictions - n_next - 1)* data_set_num ):
    #     a = data[i:(i + n_predictions), :]
    #     train_X.append(a)
    #     tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
    #     b = []
    #     for j in range(len(tempb)):
    #         for k in range(dim):
    #             b.append(tempb[j, k])
    #     train_Y.append(b)


    # train_X = np.array(train_X, dtype='float64')
    # train_Y = np.array(train_Y, dtype='float64')
    #
    # test_X, test_Y = [], []
    # begin = data.shape[0] - (n_predictions - n_next - 1) * data_set_num + 1
    # for i in range((n_predictions - n_next - 1)* data_set_num):
    #     a = data[begin + i: begin + (i + n_predictions), :]
    #     test_X.append(a)
    #
    # tempb = data[begin:, :]
    # b = []
    # for j in range(len(tempb)):
    #     for k in range(dim):
    #         b.append(tempb[j, k])
    # test_Y.append(b)
    # test_X = np.array(test_X, dtype='float64')
    # test_Y = np.array(test_Y, dtype='float64')

    return train_X, train_Y, test_X, test_Y


def NormalizeMult(data, set_range):
    '''
    返回归一化后的数据和最大最小值
    '''
    normalize = np.arange(2 * data.shape[1], dtype='float64')
    normalize = normalize.reshape(data.shape[1], 2)

    for i in range(0, data.shape[1]):
        if set_range == True:
            list = data[:, i]
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
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta

    return data, normalize


if __name__ == "__main__":

    # set_range = False

    series_idx = ['lat', 'lon']
    # 读入时间序列的文件数据
    data = pd.read_csv('4B04A0.csv', sep=',').values

    import pyproj

    p_exchange = pyproj.Proj(init='epsg:26915')
    for each in data:
        print(each, p_exchange(each[0], each[1]))

    print("样本数：{0}，维度：{1}".format(data.shape[0], data.shape[1]))
    # print(data)

    # 画样本数据库
    # plt.scatter(data[:, 1], data[:, 0], c='b', marker='o', label='traj_A')
    # plt.legend(loc='upper left')
    # plt.grid()
    # plt.show()

    # 归一化
    set_range = False
    data, normalize = NormalizeMult(data, set_range)
    # print(normalize)

    # 生成训练数据
    train_num = 6
    # per_num = 1
    train_X, train_Y, test_X, test_Y = create_dataset(data, train_num)
    print("x\n", train_X.shape)
    print("y\n", train_Y.shape)

    # 得到模型
    model = trainModel(train_X, train_Y)

    model.summary()
    # 训练模型
    model.fit(train_X, train_Y, epochs=10, batch_size=64, verbose=1, validation_data=(test_X, test_Y))

    loss, acc = model.evaluate(test_X, test_Y, verbose=2)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))

    y_hat = model.predict(train_X)

    # 保存模型
    np.save("./traj_model_trueNorm.npy", normalize)
    model.save("./traj_model_240_3layers_altitude.h5")