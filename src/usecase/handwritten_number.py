# 3.6手書き数字認識
# forward propagation

# イメージは: IMG_0974.jpeg

import pickle
import os

import numpy as np
from domain.activation_function import sigmoid, softmax
from infrastructure.mnist import load_mnist

# get_test_data 学習済networkがどれほどの精度が出るかテストするためのデータを取得
def get_test_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# 学習済networkを取得
def init_network():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data/sample_weight.pkl")
    with open(file_path, 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

def run():
    # テスト用データを取得
    x, t = get_test_data()

    # 学習済データを取得  
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y) # 最も確率の固い要素のインデックスを取得
        print("p is ", str(p) + " t["+str(i)+"] is ", str(t[i]))
        if p == t[i]:
            accuracy_cnt += 1
    
    print(str(len(x)) + "個中" + str(accuracy_cnt) + "個正解")
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


# batch処理で実行する P79
def batch_run():
    x, t = get_test_data()

    network = init_network()

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
