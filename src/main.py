import numpy as np
import matplotlib.pyplot as plt

import usecase
import usecase.third_dimention_nural_network
import usecase.handwritten_number

def AND(x1: int, x2: int) -> bool:
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    return np.sum(w*x) + b > 0

def NAND(x1: int, x2: int) -> bool:
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    return np.sum(w*x) + b > 0

def OR(x1: int, x2: int) -> bool:
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    return np.sum(w*x) + b > 0

def XOR(x1: int, x2: int) -> bool:
    a = NAND(x1, x2)
    b = OR(x1, x2)
    return AND(a, b)

def matrix_dot():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2], [3, 4], [4, 5]])

def matrix_dot_v2():
    a = np.array([[1, 2], [4, 5], [7, 8]])
    b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    print(a.shape)
    print(b.shape)
    c = a @ b
    print(c.shape)
    print(c)

def a():
    # 行列でのドット積
    people = np.array([
        [170, 70, 22, 40], # 身長, 体重, 年齢, 月収
        [150, 50, 30, 90],
        [175, 75, 29, 90]
    ])

    values = [0.2, 0.4, -0.5, 1]

    mean = people.mean(axis=0)
    std = people.std(axis=0) # 各行の標準偏差
    print("mean is ")
    print(mean)
    print("std is")
    print(std)
    people_normalized = (people - mean) / std
    print(people_normalized @ values)

def spin_90():
    theta = np.pi / 2
    rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    v = np.array([
        [1, 2, 3], 
        [4, 5, 6], 
        [7, 8, 9]
    ])
    print(v)
    print(v.shape)

    print((rz @ v.T).T)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def neural_network_third_dimention():
    # 入力層
    x = np.array([1.0, 0.5])

    # 第1層
    w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    b1 = np.array([0.1, 0.2, 0.3])
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    print(z1)

    # 第2層
    w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    b2 = np.array([0.1, 0.2])
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    print(z2)

    # 出力層
    w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    b3 = np.array([0.1, 0.2])
    a3 = np.dot(z2, w3) + b3
    y = identity_function(a3)
    print(y)

# print("=========== AND")
# print(AND(0, 0))
# print(AND(1, 0))
# print(AND(0, 1))    
# print(AND(1, 1))

# print("=========== NAND")
# print(NAND(0, 0))
# print(NAND(1, 0))
# print(NAND(0, 1))    
# print(NAND(1, 1))

# print("=========== OR")
# print(OR(0, 0))
# print(OR(1, 0))
# print(OR(0, 1))    
# print(OR(1, 1))

# print("=========== XOR")
# print(XOR(0, 0))
# print(XOR(1, 0))
# print(XOR(0, 1))    
# print(XOR(1, 1))

# matrix_dot()
# a()
# spin_90()

# neural_network()
# neural_network_third_dimention()

# usecase.third_dimention_nural_network.run()

print("==== 3.6手書き数字認識")
usecase.handwritten_number.batch_run()