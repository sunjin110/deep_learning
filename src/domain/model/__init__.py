import numpy as np

# 2乗和誤差
# yはニューラルネットワークの出力(ソフトマックス関数出力), tは教師データ(one-hot-表現)
# どれくらいの性能の悪さを出すかのやつ
def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
