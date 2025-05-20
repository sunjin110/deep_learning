import numpy as np

def identity_function(a):
    return a

# softmax -> スコアを確率に変換するものみたい
# ただ単純な合計ではなく、スコアをもっと強調させるためにexpを利用するみたい
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # overflow対策
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a
