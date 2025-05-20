import numpy as np
from infrastructure.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def main():

    index = 9879

    # (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=False, normalize=False, one_hot_label=False)
    print("x_train[@](訓練画像) is ")
    print(x_train[index])
    print("t_train[@]（訓練ラベル is ")
    print(t_train[index])

    print("x_test[@](テスト画像) is")
    print(x_test[index])

    print("t_test[@](テストラベル)")
    print(t_test[index])

    # print(img.shape)
    # img = img.reshape(28, 28)
    # print(img.shape)
    # img_show(x_test[index].reshape(28, 28))
    img_show(x_test[index].reshape(28, 28))

if __name__ == '__main__':
    main()
