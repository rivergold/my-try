import sympy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt


def draw_ExponentialLR():
    lr = 0.01
    gamma = 0.9
    epoch = sp.symbols('epoch')
    lr_fn = lr * gamma**epoch
    # fig = plt.figure()
    p = sp.plotting.plot(lr_fn, xlim=(0, 500), ylim=(1e-10, 0.01))
    # fig.savefig('./debug.jpg')
    # plt.savefig('./debug.jpg')
    p.save('./debug.jpg')


if __name__ == '__main__':
    draw_ExponentialLR()