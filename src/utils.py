import numpy as np
import matplotlib.pyplot as plt


def create_data_2(n):
    omega = np.random.randn(1)[0]
    noise = 0.5*np.random.randn(n)
    x = np.random.randn(n, 2)
    y = 2*(omega * x[:, 0] + x[:, 1] + noise > 0) - 1
    return x, y


def plot_data(weights, labels, x, y, savename=None):
    fig, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1], c=y)
    x_points = np.arange(x[:, 0].min(), x[:, 0].max(), 0.1)
    for i in range(len(weights)):
        ax.plot(
            x_points,
            -weights[i][0]*x_points/weights[i][1],
            label=labels[i],
        )
    ax.axis([
        x[:, 0].min()*1.1, x[:, 0].max()*1.1,
        x[:, 1].min()*1.1, x[:, 1].max()*1.1
    ])
    ax.legend()
    ax.set(
        xlabel='Dimension 1',
        ylabel='Dimension 2'
    )
    fig.tight_layout()
    if savename is not None:
        fig.savefig('./{}.pdf'.format(savename))
    plt.show()


def plot_loss(loss_hists, labels, savename=None):
    fig, ax = plt.subplots()
    for i in range(len(loss_hists)):
        ax.scatter(
            np.arange(0, len(loss_hists[i])), loss_hists[i], label=labels[i]
        )
    ax.legend()
    ax.set(
        xlabel='Epoch',
        ylabel='Loss function value'
    )
    fig.tight_layout()
    if savename is not None:
        fig.savefig('./{}.pdf'.format(savename))
    plt.show()
