import numpy as np

from utils import plot_loss, plot_data


def create_data(n):
    omega = np.random.randn(1)[0]
    noise = 0.5*np.random.randn(n)
    x = np.random.randn(n, 2)
    y = 2*(omega * x[:, 0] + x[:, 1] + noise > 0) - 1
    return x, y


class LinLogRegression():
    def __init__(self, learning_rate, reg_constant):
        self.learning_rate = learning_rate
        self.reg_constant = reg_constant
        self.weight = np.array([0.5, 1])

    def loss(self, x, y):
        assert x.shape[0] == y.shape[0]
        loss = self.reg_constant * np.dot(self.weight, self.weight)
        loss += np.sum(np.log(
            1+np.exp(-np.multiply(y, np.dot(x, self.weight)))
        ))
        return loss

    def evaluate(self, x, y):
        assert x.shape[0] == y.shape[0]
        loss = self.loss(x, y)
        accuracy = 1 - (
            np.count_nonzero(
                y - np.where(np.dot(x, self.weight) > 0, 1, -1)
            ) / y.shape[0]
        )
        return loss, accuracy

    def logit(self, x, y):
        exp = np.exp(-np.multiply(y, np.dot(x, self.weight)))
        return exp/(1.+exp)

    def loss_gradient(self, x, y):
        assert x.shape[0] == y.shape[0]
        gradient = 2 * self.reg_constant * self.weight
        gradient -= np.dot(np.multiply(self.logit(x, y), y), x)
        return gradient


class BatchSGD(LinLogRegression):
    def fit(self, x, y):
        self.weight -= self.learning_rate * self.loss_gradient(x, y)
        return self.loss(x, y)


class Newton(LinLogRegression):
    def hessian(self, x, y):
        hessian = np.dot(
            x.transpose(),
            np.dot(np.diag(self.logit(x, y)*(1 - self.logit(x, y))), x)
        )
        hessian += 2 * self.reg_constant * np.eye(x.shape[1])
        return hessian

    def fit(self, x, y):
        self.weight -= self.learning_rate * np.dot(
            self.hessian(x, y), self.loss_gradient(x, y))
        return self.loss(x, y)


if __name__ == '__main__':
    n = 200
    batch_n = 16
    epochs = 500
    learning_rate = 0.003
    reg_constant = 1
    x, y = create_data(n)
    sgd = BatchSGD(learning_rate, reg_constant)
    newton = Newton(learning_rate, reg_constant)
    plot_data(sgd.weight, newton.weight, x, y, 'weights_data_before')
    sgd_loss_hist = []
    newton_loss_hist = []
    for epoch in range(epochs):
        i = 0
        while i < n:
            newton.fit(x[i:min(i+batch_n, n)], y[i:min(i+batch_n, n)])
            sgd.fit(x[i:min(i+batch_n, n)], y[i:min(i+batch_n, n)])
            i += batch_n
        loss, accuracy = sgd.evaluate(x, y)
        sgd_loss_hist.append(loss)
        loss, accuracy = newton.evaluate(x, y)
        newton_loss_hist.append(loss)
    plot_data(sgd.weight, newton.weight, x, y, 'weights_data_after')
    plot_loss(sgd_loss_hist, newton_loss_hist)
    loss, accuracy = sgd.evaluate(x, y)
    print(
        'SGD evaluation__________\nLoss\t: {:2.4f}\nAccuracy: {:2.4f}'.format(
            loss, accuracy)
    )
    loss, accuracy = newton.evaluate(x, y)
    print()
    print(
        'Newton evaluation_______\nLoss\t: {:2.4f}\nAccuracy: {:2.4f}'.format(
            loss, accuracy)
    )
