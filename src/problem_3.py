import numpy as np

from utils import create_data_2, plot_loss


def project(vector):
    out = np.where(vector > 1, 1, vector)
    out = np.where(out < 0, 0, out)
    return out


class DualSVM():
    def __init__(self, learning_rate, reg_constant, n_pts):
        self.learning_rate = learning_rate
        self.reg_constant = reg_constant
        self.a = np.random.random((n_pts,))

    def dual_score(self, K):
        score = 0
        score -= 1./(4.*self.reg_constant)*np.dot(
            self.a.transpose(), np.dot(K, self.a)
        )
        score += np.sum(self.a)
        return score

    def get_weight(self, X, y):
        w = np.zeros((X.shape[1],))
        for i in range(y.shape[0]):
            w += self.a[i] * y[i] * X[i]
        w *= 1./(2.*self.reg_constant)
        return w

    def primary_score(self, X, y):
        w = self.get_weight(X, y)
        score = 0
        for i in range(y.shape[0]):
            score += max(0, 1 - y[i]*np.dot(w, X[i]))
        score += self.reg_constant*np.dot(w, w)
        return score

    def fit(self, K):
        self.a = project(
            self.a
            - self.learning_rate*(
                1./(2.*self.reg_constant)*np.dot(K, self.a) - 1.
            )
        )


def main():
    n_pts = 200
    X, y = create_data_2(n_pts)
    K = np.dot(X, X.transpose())
    K = np.multiply(np.outer(y, y), K)
    epochs = 1000
    dsvm = DualSVM(0.003, 1., n_pts)
    dual_loss_hist = [dsvm.dual_score(K)]
    prim_loss_hist = [dsvm.primary_score(X, y)]
    for i in range(epochs):
        dsvm.fit(K)
        p_loss = dsvm.primary_score(X, y)
        d_loss = dsvm.dual_score(K)
        dual_loss_hist.append(d_loss)
        prim_loss_hist.append(p_loss)
    plot_loss(
        [prim_loss_hist, dual_loss_hist],
        ['Primal loss', 'Dual loss'],
        'svm_loss'
    )
    print(prim_loss_hist[-1] / dual_loss_hist[-1])


if __name__ == '__main__':
    main()
