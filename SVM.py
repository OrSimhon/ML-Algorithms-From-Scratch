import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape  # rows, cols

        # Initialize
        self.w = np.zeros(n_features)
        self.b = 0

        # Update step
        for _ in range(self.n_iters):  # Number of passes over the entire data
            for idx, x_i in enumerate(X):  # update for each sample: MiniBatch = 1
                # Cost function derivatives
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, x):
        linear_output = np.dot(x, self.w) - self.w
        return np.sign(linear_output)


# Testing
if __name__ == '__main__':
    # imports
    import matplotlib.pyplot as plt
    from sklearn import datasets

    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)

    clf = SVM()
    clf.fit(X, y)

    print(clf.w, clf.b)


    def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

        x0_1 = np.min(X[:, 0])
        x0_2 = np.max(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')

        x1_min = np.min(X[:, 1])
        x1_max = np.max(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()


    visualize_svm()
