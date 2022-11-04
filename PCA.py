"""
Unsupervised learning method
Use to reduce dimensionality
Goal:
PCA finds a new set of dimensions such that all the dimensions are orthogonal
(and hence linearly independent) and ranked according to the variance of the
data along them.
"""
import numpy as np


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, functions need samples as columns
        cov = np.cov(X.T)

        # eigenvectors, eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # eigenvectors v = [:,i] column vector, transpose this for easier calculations
        eigenvectors = eigenvectors.T

        # sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        # projects data
        X = X - self.mean
        return np.dot(X, self.components.T)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn import datasets

    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print('Shape of X:', X.shape)
    print('Shape of transformed X:', X_projected.shape)

    x1 = X_projected[:,0]
    x2 = X_projected[:, 1]

    plt.scatter(x1,x2,c=y,edgecolors='none',alpha=.8,cmap=plt.cm.get_cmap('viridis',3))
    plt.xlabel('Principal Component 1')
    plt.xlabel('Principal Component 2')
    plt.colorbar()
    plt.show()
