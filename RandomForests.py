from DecisionTree import DecisionTree
import numpy as np
from collections import Counter


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(self.min_samples_split, self.max_depth, self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_lable(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # [tree1: [first_sample,0,1,last_sample], tree2: [first_sample,1,1,last_sample] ...]
        # But we want: [first_sample: [tree1, 1, 0, last_tree], second_sample: [tree1, 1, 0, last_tree]]
        tree_preds = np.swapaxes(predictions, axis1=0, axis2=1)
        predictions = np.array([self._most_common_lable(pred) for pred in tree_preds])
        return predictions


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)


    def accuracy(y_true, y_pred):
        return np.sum(y_pred == y_true) / len(y_true)


    clf = RandomForest()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(accuracy(y_test, predictions))
