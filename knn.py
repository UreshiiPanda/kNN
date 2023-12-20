import numpy as np


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.y_pred = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_euclid(self, X_test):
        for p in X_test:
            # get distances to p from each vector in X_train
            # dist_time = time.time()
            euclidean_distances = np.linalg.norm(self.X_train-p, axis=1)
            # get the indices of the distance arr which are the k min distances 
            # argpartition() uses QuickSelect under the hood
            k_indices = np.argpartition(euclidean_distances, self.k)[:self.k]
            # now we just grab these indices from the y_train targets
            # the indices should be matched btw the inputs: X, y
            k_nearest_neighbors = [self.y_train[i] for i in k_indices]
            # now that we have all of the k closest targets, 
            # count the occurrences of each target y-label in our neighbors, and
            # take the max of those, these values will be 0 or 1 if the input 
            # data has been binarized
            pred = np.bincount(k_nearest_neighbors).argmax()
            # we have a new prediction, add it to our prediction arr
            self.y_pred.append(pred)
        return np.array(self.y_pred)

    def predict_manhat(self, X_test):
        for p in X_test:
            # get distances to p from each vector in X_train
            manhattan_distances = np.linalg.norm(self.X_train-p, ord=1, axis=1)
            # get the indices of the distance arr which are the k min distances 
            k_indices = np.argpartition(manhattan_distances, self.k)[:self.k]
            # now we just grab these indices from the y_train targets
            # the indices should be matched btw the inputs: X, y
            k_nearest_neighbors = [self.y_train[i] for i in k_indices]
            # now that we have all of the k closest targets, 
            # count the occurrences of each target label in our neighbors, and
            # take the max of those, these values will be 0 or 1 if the input
            # data has been binarized
            pred = np.bincount(k_nearest_neighbors).argmax()
            # we have a new prediction, add it to our prediction arr
            self.y_pred.append(pred)
        return np.array(self.y_pred)
            

