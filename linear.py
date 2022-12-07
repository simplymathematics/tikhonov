import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_moons
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from autograd import grad
import autograd.numpy as np

from autograd import grad
from tqdm import tqdm

class LinearTikhonovClassifier():
    def __init__(self, scale):
        self.scale=scale


    def fit(self, X, y, learning_rate = 1e-8, epochs = 1000, warm_start = False):
        d = X.shape[1]
        c = len(np.unique(y))
        self.coef_ = np.random.random((c,d)) * 1e-8 if warm_start is False else self.coef_
        self.intercept_ = np.zeros(1) * 1e-8 if warm_start is False else self.intercept_
        old_loss = 1e9
        L_w = self.coef_ * 0.0
        L_b = 0
        for i in tqdm(range(epochs), desc = f"Epochs: {epochs}. Learning Rate: {learning_rate}. Scale: {self.scale}. Warm Start: {warm_start}"):
            L_w, L_b = self.gradient(X, y, self.coef_, self.intercept_)
            # print(L_w.shape)
            # print(L_b.shape)
            # input(f"Epoch {i+1} Loss: {self.loss(X, y, self.coef_, self.intercept_)}")
            self.coef_ -= L_w * learning_rate
            self.intercept_ -= L_b * learning_rate
            y_pred = self.predict(X)
            new_loss = self.loss(X, y, self.coef_, self.intercept_)
            # print(f"Epoch {i+1} Loss: {new_loss}")
            if new_loss > old_loss:
                learning_rate /= 2
            old_loss = new_loss
        print(f"Final Loss: {self.loss(X, y, self.coef_, self.intercept_)}")
        print(f"Final Accuracy: {self.score(y, self.predict(X))}")
        print(f"Final Learning Rate: {learning_rate}")
        return self

    def loss(self,  X, y, weights, bias):
        y_hat = weights @ X.T
        y_hat = y_hat + bias
        errors = np.subtract(y_hat, y)
        squared = errors * errors
        summed = np.sum(squared)
        tikho = np.sum(weights * weights)
        tikho /= 2
        return np.mean(summed + self.scale * tikho)
    
    def gradient(self, x, y, weights, bias):
        gradL_w = grad(self.loss, 2)(x, y, weights, bias)
        gradL_b = grad(self.loss, 3)(x, y, weights, bias)
        return (gradL_w, gradL_b)
    
    def predict(self, X):
        X_dot_weights =  self.coef_ @ X.T + self.intercept_
        # print(X_dot_weights.shape)
        # print(np.argmax(X_dot_weights, axis = 0).shape)
        # print(np.argmax(X_dot_weights, axis = 1).shape)
        # print(np.sum(X_dot_weights, axis = 0).shape)
        # print(X_dot_weights[0].shape)
        # input("Inside predict")
        return [1 if x > .5 else 0 for x in X_dot_weights[0]]

    def score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
    

if __name__ == "__main__":
    samples = 10000
    X, y = make_classification(n_samples=samples, n_classes=2, n_features=10, n_informative=8, n_redundant=0, n_clusters_per_class=1, class_sep=10)
    # X, y = make_blobs(n_samples=10000, n_features=2, centers=2, cluster_std=1.0, random_state=40)
    
    print(f"No. of Samples: {samples}")
    print(f"Classes: {np.unique(y)}")
    print(f"Largest Class: {np.bincount(y).max()}")
    print(f"Smallest Class: {np.bincount(y).min()}")
    print("Test Basic Functionality")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearTikhonovClassifier(scale=1000.0)
    model = model.fit(X_train, y_train, learning_rate=1e-8, epochs=1000)
    predictions = model.predict(X_train)
    # print(f"Predictions.shape: {predictions.shape}")
    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    score = model.score(y_train, predictions)
    loss = model.loss(X_train, y_train, model.coef_, model.intercept_)
    print(f"Score: {score}, Loss: {loss}")
    print("#"*80)