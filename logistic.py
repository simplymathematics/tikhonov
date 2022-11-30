import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_moons
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from linear import LinearTikhonovClassifier



class LogisticTikhonovClassifier(LinearTikhonovClassifier):
    def __init__(self, scale):
        self.scale = scale


    def loss(self, x:np.ndarray, y:np.ndarray) -> float:
        """ Calculates the loss function for the logistic regression model. 
            The regularizaiton paramater is scaled by a parameter set during model initialization (self.scale).
        Args:
            x (np.ndarray): The input data.
            y (np.ndarray): The target data.

        Returns:
            float: The loss value.
        """
        y_hat = self.sigmoid(x @ self.coef_ + self.intercept_)
        loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        reg = .5 * np.sum(self.coef_ ** 2)
        return loss + reg * self.scale
    
    def gradient(self, x, y):
        y_hat = self.sigmoid(x @ self.coef_ + self.intercept_)
        reg = self.coef_ * self.scale
        w = (y_hat - y) @ x + reg
        b = (y_hat - y)
        return (w, b)
    
    

    def sigmoid(self, z):
        return np.divide(1, 1 + np.exp(-z))

if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_classes=2, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticTikhonovClassifier(scale=0)
    model = model.fit(X_train, y_train)
    grad1 = model.check_grad(X_train, y_train)
    grad2 = model.gradient(X_train, y_train)
    print(f"Shape of Tommy's gradient: {grad1.shape}")
    print(f"Shape of My gradient: {grad2[1].shape}")
    input("Press Enter to continue...")
    for g1, g2 in zip(grad1, grad2[0]):
        print(g1.shape)
        print(g2.shape)
        input("Press Enter to continue...")
        print(g1-g2)
    # from scipy.optimize import check_grad
    # model = model.fit(X_train[0, :], y_train[0])
    # print(check_grad(model.loss, model.gradient, X_train[0, :], y_train[0]))
    # predictions = model.predict(X_test)
    # score = model.score(y_test, predictions)
    # print(score)
    