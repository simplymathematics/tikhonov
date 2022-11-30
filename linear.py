import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_moons
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# import autograd.numpy as np

from autograd import grad
from tqdm import tqdm

class LinearTikhonovClassifier():
    def __init__(self, scale):
        self.scale=scale


    def fit(self, X, y, learning_rate = 1e-8, epochs = 1000):
        self.coef_ = np.zeros((X.shape[1])) if len(X.shape) > 1 else np.zeros(len(X))
        self.intercept_ = np.zeros(len(X)) if len(X.shape) > 1 else np.zeros(1)
        L_w = 0
        L_b = 0
        loss = 1e9
        for i in tqdm(range(epochs)):
            old_L = L_w
            old_b = L_b
            L_w, L_b = self.gradient(X, y)
            self.coef_ -= L_w * learning_rate
            self.intercept_ -= L_b * learning_rate
            y_pred = self.predict(X)
            new_loss = self.loss(y, y_pred)
            if new_loss > loss:
                self.coef_ = old_L
                self.intercept_ = old_b
                learning_rate /= 2
            loss = new_loss
        return self

    def loss(self,  y, y_hat):
        n_classes = len(np.unique(y))
        errors = y_hat - y
        squared_sum = 1/2 * np.sum(errors)**2
        tikhonov = 1/(2* n_classes) * np.sum(self.coef_)**2
        return squared_sum + tikhonov * self.scale
    
    def gradient(self, X, y):
        reg = self.coef_ * self.scale
        w = (X @ self.coef_.T + self.intercept_ - y) @  X + reg
        b = (X @ self.coef_.T+ self.intercept_ - y)
        return (w, b)
    
    def predict(self, X):
        # print(f"X: {X.shape}")
        # print(f"coef: {self.coef_.shape}")
        # print(f"intercept: {self.intercept_.shape}")
        orig = X.shape[0]
        # if X.shape[0] != self.intercept_.shape[0]:
        #     pri)
        #     height =  self.intercept_.shape[0] - orig
        #     width = X.shape[1]
        #     new = np.zeros((height, width))
        #     X = np.concatenate((X, new), axis=0)
        #     X_dot_weights = X @ self.coef_.T + self.intercept_
        #     X_dot_weights = X_dot_weights[:orig]
        # else:
        #     
        X_dot_weights = np.dot(X, self.coef_) + self.intercept_
        # print(f"X_dot_weights: {X_dot_weights.shape}")
        return np.array([1 if p > 0.5 else 0 for p in X_dot_weights])

    def score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


    def check_grad(self, x, y,  eps=1e-8):
        """Numerical approximation of the gradient.
        Parameters
        ----------
        x : numpy.ndarray, shape (p, 1)
            The point at which to evaluate the gradient.
        eps : float, optional
            Positive float. The precision of the numerical solution. Smaller is
            better, but too small may result in floating point precision
            errors. Default is 1e-4.
        """
        p = x.shape[0]
        grad = np.zeros(x.shape[0])
        for i in tqdm(range(p * 1)):
            x[i] -= eps
            yp = self.predict(x)
            loss1 = self.loss(yp, y)
            x[i] += eps
            yp = self.predict(x)
            loss2 = self.loss(yp, y)
            grad[i] = (loss2 - loss1) / (eps)
        return grad
    
#Defining the class
# class LinearTikhonovClassifier():
#     def __init__(self, scale=0.0):
#         self.scale = scale 
        
#     def fit(self, X, y,  epochs =1000 , lr = 1e-8):
#         weights = np.zeros((X.shape[1])) if len(X.shape) > 1 else np.zeros(len(X))
#         bias = 0
#         loss = 1e9
#         for i in tqdm(range(epochs)):
#             x_dot_weights = np.dot(X, weights) + bias
#             pred = np.array([1 if p > 0.5 else 0 for p in x_dot_weights])
#             new_loss = self.loss(y, pred)
#             d_w, d_b = self.gradient(X, pred, y)
#             if new_loss > loss:
#                 weights -= d_w * lr
#                 bias -= d_b * lr
#                 lr /= 2
#             weights -= d_w * lr
#             bias -= d_b * lr
#             loss = new_loss
#         self.coef_ = weights
#         self.intercept_ = bias
#         return self
            
    
#     def loss(self, y_true, y_pred):
#         # print(f"y_true: {y_true.shape}")
#         # print(f"y_pred: {y_pred.shape}")
#         # input("Press Enter to continue...")
#         errors = y_pred - y_true
#         squared_sum = 1/2 * np.mean(errors**2)
#         return squared_sum
    
#     def gradient(self, x, y_pred, y_true):
#         errors = y_pred - y_true
#         grad_b = np.mean(errors)
#         grad_w = np.matmul(errors, x)
#         grad_w = np.array([np.mean(gradient) for gradient in grad_w])
#         return (grad_w, grad_b)
            
    
    
    
    

if __name__ == "__main__":
    samples = 10000
    X, y = make_classification(n_samples=samples, n_classes=2, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, class_sep=10)
    # X, y = make_blobs(n_samples=10000, n_features=2, centers=2, cluster_std=1.0, random_state=40)
    
    print(f"{np.sum(y)/samples * 100}% of the data is in class 1")
    print("Test Basic Functionality")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearTikhonovClassifier(scale=0.0)
    model = model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    score = model.score(y_train, predictions)
    loss = model.loss(y_train, predictions)
    print(f"Score: {score}, Loss: {loss}")
    print("#"*80)
    
    print("Test Gradient Check")
    # Analytical gradient
    grad1 = model.gradient(X_train, y_train)
    print(f"Analytical Gradient: {grad1[1].shape}")
    input("Press Enter to continue...")
    grad2 = model.check_grad(X_train, y_train)
    assert grad1[1].shape == grad2.shape, f"Gradient shapes do not match: {grad1[1].shape} != {grad2.shape}"
    print(f"Numerical Gradient: {grad2.shape}")
    diff = grad1[1] - grad2
    print(f"Difference: {diff}")
    print(f"Max Difference: {np.max(diff)}")
    print(f"Min Difference: {np.min(diff)}")
    print(f"Mean Difference: {np.mean(diff)}")
    print(f"Std Difference: {np.std(diff)}")
    print(f"Norm Difference: {np.linalg.norm(diff)}")
    print(f"{grad1[0]}")
    
    