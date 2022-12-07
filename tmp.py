import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
import autograd.numpy as np
from autograd import grad
from autograd.test_util import check_grads




def generate_halfmoon(n1, n2, max_angle=3.14):
    alpha = np.linspace(0, max_angle, n1)
    beta = np.linspace(0, max_angle, n2)
    x1 = np.vstack([np.cos(alpha), np.sin(alpha)]) + 0.1 * np.random.randn(2,n1)
    x2 = np.vstack([1 - np.cos(beta), 1 - np.sin(beta) - 0.5]) + 0.1 * np.random.randn(2,n2)
    y1, y2 = np.zeros(n1), np.ones(n2)
    X = np.hstack([x1, x2]).T
    y = np.hstack([y1, y2]).T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, shuffle = True, stratify = y)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test


def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def loss(y:np.ndarray, y_hat:np.ndarray) -> float:
    """
    Calculate the loss.
    :param y: true/target value.
    :param y_hat: hypothesis/predictions.
    :return: loss.
    """
    loss = log_loss(y, y_hat, normalize = False)
    return loss

def gradient(x:np.ndarray, y:np.ndarray, y_hat:np.ndarray) -> tuple:
    """
    Calculate the gradients.
    :param x: input data.
    :param y: true/target value.
    :param y_hat: hypothesis/predictions.
    :return: weight gradient, bias gradient.
    """
    n = x.shape[0]
    # Gradient of loss w.r.t weights.
    dw = (1/n)*np.dot(x.T, (y_hat - y))
    # Gradient of loss w.r.t bias.
    db = (1/n)*np.sum((y_hat - y)) 
    return dw, db

def fit(x:np.ndarray, y:np.ndarray, bs:np.ndarray, epochs:int, lr:float, method:str = 'sgd') -> tuple:
    """
    Train the model.
    :param x: input data.
    :param y: true/target value.
    :param bs: batch size.
    :param epochs: number of epochs.
    :param lr: learning rate.
    :return: weights, bias.
    """
    m, n = x.shape
    # Setting initial weights and bias to 0.
    w = np.zeros((n,1))
    b = 0
    # Ensuring y is in the right shape.
    y = y.reshape(m,1)
    # Training the model.
    losses = []
    grads = []
    for _ in range(epochs):
        for i in range((m-1)//bs + 1):
            # Finding the batches.
            start_i = i*bs
            end_i = start_i + bs
            xb = x[start_i:end_i]
            yb = y[start_i:end_i]
            # Predictions
            y_hat = sigmoid(np.dot(xb, w) + b)
            # Calculating the gradients.
            dw, db = gradient(xb, yb, y_hat)
            # Adjust the gradients
            if method == 'sgd':
                w -= lr*dw
                b -= lr*db
        # Finding the loss.
        l = loss(y, sigmoid(np.dot(x, w) + b))
        grad = (np.linalg.norm(dw) ** 2 + np.linalg.norm(db) ** 2) ** .5
        losses.append(l)
        grads.append(grad)
    return w, b, losses, grads

def predict(x:np.ndarray, w:np.ndarray, b:np.ndarray, proba:bool = False) -> np.ndarray:
    """
    Predict the class of the input data.
    :param x: input data.
    :param w: weights.
    :param b: biases.
    :return: predictions.
    """
    preds = sigmoid(np.dot(x, w) + b)
    if not proba:
        pred_classes = []
        pred_classes = [1 if i > 0.5 else 0 for i in preds]
        preds = np.array(pred_classes)
    else:
        print("Returning probabilities.")
    return preds

x_train, x_test, y_train, y_test = generate_halfmoon(100, 100)
w, b, l1, g1 = fit(x_train, y_train, bs=100, epochs=1000, lr=.001, method = 'sgd')
y_pred1  = predict(x_test, w, b)
print("Accuracy: ", accuracy_score(y_test, y_pred1))