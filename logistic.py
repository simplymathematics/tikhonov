from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import accuracy_score, log_loss
from linear import LinearTikhonovClassifier
import numpy as np
from tqdm import tqdm

class LogisticTikhonovClassifier(LinearTikhonovClassifier):
    def __init__(self, scale):
        self.scale = scale

    def fit(self, x:np.ndarray, y:np.ndarray,  learning_rate:float = 1e-8, epochs:int=1000, method:str = 'sgd',) -> tuple:
        """
        Train the model.
        :param x: input data.
        :param y: true/target value.
        :param bs: batch size.
        :param epochs: number of epochs.
        :param lr: learning rate.
        :return: weights, bias.
        """
        lr = learning_rate
        m, n = x.shape
        # Setting initial weights and bias to 0.
        w = np.zeros((n,1))
        b = np.zeros(1)
        self.coef_ = w
        self.intercept_ = b
        # Ensuring y is in the right shape.
        y = y.reshape(m,1)
        # Training the model.
        losses = []
        grads = []
        old_loss = 1e10
        for _ in tqdm(range(epochs)):
            # Calculating the gradients.
            dw, db = self.gradient(x, y, w, b)
            # Adjust the gradients
            if method == 'sgd':
                w -= lr*dw
                b -= lr*db
                self.coef_ = w
                self.intercept_ = b
            # Finding the loss.
            l = self.loss(x, y, w, b)
            if l > old_loss:
                lr = lr * .5
            else:
                old_loss = l
            grad = (np.linalg.norm(dw) ** 2 + np.linalg.norm(db) ** 2) ** .5
            losses.append(l)
            grads.append(grad)
        
        return self
    
    def sigmoid(self, z):
        return 1.0/(1 + np.exp(-z))

    def loss(self,  X, y, weights, bias) -> float:
        """
        Calculate the loss.
        :param y: true/target value.
        :param y_hat: hypothesis/predictions.
        :return: loss.
        """
        y_hat = self.sigmoid(np.dot(X, weights) + bias)
        loss_ = log_loss(y, y_hat, normalize = False)
        tikho = .5 * np.sum(np.sum(np.divide(1, y_hat * (y_hat - 1))))
        return np.mean(loss_ + tikho * self.scale)

    def gradient(self, x:np.ndarray, y:np.ndarray, weights:np.ndarray, bias:np.ndarray) -> tuple:
        """
        Calculate the gradients.
        :param x: input data.
        :param y: true/target value.
        :param y_hat: hypothesis/predictions.
        :return: weight gradient, bias gradient.
        """
        y_hat = self.predict(x, proba = True)
        n = x.shape[0]
        # Gradient of loss w.r.t weights.
        dw = (1/n)*np.dot(x.T, (y_hat - y))
        # Gradient of loss w.r.t bias.
        db = (1/n)*np.sum((y_hat - y)) 
        return dw, db

    def predict(self, x:np.ndarray, proba:bool = False) -> np.ndarray:
        """
        Predict the class of the input data.
        :param x: input data.
        :param w: weights.
        :param b: biases.
        :return: predictions.
        """
        w = self.coef_
        b = self.intercept_
        preds = self.sigmoid(np.dot(x, w) + b)
        if not proba:
            pred_classes = []
            pred_classes = [1 if i > 0.5 else 0 for i in preds]
            preds = np.array(pred_classes)
        else:
            pass
        return preds
    
    def score(self, y_true, y_pred):
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        if len(y_pred.shape) == 2:
            y_pred = np.argmax(y_pred, axis = 1)
        else:
            new_y_pred = []
            for  y in y_pred:
                if y > 0.5:
                    y = 1
                else:
                    y = 0
                new_y_pred.append(y)
            y_pred = np.array(new_y_pred)
        return accuracy_score(y_true, y_pred)
    

    def sigmoid(self, x):
        zs = []
        for z in x:
            z = np.divide(1, 1 + np.exp(-z))
            zs.append(z)
        return np.array(zs)

if __name__ == "__main__":
    n_classes = 2
    X, y = make_classification(n_samples=10000, n_classes=n_classes, n_features=5, n_informative=4, n_redundant=0, n_clusters_per_class=1, class_sep = 10)
    # X, y = make_blobs(n_samples=10000, n_features=2, centers=2, cluster_std=10, random_state=40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticTikhonovClassifier(scale=0)
    model = model.fit(X_train, y_train, learning_rate = 1e-8, epochs = 100)
    w = model.coef_
    b = model.intercept_
    grad2 = model.gradient(X_train, y_train, w, b)
    predictions = model.predict(X_test)
    probas = model.predict(X_test, proba = True)
    score = model.score(y_test, predictions)

    print(f"Test Accuracy: {score}")
    