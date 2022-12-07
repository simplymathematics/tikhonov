import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from autograd import grad, jacobian
from tqdm import tqdm

class LinearTikhonovClassifier():
    def __init__(self, scale):
        self.scale=scale
    
    def predict(self, X, weights=None, bias=None):
        if weights is None:
            weights = self.coef_
        if bias is None:
            bias = self.intercept_
        print(f"Type of weights", type(weights))
        print(f"Shape of weights", weights.shape)
        print(f"Weight samples", weights[:5])
        input("Press Enter to continue...")
        X_dot_weights = torch.matmul(X, weights) + bias
        return X_dot_weights

    def _setup(self, X, y):
        if not hasattr(self, "coef_"):
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            X = torch.tensor(X, dtype=torch.float32)
            X = X.to(device)
            y = torch.tensor(y, dtype=torch.int32)
            y = y.to(device)
            _, p = X.shape
            self.coef_ = torch.randn(p, 1, dtype=torch.float32)
            self.coef_ = torch.tensor(self.coef_, dtype=torch.float32, device=device, requires_grad=True)
            self.intercept_ = torch.zeros(1, dtype=torch.float32)
            self.intercept_ = torch.tensor(self.intercept_, dtype=torch.float32, device=device, requires_grad=True)
        return self
    
    def loss(self, X, y, weights, bias):
        y_pred = self.predict(X, weights = weights, bias = bias)
        loss = .5 * torch.mean((y_pred - y) ** 2)
        tikhonov_loss = self.tikhonov_loss(X, weights = weights, bias = bias)
        return loss + self.scale * tikhonov_loss

    def tikhonov_loss(self, x, weights, bias):
        
        grady_x = jacobian(self.predict, argnum = 0)(x, weights = weights, bias = bias)
        reduced = grady_x.reshape(grady_x.shape[0], -1)
        summed = torch.sum(reduced ** 2, axis = 1)
        return torch.mean(summed)/2
    
    
    def gradient(self, X, y, weights, bias):
        result = grad(self.loss)(X, y, weights = weights, bias = bias)
        return (torch.mean(result[0]), torch.mean(result[1]))
    
    def fit(self, X, y, learning_rate = 1e-8, epochs = 1000, warm_start = False):
        self = self._setup(X, y)
        old_loss = 1e9
        L_w = self.coef_ * 0.0
        L_b = 0
        for i in tqdm(range(epochs), desc = f"Epochs: {epochs}. Learning Rate: {learning_rate}. Scale: {self.scale}. Warm Start: {warm_start}"):
            L_w, L_b = self.gradient(X, y, self.coef_, self.intercept_)
            self.coef_ -= L_w * learning_rate
            self.intercept_ -= L_b * learning_rate
            y_pred = self.predict(X)
            new_loss = self.loss(X, y, self.coef_, self.intercept_)
            if new_loss > old_loss:
                learning_rate /= 2
            else:
                old_loss = new_loss
        print(f"Final Loss: {self.loss(X, y, self.coef_, self.intercept_)}")
        print(f"Final Accuracy: {self.score(y, y_pred)}")
        print(f"Final Learning Rate: {learning_rate}")
        return self

    def score(self, y_true, y_pred):
        classes = []
        for y in y_pred:
            if y > .5:
                classes.append(1)
            else:
                classes.append(0)
        return accuracy_score(y_true, classes)
    
    

if __name__ == "__main__":
    samples = 10
    X, y = make_classification(n_samples=samples, n_classes=2, n_features=10, n_informative=8, n_redundant=0, n_clusters_per_class=1, class_sep=10)
    
    print(f"No. of Samples: {samples}")
    print(f"Classes: {np.unique(y)}")
    print(f"Largest Class: {np.bincount(y).max()}")
    print(f"Smallest Class: {np.bincount(y).min()}")
    print("Test Basic Functionality")
    input("Press Enter to continue...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearTikhonovClassifier(scale=0)
    model = model.fit(X_train, y_train, learning_rate=1e-8, epochs=1)
    y_pred = model.predict(X_test)
    score = model.score(y_train, y_pred)
    loss = model.loss(X_train, y_train, model.coef_, model.intercept_)
    print(f"Score: {score}, Loss: {loss}")
    # print("#"*80)