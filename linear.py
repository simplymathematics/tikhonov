from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from autograd import elementwise_grad, grad, numpy as np
from tqdm import tqdm
from torch.autograd.functional import jacobian
import torch
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from autograd import jacobian as jacob2
class LinearTikhonovClassifier():
    def __init__(self, scale):
        self.scale=scale
    
    def predict(self, X, weights=None, bias=None):
        if weights is None:
            weights = self.coef_.requires_grad_(True)
        if bias is None:
            bias = self.intercept_.requires_grad_(True)
        X_dot_weights = torch.matmul(X, weights) + bias
        return X_dot_weights

    def _setup(self, X, y):
        if not hasattr(self, "coef_"):
            _, p = X.shape
            self.coef_ = np.random.randn(p,1) * 1e-4
            self.intercept_ = np.zeros(1)
            self.coef_ = torch.from_numpy(self.coef_).float()
            self.intercept_ = torch.from_numpy(self.intercept_).float()
            assert isinstance(X, torch.Tensor), f"X must be a torch tensor. It is a {type(X)}"
            assert isinstance(y, torch.Tensor), f"y must be a torch tensor. It is a {type(y)}"
            assert isinstance(self.coef_, torch.Tensor), f"coef_ must be a torch tensor. It is a {type(self.coef_)}"
            assert isinstance(self.intercept_, torch.Tensor), f"intercept_ must be a torch tensor. It is a {type(self.intercept_)}"
        return self
    
    def loss(self, X, y, weights, bias):
        y_pred = self.predict(X, weights = weights, bias = bias)
        loss = .5 * torch.mean((y_pred - y) ** 2)
        tikhonov_loss = self.tikhonov_loss(X, weights = weights, bias = bias)
        return loss + self.scale * tikhonov_loss

    def tikhonov_loss(self, x, weights, bias):
        result = jacobian(self.predict, (x, weights, bias))
        grad_X = result[0]
        reduced = grad_X.reshape(grad_X.shape[0], -1)
        return torch.mean(reduced ** 2)/2
    
    def gradient(self, X, y, weights, bias):
        result = jacobian(self.loss, (X, y, weights, bias))
        gradw = result[2]
        gradb = result[3]
        # gradw = torch.tensor(torch.mean(self.scale * self.coef_ + torch.matmul(torch.matmul(self.coef_.T, X.T) + self.intercept_ - y, X), axis = 0))
        # print(f"gradw shape: {gradw.shape}, weights shape: {self.coef_.shape}")
        # input("Press Enter to Continue...")
        # gradb = torch.mean(self.scale * torch.matmul(self.coef_.T, X.T) + self.intercept_ - y)
        # print(f"gradb shape: {gradb.shape}, bias shape: {self.intercept_.shape}")
        # input("Press Enter to Continue...")
        assert gradw.shape == weights.shape, f"gradw shape: {gradw.shape}, weights shape: {weights.shape}"
        return (gradw, gradb)
    
    def fit(self, X, y, learning_rate = 1e-8, epochs = 1000, warm_start = False):
        self = self._setup(X, y)
        L_w = self.coef_ * 0.0
        L_b = 0
        for i in tqdm(range(epochs), desc = f"Training {self.__class__.__name__}", leave = False):
            L_w, L_b = self.gradient(X, y, self.coef_, self.intercept_)
            self.coef_ -= L_w * learning_rate
            self.intercept_ -= L_b * learning_rate
            y_pred = self.predict(X, self.coef_, self.intercept_)
            new_loss = self.loss(X, y, self.coef_, self.intercept_)
            if "old_loss" not in locals():
                old_loss = new_loss
            if new_loss > old_loss:
                print(f"old loss: {old_loss}, new loss: {new_loss}")
                learning_rate = learning_rate / 2
                print(f"Learning Rate: {learning_rate}")
                print("Learning Rate Decreased. Press Enter to Continue...")
            else:
                old_loss = new_loss
            if i%100 == 0:
                print()
                print(f"Epoch: {i}, Loss: {new_loss}, Learning Rate: {learning_rate}")
        print(f"Final Loss: {self.loss(X, y, self.coef_, self.intercept_)}")
        print(f"Final Accuracy: {self.score(y, y_pred)}")
        print(f"Final Learning Rate: {learning_rate}")
        return self
    
    

    
    

if __name__ == "__main__":
    samples = 1000
    X, y = make_classification(n_samples=samples, n_classes=2, n_features=100, n_informative=98, n_redundant=0, n_clusters_per_class=1, class_sep=10)
    X = torch.from_numpy(X).float().requires_grad_(True)
    y = torch.from_numpy(y).float().requires_grad_(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearTikhonovClassifier(scale=0.0)
    model._setup(X_train, y_train)
    model = model.fit(X_train, y_train, learning_rate=1e-8, epochs=1000)
    y_pred = model.predict(X_test, model.coef_, model.intercept_)
    train_loss = model.loss(X_train, y_train, model.coef_, model.intercept_)
    test_loss = model.loss(X_test, y_test, model.coef_, model.intercept_)
    print(f"Training Loss: {train_loss}, Testing Loss: {test_loss}")
    print("#"*80)