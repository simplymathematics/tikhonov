from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import accuracy_score, log_loss
from linear import LinearTikhonovClassifier
from tqdm import tqdm
import torch
class LogisticTikhonovClassifier(LinearTikhonovClassifier):

    def sigmoid(self, z):
        return torch.divide(1, (1 + torch.exp(-z)))

    def predict(self, X, weights=None, bias=None):
        if weights is None:
            weights = self.coef_
        if bias is None:
            bias = self.intercept_
        X_dot_weights = torch.matmul(X, weights) + bias
        return self.sigmoid(X_dot_weights)

    def score(self, y, y_pred):
        if len(y_pred.shape) > 1 or len(y_pred.shape) > len(y.shape):
            y_pred = torch.argmax(y_pred, dim=1)
        if len(y.shape) > 1 or len(y.shape) > len(y_pred.shape):
            y = torch.argmax(y, dim=1)
        return accuracy_score(y, y_pred)

if __name__ == "__main__":
    n_classes = 2
    X, y = make_classification(n_samples=1000, n_classes=n_classes, n_features=10, n_informative=9, n_redundant=0, n_clusters_per_class=1, class_sep = 10)
    # X, y = make_blobs(n_samples=10000, n_features=2, centers=2, cluster_std=10, random_state=40)
    X = torch.from_numpy(X).float().requires_grad_(True)
    y = torch.from_numpy(y).float().requires_grad_(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticTikhonovClassifier(scale=0)
    model = model.fit(X_train, y_train, learning_rate = 1e-8, epochs = 1000)
    w = model.coef_
    b = model.intercept_
    grad2 = model.gradient(X_train, y_train, w, b)
    probas = model.predict(X_test)
    score = model.score(y_test, predictions)
    print(f"Test Log Loss: {log_loss(y_test, probas)}")
    print(f"Train Log Loss: {log_loss(y_train, model.predict(X_train))}")
    