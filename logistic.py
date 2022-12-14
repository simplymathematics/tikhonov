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

    

if __name__ == "__main__":
    n_classes = 2
    X, y = make_classification(n_samples=10000, n_classes=n_classes, n_features=5, n_informative=4, n_redundant=0, n_clusters_per_class=1, class_sep = 10)
    # X, y = make_blobs(n_samples=10000, n_features=2, centers=2, cluster_std=10, random_state=40)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
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
    