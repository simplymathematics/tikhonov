import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from autograd import grad, jacobian
sns.set_style("darkgrid")


def tikhonov_loss(model, x, weights, bias):
    
    grady_x = jacobian(model.predict, argnum = 0)(x, weights = weights, bias = bias)
    reduced = grady_x.reshape(grady_x.shape[0], -1)
    summed = np.sum(reduced ** 2, axis = 1)
    return np.mean(summed)/2


def loss(model, X, y, scale = 0.0):
    weights = list(model.parameters())[0]
    bias = list(model.parameters())[1]
    y_pred = model.predict(X, weights = weights, bias = bias)
    loss = .5 * np.mean((y_pred - y) ** 2)
    tikhonov_loss = tikhonov_loss(model, X, weights = weights, bias = bias)
    return loss + scale * tikhonov_loss


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, type = 'linear'):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = self.linear(x)
        return outputs


    def fit(self, X_train, y_train, epochs,  model, criterion, optimizer):
        losses = []
        losses_test = []
        Iterations = []
        iter = 0
        for _ in tqdm(range(int(epochs)),desc='Training Epochs'):
            x = X_train
            labels = y_train
            optimizer.zero_grad() # Setting our stored gradients equal to zero
            outputs = model(X_train)
            loss = criterion(torch.squeeze(outputs), labels) # [200,1] -squeeze-> [200]
            weights = list(model.parameters())[0]
            bias = list(model.parameters())[1]
            # tikhonov_loss = self.tikhonov_loss(x, weights = weights, bias = bias)
            loss.backward() # Computes the gradient of the given tensor w.r.t. graph leaves 

            optimizer.step() # Updates weights and biases with the optimizer (SGD)
            iter+=1
           
            with torch.no_grad():
                # Calculating the loss and accuracy for the test dataset
                correct_test = 0
                total_test = 0
                outputs_test = torch.squeeze(model(X_test))
                loss_test = criterion(outputs_test, y_test)
                
                predicted_test = outputs_test.round().detach().numpy()
                total_test += y_test.size(0)
                correct_test += np.sum(predicted_test == y_test.detach().numpy())
                accuracy_test = 100 * correct_test/total_test
                losses_test.append(loss_test.item())
                
                # Calculating the loss and accuracy for the train dataset
                total = 0
                correct = 0
                total += y_train.size(0)
                correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_train.detach().numpy())
                accuracy = 100 * correct/total
                losses.append(loss.item())
                
                
                # Calculating the loss and accuracy for the test dataset
                total = 0
                correct = 0
                total += y_test.size(0)
                correct += np.sum(torch.squeeze(outputs_test).round().detach().numpy() == y_test.detach().numpy())
                accuracy = 100 * correct/total
                losses.append(loss.item())
                
                
                Iterations.append(iter)
                print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
                print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")
        return losses, losses_test, Iterations
    
    def predict(self, X_test, y_test):
        with torch.no_grad():
            # Calculating the loss and accuracy for the test dataset
            correct_test = 0
            total_test = 0
            outputs_test = torch.squeeze(model(X_test))
            loss_test = criterion(outputs_test, y_test)
            
            predicted_test = outputs_test.round().detach().numpy()
            total_test += y_test.size(0)
            correct_test += np.sum(predicted_test == y_test.detach().numpy())
            accuracy_test = 100 * correct_test/total_test
            
            print(f"Test - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--scale', type=float, default=0.01)
    args = parser.parse_args()
    separable = False
    while not separable:
        samples = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, flip_y=-1)
        red = samples[0][samples[1] == 0]
        blue = samples[0][samples[1] == 1]
        separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])


    red_labels = np.zeros(len(red))
    blue_labels = np.ones(len(blue))

    labels = np.append(red_labels,blue_labels)
    inputs = np.concatenate((red,blue),axis=0)

    X_train, X_test, y_train,  y_test = train_test_split(
        inputs, labels, test_size=0.33, random_state=42)

    epochs = args.epochs
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train)) - 1
    learning_rate = args.lr

    model = LogisticRegression(input_dim,output_dim)

    criterion = torch.nn.MSELoss() #+ tikhonov_loss(x, weights = weights, bias = bias)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    X_train, X_test = torch.Tensor(X_train),torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(y_train),torch.Tensor(y_test)

    model.fit(X_train, y_train, epochs, model, criterion, optimizer)
    # predictions = model.predict(X_test, y_test)