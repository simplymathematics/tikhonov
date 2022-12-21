import numpy as np
import torch
from pathlib import Path
from torch.autograd.functional import jacobian
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc

class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, criterion = 'mse', scale = 0.0):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias = True)
        self.scale = scale
        if criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'cross_entropy' or criterion == 'ce':
            self.criterion = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            criterion = criterion
        
    def forward(self, x):
        outputs = self.linear(x)
        return outputs

    def loss(self, X_train, y_train):
        outputs = self.predict(X_train)
        loss = self.criterion(outputs, y_train) /2
        # Tikhonov regularization
        grady_x,  = jacobian(self.predict, (X_train, ), strict = True)
        squared = torch.square(grady_x)
        mean = torch.mean(squared, axis = 0)
        tikhonov_loss = torch.sum(mean)/2
        loss = loss + self.scale * tikhonov_loss
        return loss
        
            
    
    
    def fit(self, X_train, y_train, epochs, learning_rate= 1e-8, X_test = None, y_test = None):
        losses = []
        scores = []
        Iterations = []
        iter = 0
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        optimizer.zero_grad() # Setting our stored gradients equal to zero
        old_loss = 1e9
        
        for iter in tqdm(range(int(epochs)),desc='Training Epochs'):
            
            loss = self.loss(X_train, y_train)
            if loss > old_loss:
                learning_rate = learning_rate/2
                optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
                optimizer.zero_grad() # Setting our stored gradients equal to zero
                print(f"Learning rate decreased to {learning_rate}")
            else:
                pass
            # Computes the gradient of the given tensor w.r.t. graph leaves 
            loss.backward()
            # Updates weights and biases with the optimizer (SGD)
            optimizer.step()
            old_loss = loss
            if iter%100 ==0:
                with torch.no_grad():
                    # Calculating the train loss
                    # outputs_train = torch.squeeze(self.forward(X_train))
                    loss_train = self.loss(X_train, y_train)
                    print("\nTraining Scores:")
                    score = self.score(X_train, y_train)
                    if y_test is None and X_test is None:
                        losses.append(loss_train.item())
                        scores.append(score)
                        Iterations.append(iter)
                    if X_test is not None and y_test is not None:
                        print("Testing Scores:")
                        loss_test = self.loss(X_test, y_test)
                        losses.append(loss_test.item())
                        score = self.score(X_test, y_test)
                        scores.append(score)
        return losses, scores, Iterations
    
    def predict(self, X_test):
        # with torch.no_grad():
        outputs_test = torch.squeeze(self.forward(X_test))
        return outputs_test
    
    def score(self, X_test, y_test):
        print(f"MSE: {torch.mean((self.predict(X_test) - y_test) ** 2).item()}")
        print(f"RMSE: {torch.sqrt(torch.mean((self.predict(X_test) - y_test) ** 2)).item()}")
        class_labels = (self.predict(X_test) > 0.5).float()
        accuracy = (class_labels == y_test).float().mean()
        print(f"Accuracy: {accuracy.item()}")
        return accuracy
        

if __name__ == "__main__":
    import argparse
    import gc
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scale', type=float, default=0.00)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    model_args = parser.parse_args()
    torch.manual_seed(model_args.seed)  
    sample_args = argparse.Namespace(test_size = model_args.test_size, random_state = model_args.seed)
    random_state  = np.random.RandomState(model_args.seed)
    files = Path('data/').glob('*.npz')
    for file_ in files:
        device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        data = np.load(file_)
        inputs, labels = data['X'], data['y']
        X_train, X_test, y_train,  y_test = train_test_split(
            inputs, labels, test_size=sample_args.test_size, random_state=random_state, stratify = labels)
        print("Data split into train and test sets")
        X_train = torch.Tensor(X_train).to(device)
        X_train.requires_grad = True
        X_test = torch.Tensor(X_test).to(device)
        X_test.requires_grad = True
        y_train = torch.squeeze(torch.Tensor(y_train).to(device))
        y_test = torch.squeeze(torch.Tensor(y_test).to(device))
        
        
        
        device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        print("Using device: ", device, " for training.")
        print("with args: ", model_args)
        epochs = model_args.epochs
        input_dim = X_train.shape[1]
        output_dim = 1
        learning_rate = model_args.lr

        model = LinearRegression(input_dim,output_dim, criterion = 'mse').to(device)
        X_train, X_test = torch.Tensor(X_train),torch.Tensor(X_test)
        y_train, y_test = torch.Tensor(y_train),torch.Tensor(y_test)
        
        print("Testing Loss Function")
        loss = model.loss(X_train, y_train)
        print("Training model...")
        model.fit(X_train, y_train, epochs, learning_rate, X_test, y_test)
        print("Model trained!")
        print("Testing model...")
        score = model.score(X_test, y_test)
        print("Model tested!")
        torch.cuda.empty_cache()
        gc.collect()
    sys.exit(0) 