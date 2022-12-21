import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch_linear import LinearRegression
from pathlib import Path

class LogisticRegression(LinearRegression):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__(input_dim, output_dim, criterion='ce')
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def predict(self, X_test):
        outputs_test = torch.squeeze(self(X_test))
        return torch.sigmoid(outputs_test)    
    
    def score(self, X_test, y_test):
        # Calculate CE loss
        y_pred = self.predict(X_test)
        loss = torch.nn.CrossEntropyLoss()
        print(f"Loss: {loss(y_pred, y_test)}")
        y_pred = (y_pred > 0.5).float()
        accuracy = torch.mean((y_pred == y_test).float())
        print(f"Accuracy: {accuracy.item()}")
        return accuracy
        

if __name__ == "__main__":
    import argparse
    import gc
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--scale', type=float, default=0.00)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    model_args = parser.parse_args()    
    sample_args = argparse.Namespace(test_size = model_args.test_size, random_state = model_args.seed)
    random_state  = np.random.RandomState(model_args.seed)
    torch.manual_seed(model_args.seed)
    files = Path('data/').glob('*.npz')
    for file_ in files:
        device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        data = np.load(file_)
        inputs, labels = data['X'], data['y']
        X_train, X_test, y_train,  y_test = train_test_split(
            inputs, labels, test_size=sample_args.test_size, random_state=random_state, stratify = labels)
        print("Data split into train and test sets")
        X_train = torch.Tensor(X_train).to(device)
        X_test = torch.Tensor(X_test).to(device)
        y_train = torch.squeeze(torch.Tensor(y_train).to(device))
        y_test = torch.squeeze(torch.Tensor(y_test).to(device))
        
        
        
        device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        print("Using device: ", device, " for training.")
        print("with args: ", model_args)
        epochs = model_args.epochs
        input_dim = X_train.shape[1]
        output_dim = 1
        learning_rate = model_args.lr

        model = LogisticRegression(input_dim,output_dim).to(device)
        print("Model created")
        X_train, X_test = torch.Tensor(X_train),torch.Tensor(X_test)
        y_train, y_test = torch.Tensor(y_train),torch.Tensor(y_test)
        print("Training model...")
        model.fit(X_train, y_train, epochs, learning_rate,)
        print("Model trained!")
        print("Testing model...")
        score = model.score(X_test, y_test)
        print("Model tested!")
        torch.cuda.empty_cache()
        gc.collect()
    sys.exit(0) 
    