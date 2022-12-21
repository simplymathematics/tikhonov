
import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from torch_linear import LinearRegression
from torch_logistic import LogisticRegression
import torch
import numpy as np

device = 'cpu' if not torch.cuda.is_available() else 'cuda'
n_classes = 2

def generate_data(n_samples=1000, n_classes=n_classes, n_features=10, n_informative=9, n_redundant=0, n_clusters_per_class=1, class_sep = 10):
    separable = False
    print("Generating separable data...")
    while not separable:
        samples = make_classification(n_samples=n_samples, n_classes=n_classes, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_clusters_per_class=n_clusters_per_class, class_sep = class_sep)
        red = samples[0][samples[1] == 0]
        blue = samples[0][samples[1] == 1]
        separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])
    print("Data generated!")
    X, y = samples
    return X, y

def run_tikho_experiment(model, X_train, X_test, y_train, y_test, epochs, learning_rate = 1):
    losses, scores, iterations = model.fit(X_train, y_train, learning_rate = learning_rate, epochs = epochs, X_test = X_test, y_test = y_test)
    return losses, scores, iterations
    
    
def run_data_experiment(model, X, y, epochs=1000, test_noise=0.0, train_noise=0.0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if train_noise >= 0.0:
        X_train = X_train + np.random.normal(loc=0, scale=train_noise, size = X_train.shape)
    else:
        X_train = X_train - np.random.normal(loc=0, scale=-train_noise, size = X_train.shape)
    if test_noise >= 0.0:
        X_test = X_test + np.random.normal(loc=0, scale=test_noise, size = X_test.shape)
    else:
        X_test = X_test - np.random.normal(loc=0, scale=-test_noise, size = X_test.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.Tensor(X_train).to(device)
    X_test = torch.Tensor(X_test).to(device)
    y_train = torch.squeeze(torch.Tensor(y_train).to(device))
    y_test = torch.squeeze(torch.Tensor(y_test).to(device))
    return run_tikho_experiment(model, X_train, X_test, y_train, y_test, epochs=epochs)




def run_feature_experiment(model, train_noise = 0.0, test_noise = 0.0, class_sep = 10, epochs =1000,  **kwargs):
    train_noise = kwargs.pop("train_noise", train_noise)
    test_noise = kwargs.pop("test_noise", test_noise)
    X, y = make_classification(**kwargs, class_sep = class_sep)
    return run_data_experiment(X =X, y=y,train_noise=train_noise, test_noise=test_noise, model=model, epochs=epochs)

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, default="linear")
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()
    EPOCHS = args.epochs

    final_score = 0.0

    while final_score < 0.99:
        X, y = generate_data(n_samples=1000, n_classes=n_classes, n_features=10, n_informative=9, n_redundant=0, n_clusters_per_class=1, class_sep = 10)
        if "logistic" == args.model:
            model = LogisticRegression(scale = 0.0, input_dim = X.shape[1], output_dim = 1, criterion = 'ce')
        elif "linear" == args.model:
            model = LinearRegression(scale = 0.0, input_dim = X.shape[1], output_dim = 1, criterion = 'mse')
        else:
            raise ValueError(f"Unknown model: {args.model}")
        model.to(device)
        loss, score, iterations = run_data_experiment(model, X, y, train_noise = 0.0, test_noise = 0.0, epochs=EPOCHS)
        final_score = score[-1]
        print(f"Score: {score[-1]}, Loss: {loss[-1]}")
    input("Press Enter to continue...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.Tensor(X_train).to(device)
    X_test = torch.Tensor(X_test).to(device)
    y_train = torch.squeeze(torch.Tensor(y_train).to(device))
    y_test = torch.squeeze(torch.Tensor(y_test).to(device))
    print("Experiment 1: Tikhonov regularization")
    scale_scores = []
    scale_losses = []
    scales = list(np.logspace(-5, 5, 11) * -1)
    scales.reverse()    
    scales.append(0.0)
    scales.extend(np.logspace(-5, 5, 11).tolist())
    for scale in scales:
        if "logistic" == args.model:
            model = LogisticRegression(scale = 0.0, input_dim = X.shape[1], output_dim = 1, criterion = 'ce')
        elif "linear" == args.model:
            model = LinearRegression(scale = 0.0, input_dim = X.shape[1], output_dim = 1, criterion = 'mse')
        else:
            raise ValueError(f"Unknown model: {args.model}")
        model.to(device)
        loss, score, iterations = run_tikho_experiment(model, X_train, X_test, y_train, y_test, epochs = EPOCHS)
        scale_scores.append(score[-1])
        scale_losses.append(loss[-1])
        print(f"Scale: {scale:.3e}, Score: {score[-1]}, Loss: {loss[-1]}")
    print("#"*80)

    
    print("Experiment 2: Varying number of samples")
    sample_losses = []
    sample_scores = []
    samples = [100, 300, 500, 1000, 3000, 5000]
    samples_1 = [100, 300, 500, 1000, 3000, 5000]
    for samples in samples:
        scale=scales[scale_losses.index(min(scale_losses))]
        if "logistic" == args.model:
            model = LogisticRegression(scale = scale, input_dim = X.shape[1], output_dim = 1, criterion = 'ce')
        elif "linear" == args.model:
            model = LinearRegression(scale = scale, input_dim = X.shape[1], output_dim = 1, criterion = 'mse')
        else:
            raise ValueError(f"Unknown model: {args.model}")
        model.to(device)
        loss, score, iterations = run_feature_experiment(model, n_samples=samples, n_features=10, n_informative=9, n_redundant = 0, n_clusters_per_class=1, epochs = EPOCHS, class_sep = 10)
        sample_losses.append(loss[-1])
        sample_scores.append(score[-1])
        print(f"Samples: {samples}, Score: {score[-1]}, Loss: {loss[-1]}")
    print("#"*80)
    
    print("Experiment 3: Training with Noise")
    train_scores = []
    train_losses = []
    train_noises = [-100, -10, -1, -.1, -.01, -.001, -.0001, 0, .0001, .001, .01, .1, 1, 10, 100, ]
    for train_noise in train_noises:
        loss, score, iterations = run_data_experiment(model, X, y, test_noise=0.0, train_noise=train_noise, epochs = EPOCHS)
        print(f"Train Noise: {train_noise:.3e}, Score: {score[-1]}, Loss: {loss[-1]}")
        train_scores.append(score[-1])
        train_losses.append(loss[-1])
    print("#"*80)    

    print("Experiment 4: Testing with Noise")
    test_scores = []
    test_losses = []
    test_noises = [-100, -10, -1, -.1, -.01, -.001, -.0001, 0, .0001, .001, .01, .1, 1, 10, 100, ]
    for test_noise in [-100, -10, -1, -.1, -.01, -.001, -.0001, 0, .0001, .001, .01, .1, 1, 10, 100, ]:
        loss, score, iterations = run_data_experiment(model, X, y, train_noise=0.0, test_noise=test_noise, epochs = EPOCHS)
        print(f"Test Noise: {test_noise:.3e}, Score: {score[-1]}, Loss: {loss[-1]}")
        test_scores.append(score[-1])
        test_losses.append(loss[-1])
    print("#"*80)

    print("Experiment 5: Varying number of features")
    data_scores = []
    data_losses = []
    features_1 = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    n_informative = [int(round(x * .8))  for x in features_1]
    for feat, inf in zip(features_1, n_informative):
        if "logistic" == args.model:
            model = LogisticRegression(scale = scale, input_dim = feat, output_dim = 1, criterion = 'ce')
        elif "linear" == args.model:
            model = LinearRegression(scale = scale, input_dim = feat, output_dim = 1, criterion = 'mse')
        else:
            raise ValueError(f"Unknown model: {args.model}")
        model.to(device)
        loss, score, iterations = run_feature_experiment(model, n_features=feat, n_informative=inf, n_redundant = feat-inf, n_clusters_per_class=1, epochs = EPOCHS)
        print(f"Features: {feat}, Score: {score[-1]}, Loss: {loss[-1]}")
        data_scores.append(score[-1])
        data_losses.append(loss[-1])

    print("Experiment 6: Varying number of informative features")
    info_scores = []
    info_losses = []
    features_2 = [100] * 9
    n_informative = [int(round(x * 100))  for x in [ .01, .1, .2, .3, .4, .5, .6, .7, .8]]
    for feat, inf in zip(features_2, n_informative):
        if "logistic" == args.model:
            model = LogisticRegression(scale = scale, input_dim = feat, output_dim = 1, criterion = 'ce')
        elif "linear" == args.model:
            model = LinearRegression(scale = scale, input_dim = feat, output_dim = 1, criterion = 'mse')
        else:
            raise ValueError(f"Unknown model: {args.model}")
        model.to(device)
        loss, score, iterations = run_feature_experiment(model, n_features=feat, n_informative=inf, n_redundant = feat-inf, n_clusters_per_class=1, epochs = EPOCHS)
        print(f"Number of Informative: {inf}, Score: {score[-1]}, Loss: {loss[-1]}")
        info_scores.append(score[-1])
        info_losses.append(loss[-1])
    print("#"*80)


        

    
    scale_res ={
        "scale_losses": scale_losses,
        "scale_scores": scale_scores,
        "scales": scales,   
    }
    sample_res = {
        "sample_scores": sample_scores,
        "sample_losses": sample_losses,
        "samples": samples_1,
    }
    train_res ={
        "train_losses": train_losses,
        "train_scores": train_scores,
        "train_noises": train_noises,
    }
    test_res ={
        "test_losses": test_losses,
        "test_scores": test_scores,
        "test_noises": test_noises,
    }
    data_res = {
        "data_losses": data_losses,
        "data_scores": data_scores,
        "n_features": features_1,
        
    }
    info_res = {
        "info_losses": info_losses,
        "info_scores": info_scores,
        "n_informative": n_informative,
    }

    results = {
        "training_with_tikhonov": scale_res,
        "samples": sample_res,
        "training_with_noise" : train_res,
        "testing_with_noise" : test_res,
        "varying_features" : data_res,
        "varying_informative" : info_res,
    }
    for k,v in results.items():
        for sub_k, sub_v in v.items():
            if isinstance(sub_v, list):
                new_list = []
                for entry in sub_v:
                    if isinstance(entry, torch.Tensor):
                        entry = entry.cpu().numpy()
                    new_list.append(entry)
                sub_v = new_list
            v[sub_k] = sub_v
        df = pd.DataFrame(v)
        path = Path("linear_results") / f"{k}.csv"
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)