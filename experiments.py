
import argparse
from pathlib import Path

import pandas as pd
from autograd import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from linear import LinearTikhonovClassifier
from logistic import LogisticTikhonovClassifier

n_classes = 2
X, y = make_classification(n_samples=10000, n_classes=n_classes, n_features=5, n_informative=4, n_redundant=0, n_clusters_per_class=1, class_sep = 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



def run_tikho_experiment(model, X_train, X_test, y_train, y_test, epochs = 1000, learning_rate = 1e-8):
    model = model.fit(X_train, y_train, learning_rate = learning_rate, epochs = epochs)
    predictions = model.predict(X_test )
    score = model.score(y_test, predictions)
    loss = model.loss(X_test, y_test, model.coef_, model.intercept_)
    return score, loss
    
    
def run_data_experiment(model, X, y, test_noise=0.0, train_noise=0.0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if train_noise >= 0.0:
        X_train = X_train + np.random.normal(loc=0, scale=train_noise, size = X_train.shape)
    else:
        X_train = X_train - np.random.normal(loc=0, scale=-train_noise, size = X_train.shape)
    if test_noise >= 0.0:
        X_test = X_test + np.random.normal(loc=0, scale=test_noise, size = X_test.shape)
    else:
        X_test = X_test - np.random.normal(loc=0, scale=-test_noise, size = X_test.shape)
    
    return run_tikho_experiment(model, X_train, X_test, y_train, y_test,)




def run_feature_experiment(model, train_noise = 0.0, test_noise = 0.0, **kwargs):
    X, y = make_classification(**kwargs)
    return run_data_experiment(X =X, y=y,train_noise=train_noise, test_noise=test_noise, model=model)

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, default="linear")
    args = parser.parse_args()
    if "logistic" == args.model:
        model = LogisticTikhonovClassifier(scale = 0.0)
    elif "linear" == args.model:
        model = LinearTikhonovClassifier(scale = 0.0)
    else:
        raise ValueError(f"Unknown model: {args.model}")


    score, loss = run_tikho_experiment(model, X_train, X_test, y_train, y_test, epochs = 100)
    print(f"Score: {score}, Loss: {loss}")
    input("Press Enter to continue...")

    print("Experiment 1: Tikhonov regularization")
    scale_scores = []
    scale_losses = []
    scales = list(np.logspace(-5, 5, 11) * -1)
    scales.reverse()    
    scales.append(0.0)
    scales.extend(np.logspace(-5, 5, 11).tolist())
    for scale in scales:
        model.scale = scale
        score, loss = run_tikho_experiment(model, X_train, X_test, y_train, y_test)
        scale_scores.append(score)
        scale_losses.append(np.mean(loss))
        print(f"Scale: {scale:.3e}, Score: {score}, Loss: {loss}")
    print("#"*80)

    
    print("Experiment 2: Varying number of samples")
    sample_losses = []
    sample_scores = []
    samples = [100, 300, 500, 1000, 3000, 5000, 10000, 100000]
    samples_1 = [100, 300, 500, 1000, 3000, 5000, 10000, 100000]
    for samples in samples:
        scale=scales[scale_losses.index(min(scale_losses))]
        model.scale = scale
        score, loss = run_feature_experiment(model, n_samples=samples, n_features=100, n_informative=80, n_redundant = 20, n_clusters_per_class=1)
        sample_losses.append(np.mean(loss))
        sample_scores.append(score)
        print(f"Samples: {samples}, Score: {score}, Loss: {loss}")
    print("#"*80)
    
    print("Experiment 3: Training with Noise")
    train_scores = []
    train_losses = []
    train_noises = [-100, -10, -1, -.1, -.01, -.001, -.0001, 0, .0001, .001, .01, .1, 1, 10, 100, ]
    for train_noise in train_noises:
        score, loss = run_data_experiment(model, X, y, test_noise=0.0, train_noise=train_noise)
        print(f"Train Noise: {train_noise:.3e}, Score: {score}, Loss: {loss}")
        train_scores.append(score)
        train_losses.append(np.mean(loss))
    print("#"*80)    

    print("Experiment 4: Testing with Noise")
    test_scores = []
    test_losses = []
    test_noises = [-100, -10, -1, -.1, -.01, -.001, -.0001, 0, .0001, .001, .01, .1, 1, 10, 100, ]
    scale=scales[scale_losses.index(min(scale_losses))]
    for test_noise in [-100, -10, -1, -.1, -.01, -.001, -.0001, 0, .0001, .001, .01, .1, 1, 10, 100, ]:
        score, loss = run_data_experiment(model, X, y, train_noise=0.0, test_noise=test_noise)
        print(f"Test Noise: {test_noise:.3e}, Score: {score}, Loss: {loss}")
        test_scores.append(score)
        test_losses.append(np.mean(loss))
    print("#"*80)

    print("Experiment 5: Varying number of features")
    data_scores = []
    data_losses = []
    features_1 = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    n_informative = [int(round(x * .8))  for x in features_1]
    for feat, inf in zip(features_1, n_informative):
        score, loss = run_feature_experiment(model, n_features=feat, n_informative=inf, n_redundant = feat-inf, n_clusters_per_class=1)
        print(f"Features: {feat}, Score: {score}, Loss: {loss}")
        data_scores.append(score)
        data_losses.append(np.mean(loss))

    print("Experiment 6: Varying number of informative features")
    info_scores = []
    info_losses = []
    features_2 = [100] * 9
    n_informative = [int(round(x * 100))  for x in [ .01, .1, .2, .3, .4, .5, .6, .7, .8]]
    for feat, inf in zip(features_2, n_informative):
        score, loss = run_feature_experiment(model, n_features=feat, n_informative=inf, n_redundant = feat-inf, n_clusters_per_class=1)
        print(f"Number of Informative: {inf}, Score: {score}, Loss: {loss}")
        info_scores.append(score)
        info_losses.append(np.mean(loss))
    print("#"*80)


        

    sample_res = {
        "sample_scores": sample_scores,
        "sample_losses": sample_losses,
        "samples": samples_1,
    }
    scale_res ={
        "scale_losses": scale_losses,
        "scale_scores": scale_scores,
        "scales": scales,   
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
        "samples": sample_res,
        "training_with_tikhonov": scale_res,
        "training_with_noise" : train_res,
        "testing_with_noise" : test_res,
        "varying_features" : data_res,
        "varying_informative" : info_res,
    }
    for k,v in results.items():
        df = pd.DataFrame(v)
        path = Path("linear_results") / f"{k}.csv"
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)