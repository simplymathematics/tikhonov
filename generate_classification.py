import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import ParameterGrid
import argparse
from hashlib import md5
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=1000, nargs='+')
    parser.add_argument('--n_features', type=int, default=10, nargs='+')
    parser.add_argument('--n_informative', type=int, default=9, nargs='+')
    parser.add_argument('--n_redundant', type=int, default=0, nargs='+')
    parser.add_argument('--n_repeated', type=int, default=0, nargs='+')
    parser.add_argument('--n_classes', type=int, default=2, nargs='+')
    parser.add_argument('--n_clusters_per_class', type=int, default=1, nargs='+')
    parser.add_argument('--weights', type=float, default=None, nargs='+')
    parser.add_argument('--flip_y', type=float, default=0.0, nargs='+')
    parser.add_argument('--class_sep', type=float, default=10.0, nargs='+')
    parser.add_argument('--hypercube', type=bool, default=True, nargs='+')
    parser.add_argument('--shift', type=float, default=0.0, nargs='+')
    parser.add_argument('--scale', type=float, default=1.0, nargs='+')
    parser.add_argument('--shuffle', type=bool, default=True, nargs='+')
    parser.add_argument('--random_state', type=int, default=42, nargs='+')
    data_args = parser.parse_args()
    for key, value in vars(data_args).items():
        if not isinstance(value, list):
            new_value = [value]
            setattr(data_args, key, new_value)
    grid_ = ParameterGrid(vars(data_args))
    for data_arg in grid_:
        identifier = md5(str(data_arg).encode('utf-8')).hexdigest()
        separable = False
        print("Generating data...")
        i = 0
        while not separable:
            try:
                samples = make_classification(**data_arg)
                red = samples[0][samples[1] == 0]
                blue = samples[0][samples[1] == 1]
                separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])
                i += 1
                X, y = samples
            except ValueError:
                pass
        print("Data generated!")
        path = Path("data")
        path.mkdir(parents=True, exist_ok=True)
        print(data_arg)
        np.savez(path/f"{identifier}.npz", X=X, y=y)
        