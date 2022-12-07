import pandas as pd
import seaborn as sns
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
if "__main__" ==  __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, default="linear")
    args = parser.parse_args()
    folder = Path(args.model + "_results")
    plots_folder = Path(args.model + "_plots")
    plots_folder.mkdir(exist_ok=True)
    assert folder.is_dir(), f"Folder {folder} does not exist"
    # for file_ in folder.glob("*"):
    #     df = pd.read_csv(file_)
    samples = pd.read_csv(folder / "samples.csv")
    test_noise = pd.read_csv(folder / "testing_with_noise.csv")
    train_noise = pd.read_csv(folder / "training_with_noise.csv")
    tikhonov = pd.read_csv(folder / "training_with_tikhonov.csv")
    features = pd.read_csv(folder / "varying_features.csv")
    info = pd.read_csv(folder / "varying_informative.csv")
    
    def plot(df, x, y,  title, filename, xlabel, ylabel, path, scatter = False, **kwargs):
        if scatter is True:
            plot = sns.scatterplot(data=df, x=x, y=y, **kwargs)
        else:
            plot = sns.regplot(data=df, x=x, y=y, **kwargs)
            try:
                plot.set(xlabel=xlabel, ylabel=ylabel, xscale = "symlog", yscale = "symlog")
            except:
                plot.set(xlabel=xlabel, ylabel=ylabel, xscale = "log", yscale = "log")
        plot.set_title(title)
        path = Path(path)
        path.mkdir(exist_ok=True)
        filename = path / filename
        fig = plot.get_figure()
        fig.savefig(filename)
        plt.gcf().clear()
        return filename
    
    plot(samples, x="samples", y="sample_losses", title="Loss vs. Samples", filename="samples.png", xlabel="Samples", ylabel="Loss", path=plots_folder)
    plot(samples, x="samples", y ="sample_scores", title="Accuracy vs. Samples", filename="samples_score.png", xlabel="Samples", ylabel="Accuracy", path=plots_folder)
    plot(test_noise, x="test_noises", y="test_losses", title="Loss vs. Test Noise", filename="test_noise.png", xlabel="Test Noise", ylabel="Loss", path=plots_folder, scatter = True)
    plot(test_noise, x="test_noises", y="test_scores", title="Accuracy vs. Test Noise", filename="test_noise_score.png", xlabel="Test Noise", ylabel="Accuracy", path=plots_folder, scatter = True)
    plot(train_noise, x="train_noises", y="train_losses", title="Loss vs. Train Noise", filename="train_noise.png", xlabel="Train Noise", ylabel="Loss", path=plots_folder)
    plot(train_noise, x="train_noises", y="train_scores", title="Accuracy vs. Train Noise", filename="train_noise_score.png", xlabel="Train Noise", ylabel="Accuracy", path=plots_folder)
    plot(tikhonov, x="scales", y="scale_losses", title="Loss vs. Tikhonov Scale", filename="tikhonov.png", xlabel="Tikhonov Scale", ylabel="Loss", path=plots_folder)
    plot(tikhonov, x="scales", y="scale_scores", title="Accuracy vs. Tikhonov Scale", filename="tikhonov_score.png", xlabel="Tikhonov Scale", ylabel="Accuracy", path=plots_folder)
    plot(features, x="n_features", y="data_losses", title="Loss vs. Number of Features", filename="features.png", xlabel="Number of Features", ylabel="Loss", path=plots_folder)
    plot(features, x="n_features", y="data_scores", title="Accuracy vs. Number of Features", filename="features_score.png", xlabel="Number of Features", ylabel="Accuracy", path=plots_folder)
    plot(info, x="n_informative", y="info_losses", title="Loss vs. Number of Informative Features", filename="info.png", xlabel="Number of Informative Features", ylabel="Loss", path=plots_folder)
    plot(info, x="n_informative", y="info_scores", title="Accuracy vs. Number of Informative Features", filename="info_score.png", xlabel="Number of Informative Features", ylabel="Accuracy", path=plots_folder)