import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from tools.utils import Assist

def viz_multi():
    dataset = 'UCI'
    methods = ['B-Soft', 'C-Soft', 'EWMA', 'LsGreedy', 'MTCSC', 'SCREEN', 'SpeedAcc']
    # methods = ['Akane', 'ARIMA', 'Clean4MTS', 'IMDiffusion', 'TranAD']
    np.random.seed(42)
    assist = Assist()
    dirty = assist.read_data(f'../results/multi/{dataset}/Dirty_0.2.data', ",")
    dirty_list = dirty.get_timeseries()
    mask = np.array([
        not np.array_equal(dirty_item.get_value(), dirty_item.get_truth())
        for dirty_item in dirty_list
    ])
    for method in methods:
        ts = assist.read_data(f'../results/multi/{dataset}/{method}_0.2.data', ",")
        # ts = assist.read_data(f'../results/viz/{dataset}/viz_{method}_{dataset}.csv', ",")
        ts_list = ts.get_timeseries()

        clean = np.array([item.get_value() for item in ts_list])
        truth = np.array([item.get_truth() for item in ts_list])
        n_total = clean.shape[0]
        clean_flat = clean.reshape(n_total, -1)
        truth_flat = truth.reshape(n_total, -1)

        clean_sel = clean_flat[mask]
        truth_sel = truth_flat[mask]
        n = clean_sel.shape[0]

        data = np.vstack([clean_sel, truth_sel])  # (2*n, feature_dim)
        n_samples = data.shape[0]
        perp = min(30, n_samples - 1)

        tsne = TSNE(n_components=2,
                    init='random',
                    perplexity=perp,
                    n_iter=1000,
                    random_state=42,
                    verbose=1)
        data_2d = tsne.fit_transform(data)

        plt.figure(figsize=(6, 6))
        plt.scatter(data_2d[:n, 0], data_2d[:n, 1], color='C0', s=25, alpha=0.6)
        plt.scatter(data_2d[n:, 0], data_2d[n:, 1], color='C1', s=25, alpha=0.6)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'./results/viz/{dataset}/{method}.svg', dpi=300)

if __name__ == "__main__":
    viz_multi()