import pandas as pd
import numpy as np
import os
import csv

def multidim_dtw(seq1, seq2):
    n, d1 = seq1.shape
    m, d2 = seq2.shape
    assert d1 == d2, "Sequences must have the same dimensionality"

    dtw_mat = np.full((n + 1, m + 1), np.inf)
    dtw_mat[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dtw_mat[i, j] = cost + min(dtw_mat[i - 1, j], dtw_mat[i, j - 1], dtw_mat[i - 1, j - 1])
    return dtw_mat[n, m], dtw_mat[1:, 1:]

def write_csv_row(file_path, model, dataset, DTW):
    header = ['model', 'dataset', 'DTW']
    write_header = not os.path.exists(file_path)

    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([model, dataset, DTW])

if __name__ == "__main__":
    model_list = ['Dirty']  # 'Dirty', 'SHoTClean-B', 'SHoTClean-C', 'EWMA', 'LsGreedy', 'MTCSC', 'SCREEN', 'SpeedAcc', 'TranAD', 'Akane', 'ARIMA', 'Clean4MTS', 'IMDiffusion']
    dataset_list = ["exchange", "PSM", "SWaT", "UCI"]
    datadim_list = [8, 25, 26, 21]  # dim of datasets
    for model in model_list:
        for i in range(len(dataset_list)):
            dataset=dataset_list[i]
            datadim=datadim_list[i]
            df = pd.read_csv(f'../results/{dataset}/{model}.csv', header=None)

            label = np.ones(len(df), dtype=int)
            for i in range(len(df)):
                np1 = np.array(df.iloc[i, 1:datadim + 1])
                np2 = np.array(df.iloc[i, datadim + 1:2 * datadim + 1])
                if np.array_equal(np1, np2):
                    label[i] = 0
                else:
                    label[i] = 1

            result = np.array(df.iloc[:, 1:datadim+1])
            truth = np.array(df.iloc[:, datadim+1:2*datadim+1])
            result = result[label.reshape(-1) == 1]
            truth = truth[label.reshape(-1) == 1]
            dist, matrix = multidim_dtw(result, truth)
            write_csv_row('..results/DTW/DTW.csv', model, dataset, dist/np.sum(label))