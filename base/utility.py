import numpy as np


def get_sub_seqs(x_arr, seq_len=100, stride=1):
    seq_starts = np.arange(0, x_arr.shape[0] - seq_len + 1, stride)
    x_seqs = np.array([x_arr[i:i + seq_len] for i in seq_starts])

    return x_seqs


def get_sub_seqs_label(y, seq_len=100, stride=1):
    seq_starts = np.arange(0, y.shape[0] - seq_len + 1, stride)
    ys = np.array([y[i:i + seq_len] for i in seq_starts])
    y = np.sum(ys, axis=1) / seq_len

    y_binary = np.zeros_like(y)
    y_binary[np.where(y != 0)[0]] = 1
    return y_binary
