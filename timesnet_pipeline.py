import argparse

# import numpy as np
# import torch
from timesnet.timesnet import *
from timesnet.timesnetmodule import *
from datetime import datetime
import time


# def get_data(DATA_PATH):
#     X_train = np.load("./" + DATA_PATH + "/X_train.npy") # for TimesNet training
#     X_test = np.load("./" + DATA_PATH + "/X_train.npy") # detect anomalies in real time
#
#     return X_train, X_test


def train_TimesNet(X_train):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    clf = TimesNet(seq_len=100,
                   stride=1,
                   lr=0.0001,
                   epochs=10,
                   batch_size=128,
                   epoch_steps=20,
                   prt_steps=1,
                   device=device,
                   pred_len=0,
                   e_layers=2,
                   d_model=64,
                   d_ff=64,
                   dropout=0.1,
                   top_k=3,
                   num_kernels=1,  # hyperparameter decided by Bayesian Optimization
                   verbose=2,
                   random_state=42)

    print("Training TimesNet")
    clf.fit(X_train)

    torch.save(clf, 'timesnet.pt')


def detect_anomalies(model_name, data):
    saved_model = torch.load(model_name + ".pt")
    score = saved_model.decision_function(data)
    label_ = np.where(score > saved_model.threshold_, 1, 0)
    if 1 in label_:
        print("Annomaly occured at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def detect_anomalies_(model_name, X_test):
    saved_model = torch.load(model_name + ".pt")

    seq_num = len(X_test) // saved_model.seq_len

    for i in range(seq_num):
        data = X_test[i * saved_model.seq_len: (i + 1) * saved_model.seq_len]  # 0-99, 100-199, 200-299, ...,
        score = saved_model.decision_function(data)
        label_ = np.where(score > saved_model.threshold_, 1, 0)
        if 1 in label_:
            print("Anomaly occured at", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        time.sleep(0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="train", help="whether to train before detect anomalies")
    parser.add_argument("--path", type=str, default="PSM", help="train, test data should be inside this path")
    parser.add_argument("--model_name", type=str, default="timesnet", help="which model to use")
    opt = parser.parse_args()

    if opt.task == "train":
        # X_train, X_test = get_data(opt.path)
        X_train = np.load("./" + opt.path + "/X_train.npy")
        train_TimesNet(X_train)

    X_test = np.load("./" + opt.path + "/X_test.npy")

    # detect_anomalies(opt.model_name, X_test)
    detect_anomalies_(opt.model_name, X_test)
