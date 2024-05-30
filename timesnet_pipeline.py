import argparse

# import numpy as np
# import torch
from timesnet.timesnet import *
from timesnet.timesnetmodule import *
from datetime import datetime
import time


def train_TimesNet(train_data, model_name):
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

    print("---Training TimesNet---")
    clf.fit(train_data)

    torch.save(clf, model_name+'.pt')
    print("Saved", model_name+".pt\n")


def detect_anomalies(model_name, data):  # anomly detection in real time
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    saved_model = torch.load(model_name + ".pt", device)
    saved_model.device = device

    score = saved_model.decision_function(data)
    label_ = np.where(score > saved_model.threshold_, 1, 0)
    if 1 in label_:
        print("Annomaly occured at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def simulate_detecting_anomalies(model_name, test_data):  # for simulating anomaly detection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    saved_model = torch.load(model_name + ".pt", device)
    saved_model.device = device

    seq_num = len(test_data) // saved_model.seq_len

    for i in range(seq_num):
        data = test_data[i * saved_model.seq_len: (i + 1) * saved_model.seq_len]
        score = saved_model.decision_function(data)
        label_ = np.where(score > saved_model.threshold_, 1, 0)
        if 1 in label_:
            print("Anomaly occured at", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        time.sleep(0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="train", help="whether to train before detect anomalies")
    parser.add_argument("--data", type=str, default="PSM", help="train, test data should be inside this path")
    parser.add_argument("--model_name", type=str, default="timesnet", help="which model to use")
    opt = parser.parse_args()

    if opt.task == "train":
        X_train = np.load("./" + opt.data + "/X_train.npy")
        train_TimesNet(X_train, opt.model_name)

    X_test = np.load("./" + opt.data + "/X_test.npy")

    """
    For real time anomaly detection with real data, use the first of the codes below
    - delete the annotation mark and add one to the second code
    
    For simulating anomaly detection with test data, use the second of the codes below
    - no need to change
    """
    print("---Start detecting anomalies---")
    # detect_anomalies(opt.model_name, X_test)
    simulate_detecting_anomalies(opt.model_name, X_test)
