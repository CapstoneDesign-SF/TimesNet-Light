import numpy as np
from sklearn import metrics


def auc_roc(y_true, y_score):
    return metrics.roc_auc_score(y_true, y_score)


def auc_pr(y_true, y_score):
    return metrics.average_precision_score(y_true, y_score)


def point_adjustment(y_true, y_score):
    score = y_score.copy()
    assert len(score) == len(y_true)
    splits = np.where(y_true[1:] != y_true[:-1])[0] + 1
    is_anomaly = y_true[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(y_true)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score


def ts_metrics(y_true, y_score):
    best_f1, best_p, best_r = get_best_f1(y_true, y_score)
    return auc_roc(y_true, y_score), auc_pr(y_true, y_score), best_f1, best_p, best_r


def get_best_f1(label, score):
    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    return best_f1, best_p, best_r
