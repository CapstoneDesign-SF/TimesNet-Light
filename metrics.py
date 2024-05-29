import numpy as np
from sklearn import metrics


def auc_score(y_true, y_score):
    """
    ROC curve 아래 면적으로 AUC score 계산
    """
    return metrics.roc_auc_score(y_true, y_score)


def pr_score(y_true, y_score): # 함수 이름 바꾸기 auc 아님!!
    """
    Precision-Recall curve 아래 면적 계산
    """
    return metrics.average_precision_score(y_true, y_score)


def point_adjustment(y_true, y_score):
    """
    * [Zhihan Li et al. KDD21]의 코드를 참고, 수정함 *
    이상 데이터가 발생하면 해당 시점으로부터 특정 기간동안 지속되는 특징에 따라
    해당 타임 포인트의 최댓값으로 내부 값들을 재정의
    """
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


def get_best_f1(label, score):
    """
    가장 높은 f1 점수 기록
    """
    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    return best_f1, best_p, best_r


def ts_metrics(y_true, y_score):
    """
    AUC score, Precision-Recall score, f1, precision, recall 계산
    """
    best_f1, best_p, best_r = get_best_f1(y_true, y_score)
    return auc_score(y_true, y_score), pr_score(y_true, y_score), best_f1, best_p, best_r