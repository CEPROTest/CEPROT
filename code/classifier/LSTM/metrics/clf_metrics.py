from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from metrics import BaseClfMetric


class ClfMetric(BaseClfMetric):
    def __init__(self, threshold: float = 0.5):
        """
        :param threshold: The threshold for classification
        """
        super().__init__()
        self.threshold = threshold

    def eval(self, probs: np.array, labels: np.array) -> Dict:
        """
        :param probs: the prob of each class for each sample
        :param labels: true labels
        :return: precision, recall, f1, roc_auc, prc_auc
        """
        # calculate TPR, FPR, TNR, FNR & AUC
        assert probs.shape[0] == labels.shape[0]
        preds = (probs[:, 1] > self.threshold).astype(int)
        TP, FP, TN, FN = 0, 0, 0, 0
        idx = 0
        TPC = []
        FPC = []
        for pred, label in zip(preds, labels):
            if pred == label:
                if pred == 1:
                    TP += 1
                    TPC.append(idx)
                else:
                    TN += 1
            else:
                if pred == 1:
                    FP += 1
                    FPC.append(idx)
                else:
                    FN += 1
            idx += 1
        roc_auc = roc_auc_score(labels, probs[:, -1])
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2 * TP) / (2 * TP + FP + FN)
        pres, recs, thres = precision_recall_curve(labels, probs[:, -1])
        prc_auc = auc(recs, pres)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "prc_auc": prc_auc,
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "TPC":TPC,
            "FPC":FPC,
            "TPC_len":len(TPC),
            "FPC_len":len(FPC)
        }
