import torch
import torch.nn as nn

import numpy as np
from math import ceil

from scipy.optimize import linear_sum_assignment


def angular_distance(a, b, eps=1e-8):
    """

    :param a:
    :param b:
    :return:
    """
    a_norm = a / (np.linalg.norm(a) + eps)
    b_norm = b / (np.linalg.norm(b) + eps)
    sigma = 2 * np.arcsin(np.linalg.norm(a_norm - b_norm) / 2) * 180 / np.pi
    return sigma


def is_duplicate(a, b, duplicate_threshold):
    """

    :param a:
    :param b:
    :param duplicate_threshold:
    :return:
    """
    return angular_distance(a, b) <= duplicate_threshold


class Evaluation:

    def __init__(self, num_classes=13, prob_threshold=0.5, error_threshold=20, duplicate_threshold=15):
        self.num_classes = num_classes
        self.prob_threshold = prob_threshold
        self.error_threshold = error_threshold
        self.duplicate_threshold = duplicate_threshold
        self.reset()

    def reset(self):
        """

        :return:
        """
        num_classes = self.num_classes
        self.tp = np.zeros((num_classes,))
        self.fp = np.zeros((num_classes,))
        self.fn = np.zeros((num_classes,))
        self.kc = np.zeros((num_classes,))
        self.n = np.zeros((num_classes,))

        # Class-Dependent Localization Error and Recall
        self.le_c = [[] for _ in range(num_classes)]

        # Substitutions, Insertions, and Deletions
        self.s = 0
        self.d = 0
        self.i = 0

    def calculate_batch_statistics(self, est_multiaccdoa, true_multiaccdoa):
        """
        Calculates comparison statistics for SELD

        See also https://arxiv.org/pdf/2009.02792.pdf

        :return:
        """

        # Get Length Constants from Shape
        num_multiaccdoa, _, num_classes, frames = est_multiaccdoa.shape

        # Count Up TP, FP, and FN over 1s Intervals (10 100ms frames per interval)
        num_intervals = int(ceil(frames / 10))

        # Loop Through Each 1s Interval
        for interval in range(num_intervals):


            for c in range(num_classes):  # Each Class in the Interval
                fp_i = 0
                fn_i = 0

                for frame in range(10):  # Each 100ms Frame In Interval
                    pred_occurred_lists = []
                    true_occurred_lists = []
                    all_pred_trues = []
                    hungarians = []

                    pred_occurred_list = []
                    true_occurred_list = []

                    index = interval * 10 + frame
                    if index >= frames:
                        break  # Leave Loop If Last Frame Does Not Have 10 100ms intervals

                    # Multi-ACCDOA slices for this class and interval
                    est_ma_c = est_multiaccdoa[:, :, c, index]
                    true_ma_c = true_multiaccdoa[:, :, c, index]

                    # If A MultiACCDOA Vector has a Magnitude above the threshold, Then A Prediction has occurred
                    pred_vectors = np.linalg.norm(est_ma_c, axis=1) >= self.prob_threshold
                    if pred_vectors.any():
                        est_ma_c_real = est_ma_c[pred_vectors, :]
                        pred_occurred_list_temp = []
                        for multi in range(est_ma_c_real.shape[0]):
                            same = False
                            for pred in pred_occurred_list_temp:
                                if is_duplicate(pred, est_ma_c_real[multi], duplicate_threshold=self.duplicate_threshold):
                                    same = True
                                    break
                            if not same:
                                pred_occurred_list_temp.append(est_ma_c_real[multi])
                        pred_occurred_list.extend(pred_occurred_list_temp)

                    # If A MultiACCDOA Vector has a Magnitude above the threshold, Then a true event has occurred
                    true_vectors = np.linalg.norm(true_ma_c, axis=1) >= self.prob_threshold
                    if true_vectors.any():
                        true_ma_c_real = true_ma_c[true_vectors, :]
                        true_occurred_list_temp = []
                        for multi in range(true_ma_c_real.shape[0]):
                            same = False
                            for true in true_occurred_list_temp:
                                if is_duplicate(true, true_ma_c_real[multi], duplicate_threshold=self.duplicate_threshold):
                                    same = True
                                    break
                            #if not same:
                            true_occurred_list_temp.append(true_ma_c_real[multi])
                        true_occurred_list.extend(true_occurred_list_temp)

                    # Computer Minimizing Assignment of Predicted and True Occurrences
                    hungarian = np.zeros((len(pred_occurred_list), len(true_occurred_list)))
                    for i, pred in enumerate(pred_occurred_list):
                        for j, true in enumerate(true_occurred_list):
                            hungarian[i][j] = angular_distance(pred, true)
                    preds, trues = linear_sum_assignment(hungarian)

                    pred_occurred_lists.append(pred_occurred_list)
                    true_occurred_lists.append(true_occurred_list)
                    all_pred_trues.append((preds, trues))
                    hungarians.append(hungarian)

                    p_ci = np.sum([len(pred_occurred_list) for pred_occurred_list in pred_occurred_lists])
                    r_ci = np.sum([len(true_occurred_list) for true_occurred_list in true_occurred_lists])

                    fn_ci = max(0, r_ci - p_ci)
                    fp_ci = max(0, p_ci - r_ci)
                    k_ci = np.sum([len(preds) for preds, trues in all_pred_trues])

                    fp_ci_threshold = np.sum([np.sum(hungarian[preds, trues] > self.error_threshold)
                                              for (preds, trues), hungarian in zip(all_pred_trues, hungarians)])

                    self.fn[c] += fn_ci
                    self.fp[c] += fp_ci + fp_ci_threshold
                    self.tp[c] += k_ci - fp_ci_threshold
                    self.n[c] += r_ci
                    self.kc[c] += k_ci

                    fn_i += fn_ci
                    fp_i += fp_ci + fp_ci_threshold

                    for (preds, trues), hungarian in zip(all_pred_trues, hungarians):
                        self.le_c[c].extend(hungarian[preds, trues].tolist())
                    # lr_c[c].append(k_ci / (k_ci + fn_ci))

                # Calculate Substitutions, Insertions, and Deletions
                self.s += np.minimum(fn_i, fp_i)
                self.d += np.maximum(0, fn_i - fp_i)
                self.i += np.maximum(0, fp_i - fn_i)

    def calculate_summary_statics(self):
        # Calculate ER metric
        er = (self.s + self.d + self.i) / self.n.sum()

        # Calculate F-Score
        f_c = 2 * self.tp / (2 * self.kc + (self.fp + self.tp-self.kc) + self.fn)
        f = np.average(f_c)

        # Calculate LE
        le_c = np.array([np.average(o) for o in self.le_c]) # / kc
        le_c[np.isnan(le_c)] = 180
        le = np.average(le_c)

        # Calculate LR
        lr_c = self.kc / (self.kc + self.fn)
        lr_c[np.isnan(lr_c)] = 0
        lr = np.average(lr_c)

        return er, f, le, lr
