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
    return angular_distance(a, b) < duplicate_threshold


def calculate_statistics(est_multiaccdoa, true_multiaccdoa, duplicate_threshold=15, prob_threshold=0.5,
                         error_threshold=20):
    """
    Calculates comparison statistics for SELD

    See also https://arxiv.org/pdf/2009.02792.pdf

    :return:
    """

    # Get Length Constants from Shape
    num_multiaccdoa, _, num_classes, frames = est_multiaccdoa.shape

    # Count Up TP, FP, and FN over 1s Intervals (10 100ms frames per interval)
    num_intervals = int(ceil(frames / 10))
    tp = np.zeros((num_classes,))
    fp = np.zeros((num_classes,))
    fn = np.zeros((num_classes,))
    kc = np.zeros((num_classes,))
    rc = np.zeros((num_classes,))
    n = np.zeros((num_classes,))

    # Class-Dependent Localization Error and Recall
    le_c = [[] for _ in range(num_classes)]
    lr_c = [[] for _ in range(num_classes)]

    # Loop Through Each 1s Interval
    for interval in range(num_intervals):
        for c in range(num_classes):  # Each Class in the Interval
            pred_occurred_list = []
            true_occurred_list = []

            for test in range(num_multiaccdoa):
                for frame in range(10):  # Each 100ms Frame In Interval
                    index = interval * 10 + frame
                    if index >= frames:
                        break  # Leave Loop If Last Frame Does Not Have 10 100ms intervals

                    # Multi-ACCDOA slices for this class and interval
                    est_ma_c = est_multiaccdoa[test, :, c, index]
                    true_ma_c = true_multiaccdoa[test, :, c, index]

                    # If A MultiACCDOA Vector has a Magnitude above the threshold, Then A Prediction has occurred
                    if np.linalg.norm(est_ma_c) > prob_threshold:
                        pred_occurred_list.append(est_ma_c)

                    # If A MultiACCDOA Vector has a Magnitude above the threshold, Then a true event has occurred
                    if np.linalg.norm(true_ma_c) >= prob_threshold:
                        true_occurred_list.append(true_ma_c)

            # Computer Minimizing Assignment of Predicted and True Occurrences
            hungarian = np.zeros((len(pred_occurred_list), len(true_occurred_list)))
            for i, pred in enumerate(pred_occurred_list):
                for j, true in enumerate(true_occurred_list):
                    hungarian[i][j] = angular_distance(pred, true)
            preds, trues = linear_sum_assignment(hungarian)

            r_ci = len(pred_occurred_list)
            p_ci = len(true_occurred_list)

            fn_ci = max(0, r_ci - p_ci)
            fp_ci = max(0, p_ci - r_ci)
            k_ci = len(preds)

            fp_ci_threshold = np.sum([1 for pred, true, in zip(preds, trues)
                                      if angular_distance(pred_occurred_list[pred], true_occurred_list[true]) >= error_threshold])

            fn[c] += fn_ci
            fp[c] += fp_ci + fp_ci_threshold
            tp[c] += k_ci - fp_ci_threshold
            n[c] += p_ci
            kc[c] += k_ci
            rc[c] += r_ci

            le_c[c].extend([angular_distance(pred_occurred_list[pred], true_occurred_list[true])
                            for pred, true in zip(preds, trues)])
            # lr_c[c].append(k_ci / (k_ci + fn_ci))

    # Calculate F-Score
    f_c = 2 * tp / (2 * tp + fp + fn)
    f = np.average(f_c)

    # Calculate Substitutions, Insertions, and Deletions
    s = np.min(np.stack((fn, fp), axis=1), axis=1)
    d = np.max(np.stack((np.zeros(fn.shape), (fn - fp)), axis=1), axis=1)
    i = np.max(np.stack((np.zeros(fn.shape), (fp - fn)), axis=1), axis=1)

    # Calculate ER metric
    er_c = (s + d + i) / n
    er = np.average(er_c)

    # Calculate LE
    le_c = [np.average(le_ci) for le_ci in le_c]
    le = np.average(le_c)

    # Calculate LR
    lr_c = kc / rc
    lr = np.average(lr_c)

    return f, er, le, lr
