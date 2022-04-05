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
    tp = np.zeros((num_intervals,))
    fp = np.zeros((num_intervals,))
    fn = np.zeros((num_intervals,))
    n = np.zeros((num_intervals,))

    # Class-Dependent Localization Error and Recall
    le_c = [[] for _ in range(num_classes)]
    lr_c = [[] for _ in range(num_classes)]

    # Loop Through Each 1s Interval
    for interval in range(num_intervals):
        pred_occurred = np.zeros((num_classes,), dtype=bool)
        true_occurred = np.zeros((num_classes,), dtype=bool)
        accurate = np.zeros((num_classes,), dtype=bool)

        for c in range(num_classes):  # Each Class in the Interval
            pred_occurred_list = []
            true_occurred_list = []

            for test_pred in range(num_multiaccdoa):
                for test_true in range(num_multiaccdoa):
                    for frame in range(10):  # Each 100ms Frame In Interval
                        index = interval * 10 + frame
                        if index >= frames:
                            break  # Leave Loop If Last Frame Does Not Have 10 100ms intervals

                        # Multi-ACCDOA slices for this class and interval
                        est_ma_c = est_multiaccdoa[test_pred, :, c, index]
                        true_ma_c = true_multiaccdoa[test_true, :, c, index]

                        # If Any MultiACCDOA Vector has a Magnitude above the threshold, Then A Prediction has occurred
                        if np.linalg.norm(est_ma_c) > prob_threshold:
                            pred_occurred[c] |= True
                            pred_occurred_list.append(est_ma_c)

                        # If Any MultiACCDOA Vector has a Magnitude above the threshold, Then a true event has occurred
                        if np.linalg.norm(true_ma_c) >= prob_threshold:
                            true_occurred[c] |= True
                            true_occurred_list.append(true_ma_c)

                        # Constrain that at least one vector in the class must be within threshold degrees of truth
                        if angular_distance(est_ma_c, true_ma_c) < error_threshold and \
                                np.linalg.norm(est_ma_c) >= prob_threshold and \
                                np.linalg.norm(true_ma_c) >= prob_threshold:
                            accurate[c] |= True

            # Computer Minimizing Assignment of Predicted and True Occurrences
            hungarian = np.zeros((len(pred_occurred_list), len(true_occurred_list)))
            for i, pred in enumerate(pred_occurred_list):
                for j, true in enumerate(true_occurred_list):
                    hungarian[i][j] = angular_distance(pred, true)
            preds, trues = linear_sum_assignment(hungarian)

            le_c[c].append([angular_distance(pred, true) for pred, true in zip(preds, trues)])
            lr_c[c].append(len(preds) / len)



        # Increment Interval Count of True/False Positives/Negatives
        tp[interval] += np.sum(np.logical_and(np.logical_and(pred_occurred, true_occurred), accurate))
        fp[interval] += np.sum(np.logical_and(pred_occurred, np.logical_not(true_occurred))) + \
                        np.sum(np.logical_and(np.logical_and(pred_occurred, true_occurred), np.logical_not(accurate)))
        fn[interval] += np.sum(np.logical_and(np.logical_not(pred_occurred), true_occurred))
        n[interval] += np.sum(true_occurred)

    # Calculate F-Score
    f = 2 * np.sum(tp) / (2 * np.sum(tp) + np.sum(fp) + np.sum(fn))

    # Calculate Substitutions, Insertions, and Deletions
    s = np.min(np.stack(fn, fp, axis=1), axis=1)
    d = np.max(np.stack(np.zeros(fn.shape), (fn - fp), axis=1), axis=1)
    i = np.max(np.stack(np.zeros(fn.shape), (fp - fn), axis=1), axis=1)

    # Calculate ER metric
    er = (np.sum(s) + np.sum(d) + np.sum(i)) / np.sum(n)

    # Calculate LE
    le = np.average(le_c)

    # Calculate LR
    lr = np.average(lr_c)

    return f, er, le, lr
