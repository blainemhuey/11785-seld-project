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


def calculate_statistics(est_multiaccdoa, true_multiaccdoa, duplicate_threshold=1, prob_threshold=0.5,
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
    n = np.zeros((num_classes,))

    # Class-Dependent Localization Error and Recall
    le_c = [[] for _ in range(num_classes)]
    lr_c = [[] for _ in range(num_classes)]

    # Substitutions, Insertions, and Deletions
    s, d, i = 0, 0, 0

    # Loop Through Each 1s Interval
    for interval in range(num_intervals):
        tp_i = np.zeros((num_classes,))
        fp_i = np.zeros((num_classes,))
        fn_i = np.zeros((num_classes,))
        kc_i = np.zeros((num_classes,))
        n_i = np.zeros((num_classes,))

        for c in range(num_classes):  # Each Class in the Interval

            for frame in range(10):  # Each 100ms Frame In Interval
                pred_occurred_list = []
                true_occurred_list = []

                index = interval * 10 + frame
                if index >= frames:
                    break  # Leave Loop If Last Frame Does Not Have 10 100ms intervals

                # Multi-ACCDOA slices for this class and interval
                est_ma_c = est_multiaccdoa[:, :, c, index]
                true_ma_c = true_multiaccdoa[:, :, c, index]

                # If A MultiACCDOA Vector has a Magnitude above the threshold, Then A Prediction has occurred
                pred_vectors = np.linalg.norm(est_ma_c, axis=1) >= prob_threshold
                if pred_vectors.any():
                    est_ma_c_real = est_ma_c[pred_vectors, :]
                    pred_occurred_list_temp = []
                    for multi in range(est_ma_c_real.shape[0]):
                        same = False
                        for pred in pred_occurred_list_temp:
                            if is_duplicate(pred, est_ma_c_real[multi], duplicate_threshold=duplicate_threshold):
                                same = True
                                break
                        if not same:
                            pred_occurred_list_temp.append(est_ma_c_real[multi])
                    pred_occurred_list.extend(pred_occurred_list_temp)

                # If A MultiACCDOA Vector has a Magnitude above the threshold, Then a true event has occurred
                true_vectors = np.linalg.norm(true_ma_c, axis=1) >= prob_threshold
                if true_vectors.any():
                    true_ma_c_real = true_ma_c[true_vectors, :]
                    true_occurred_list_temp = []
                    for multi in range(true_ma_c_real.shape[0]):
                        same = False
                        for true in true_occurred_list_temp:
                            if is_duplicate(true, true_ma_c_real[multi], duplicate_threshold=duplicate_threshold):
                                same = True
                                break
                        if not same:
                            true_occurred_list_temp.append(true_ma_c_real[multi])
                    true_occurred_list.extend(true_occurred_list_temp)

                # Computer Minimizing Assignment of Predicted and True Occurrences
                hungarian = np.zeros((len(pred_occurred_list), len(true_occurred_list)))
                for i, pred in enumerate(pred_occurred_list):
                    for j, true in enumerate(true_occurred_list):
                        hungarian[i][j] = angular_distance(pred, true)
                preds, trues = linear_sum_assignment(hungarian)

                p_ci = len(pred_occurred_list)
                r_ci = len(true_occurred_list)

                fn_ci = max(0, r_ci - p_ci)
                fp_ci = max(0, p_ci - r_ci)
                k_ci = len(preds)

                fp_ci_threshold = np.sum(hungarian[preds, trues] < error_threshold)

                fn[c] += fn_ci
                fp[c] += fp_ci + fp_ci_threshold
                tp[c] += k_ci - fp_ci_threshold
                n[c] += r_ci
                kc[c] += k_ci

                fn_i[c] += fn_ci
                fp_i[c] += fp_ci + fp_ci_threshold
                tp_i[c] += k_ci - fp_ci_threshold
                n_i[c] += r_ci
                kc_i[c] += k_ci

                le_c[c].extend(hungarian[preds, trues].tolist())
                # lr_c[c].append(k_ci / (k_ci + fn_ci))

        # Calculate Substitutions, Insertions, and Deletions
        s += np.minimum(fn_i.sum(), fp_i.sum())
        d += np.maximum(0, fn_i.sum() - fp_i.sum())
        i += np.maximum(0, fp_i.sum() - fn_i.sum())

    return tp, fp, fn, n, kc, le_c, s, d, i


def calculate_summary_statics(tp, fp, fn, n, kc, oc, s, d, i):
    # Calculate F-Score
    f_c = 2 * tp / (2 * kc + fp + tp-kc + fn)
    f = np.average(f_c)

    # Calculate ER metric
    er = (s + d + i) / n.sum()
    #er = np.average(er_c)

    # Calculate LE
    le_c = oc # / kc
    le_c[np.isnan(le_c)] = 180
    le = np.average(le_c)

    # Calculate LR
    lr_c = kc / (kc + fn)
    lr_c[np.isnan(lr_c)] = 0
    lr = np.average(lr_c)

    return f, er, le, lr
