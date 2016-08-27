#!/usr/bin/env python

import numpy as np


# Sample-based measures:
# Hamming Loss, Accuracy, Precision, Recall, F1-Score:
def calculate_hamming_loss(real_labels, predicted_labels):
    num_samples, num_classes = real_labels.shape
    xor_arrays = np.logical_xor(real_labels, predicted_labels).astype('float')
    sum_rows = np.sum(xor_arrays, axis=1) / num_classes
    hamming_loss = sum_rows.sum() / num_samples
    return hamming_loss


def calculate_accuracy(real_labels, predicted_labels, average=True):
    num_samples = real_labels.shape[0]
    true_positive = np.sum(np.logical_and(real_labels, predicted_labels), axis=1).astype('float')
    tp_fn_fp = np.sum(np.logical_or(real_labels, predicted_labels), axis=1).astype('float')
    acc_per_sample = true_positive / tp_fn_fp
    accuracy = np.sum(acc_per_sample) / num_samples
    if average:
        return accuracy
    else:
        return acc_per_sample


def calculate_precision(real_labels, predicted_labels, average=True):
    num_samples = real_labels.shape[0]
    true_positive = np.sum(np.logical_and(real_labels, predicted_labels), axis=1).astype('float')
    num_real_labels = np.sum(real_labels, axis=1).astype('float')
    precision_per_sample = true_positive / num_real_labels
    precision = np.sum(precision_per_sample) / num_samples
    if average:
        return precision
    else:
        return precision_per_sample


def calculate_recall(real_labels, predicted_labels, average=True):
    num_samples = real_labels.shape[0]
    true_positive = np.sum(np.logical_and(real_labels, predicted_labels), axis=1).astype('float')
    num_predicted_labels = np.sum(predicted_labels, axis=1).astype('float')
    num_predicted_labels[num_predicted_labels == 0] = np.nan
    recall_per_sample = true_positive / num_predicted_labels
    recall = np.nansum(recall_per_sample) / num_samples
    if average:
        return recall
    else:
        recall_per_sample[np.isnan(recall_per_sample)] = 0
        return recall_per_sample


def calculate_f1_score(real_labels, predicted_labels):
    num_samples = real_labels.shape[0]
    recall = calculate_recall(real_labels, predicted_labels, average=False)
    precision = calculate_precision(real_labels, predicted_labels, average=False)
    f1_num = 2 * recall * precision
    f1_den = recall + precision
    f1_den[f1_den == 0] = np.nan
    f1_score = f1_num / f1_den
    f1_score = np.nansum(f1_score) / num_samples
    return f1_score


# Label-based measures:
# Macro Precision, Recall and F1-Score
def calculate_macro_precision(real_labels, predicted_labels, average=True):
    num_classes = real_labels.shape[1]
    true_positive = np.sum(np.logical_and(real_labels, predicted_labels), axis=0).astype('float')
    false_positive = np.sum(np.logical_and(np.logical_not(real_labels), predicted_labels), axis=0).astype('float')
    precision_den = true_positive + false_positive
    precision_den[precision_den == 0] = np.nan
    macro_precision_per_class = true_positive / precision_den
    macro_precision = np.nansum(macro_precision_per_class) / num_classes
    if average:
        return macro_precision
    else:
        macro_precision_per_class[np.isnan(macro_precision_per_class)] = 0
        return macro_precision_per_class


def calculate_macro_recall(real_labels, predicted_labels, average=True):
    num_classes = real_labels.shape[1]
    true_positive = np.sum(np.logical_and(real_labels, predicted_labels), axis=0).astype('float')
    false_negative = np.sum(np.logical_and(real_labels, np.logical_not(predicted_labels)), axis=0).astype('float')
    recall_den = true_positive + false_negative
    recall_den[recall_den == 0] = np.nan
    macro_recall_per_class = true_positive / recall_den
    macro_recall = np.nansum(macro_recall_per_class) / num_classes
    if average:
        return macro_recall
    else:
        macro_recall_per_class[np.isnan(macro_recall_per_class)] = 0
        return macro_recall_per_class


def calculate_macro_f1_score(real_labels, predicted_labels, per_class=False):
    num_classes = real_labels.shape[1]
    recall = calculate_macro_recall(real_labels, predicted_labels, average=False)
    precision = calculate_macro_precision(real_labels, predicted_labels, average=False)
    f1_num = 2 * recall * precision
    f1_den = recall + precision
    f1_den[f1_den == 0] = np.nan
    f1_score = f1_num / f1_den
    if per_class:
        f1_score[np.isnan(f1_score)] = 0
        return f1_score
    else:
        f1_score = np.nansum(f1_score) / num_classes
        return f1_score


# Overall measures:
# Micro F1-Score, Precision and Recall
def calculate_micro_precision(real_labels, predicted_labels):
    true_positive = np.sum(np.logical_and(real_labels, predicted_labels)).astype('float')
    false_positive = np.sum(np.logical_and(np.logical_not(real_labels), predicted_labels)).astype('float')
    precision_den = true_positive + false_positive
    if precision_den == 0:
        precision = 0
    else:
        precision = true_positive / precision_den
    return precision


def calculate_micro_recall(real_labels, predicted_labels):
    true_positive = np.sum(np.logical_and(real_labels, predicted_labels)).astype('float')
    false_negative = np.sum(np.logical_and(real_labels, np.logical_not(predicted_labels))).astype('float')
    recall_den = true_positive + false_negative
    if recall_den == 0:
        recall = 0
    else:
        recall = true_positive / recall_den
    return recall


def calculate_micro_f1_score(real_labels, predicted_labels):
    precision = calculate_micro_precision(real_labels, predicted_labels)
    recall = calculate_micro_recall(real_labels, predicted_labels)
    f1_num = 2 * precision * recall
    f1_den = precision + recall
    if f1_den == 0:
        f1_score = 0
    else:
        f1_score = f1_num / f1_den
    return f1_score


def calculate_micro_f1_score_from_precision_recall(precision, recall):
    f1_num = 2 * precision * recall
    f1_den = precision + recall
    if f1_den == 0:
        f1_score = 0
    else:
        f1_score = f1_num / f1_den
    return f1_score
