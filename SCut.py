#!/usr/bin/env python

import argparse
import numpy as np
from performance_measures import *


# scut_thresholding calculates the thresholds for each category
def scut_thresholding(scores, labels):
    num_labels = scores.shape[1]
    set_length = scores.shape[0]
    # print set_length
    # Calculate one threshold for each label:
    thresholds = np.zeros(num_labels)
    for lbl in xrange(num_labels):
        scores_lbl = scores[:, lbl]
        real_lbl = labels[:, lbl].astype('int')
        opt_th = 0.0
        min_mse = np.Inf
        for th in np.linspace(0, 1, 1000):
            pred_lbl = (scores_lbl >= th).astype('int')
            mse = (np.power(pred_lbl - real_lbl, 2).sum()).astype('float')/set_length
            # print "Threshold: %f - MSE: %f" % (th, mse)
            if mse < min_mse:
                min_mse = mse
                opt_th = th
        print "Label %d - Threshold: %f - Min MSE: %f" % (lbl+1, opt_th, min_mse)
        thresholds[lbl] = opt_th
    return thresholds


def apply_thresholds(scores, thresholds):
    num_labels = scores.shape[1]
    set_length = scores.shape[0]
    pred_labels = np.zeros((set_length, num_labels), dtype='uint8')
    for lbl in xrange(num_labels):
        scores_lbl = scores[:, lbl]
        th = thresholds[lbl]
        pred_labels[:, lbl] = (scores_lbl >= th).astype('uint8')
    return pred_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates SCut thresholding over the data in the filename.')
    parser.add_argument('scores', nargs=1, type=file, help='Filename of the data with the scores.')
    parser.add_argument('labels', nargs=1, type=file, help='Filename of the data with the labels.')
    args = parser.parse_args()
    scores_file = args.scores[0]
    labels_file = args.labels[0]
    # Read data from files into a numpy arrays:
    scores = np.loadtxt(scores_file, dtype='float64', delimiter=',')
    labels = np.loadtxt(labels_file, dtype='uint8', delimiter=',')
    num_samps, num_cats = scores.shape
    # Shuffle indices:
    indices = range(num_samps)
    np.random.seed(2706)
    rand_indices = np.random.choice(indices, num_samps, replace=False)
    # Separate 30% for test and 70% for validation:
    num_test_samps = int(0.7 * num_samps)
    # num_val_samps = num_samps - num_test_samps
    test_indices = rand_indices[:num_test_samps]
    val_indices = rand_indices[num_test_samps:]
    test_set = scores[test_indices, :]
    test_set_labels = labels[test_indices, :]
    val_set = scores[val_indices, :]
    val_set_labels = labels[val_indices, :]
    # SCut thresholding:
    thresholds = scut_thresholding(val_set, val_set_labels)
    print "Thresholds: %s" % thresholds
    # Perform prediction on test set:
    pred_testset_labels = apply_thresholds(test_set, thresholds)
    hloss = calculate_hamming_loss(test_set_labels, pred_testset_labels)
    acc = calculate_accuracy(test_set_labels, pred_testset_labels)
    precision = calculate_macro_precision(test_set_labels, pred_testset_labels)
    recall = calculate_macro_recall(test_set_labels, pred_testset_labels)
    fscore_micro = calculate_micro_f1_score(test_set_labels, pred_testset_labels)
    fscore_macro = calculate_macro_f1_score(test_set_labels, pred_testset_labels)
    print "Hamming Loss: %f\tAccuracy: %f\tMacro Precision: %f\tMacro Recall: %f\tMacro F1-Score: %f" \
              % (hloss, acc, precision, recall, fscore_macro)