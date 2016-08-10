#!/usr/bin/env python

import argparse
import numpy as np
from performance_measures import *


def rcut_thresholding(scores, threshold):
    num_labels = scores.shape[1]
    set_length = scores.shape[0]
    pred_labels = np.zeros(scores.shape, dtype='uint8')
    # Get the indices of the sorted array of data:
    sort_index = np.argsort(scores, axis=1)
    # Reverse the order of the columns to get High to Low order:
    sort_index = np.fliplr(sort_index)
    labels = sort_index[:, :threshold]
    row_index = np.array(range(set_length)).reshape(1, -1).T
    pred_labels[(row_index, labels)] = 1
    return pred_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates Rank Cut thresholding over the data in the filename.')
    parser.add_argument('scores', nargs=1, type=file, help='Filename of the data with the scores.')
    parser.add_argument('labels', nargs=1, type=file, help='Filename of the data with the labels.')
    args = parser.parse_args()
    scores_file = args.scores[0]
    labels_file = args.labels[0]
    # Read data from files into a numpy arrays:
    scores = np.loadtxt(scores_file, dtype='float64', delimiter=',')
    labels = np.loadtxt(labels_file, dtype='uint8', delimiter=',')
    num_samps, num_cats = scores.shape
    print "Numb. of Samples: %d\t Numb. of Categories: %d" % (num_samps, num_cats)
    # Shuffle indices:
    indices = range(num_samps)
    np.random.seed(2706)
    rand_indices = np.random.choice(indices, num_samps, replace=False)
    # Separate 30% for test and 70% for validation:
    num_test_samps = int(0.3 * num_samps)
    test_indices = rand_indices[:num_test_samps]
    val_indices = rand_indices[num_test_samps:]
    test_set = scores[test_indices, :]
    test_set_labels = labels[test_indices, :]
    val_set = scores[val_indices, :]
    val_set_labels = labels[val_indices, :]
    max_fscore = 0
    t_optim = 0
    # for t in xrange(1, num_cats+1):
    for t in xrange(1, 101):
        pred_val_labels = rcut_thresholding(val_set, t)
        hloss = calculate_hamming_loss(val_set_labels, pred_val_labels)
        acc = calculate_accuracy(val_set_labels, pred_val_labels)
        precision = calculate_macro_precision(val_set_labels, pred_val_labels)
        recall = calculate_macro_recall(val_set_labels, pred_val_labels)
        fscore_micro = calculate_macro_f1_score(val_set_labels, pred_val_labels)
        print "Threshold: %d\tHamming Loss: %f\tAccuracy: %f\tMacro Precision: %f\tMacro Recall: %f\tMacro F1-Score: %f"\
              % (t, hloss, acc, precision, recall, fscore_micro)
        if fscore_micro > max_fscore:
            max_fscore = fscore_micro
            t_optim = t
    # Optimum value for T:
    print "Optimum T: %d" % t_optim
    # Perform prediction on test set:
    pred_testset_labels = rcut_thresholding(test_set, t_optim)
    fscore_micro = calculate_micro_f1_score(test_set_labels, pred_testset_labels)
    print "Test Micro F-Score: %f" % fscore_micro
    fscore_macro = calculate_macro_f1_score(test_set_labels, pred_testset_labels)
    print "Test Macro F-Score: %f" % fscore_macro
