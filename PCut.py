#!/usr/bin/env python

import argparse
import numpy as np
from performance_measures import *


def prior_prob_per_cat(training_labels, num_cats):
    num_samples = training_labels.shape[0]
    priorprobs = np.zeros((num_cats, 1), dtype='float')
    for i in xrange(num_cats):
        num_curr_label = training_labels[:, i].sum()
        priorprobs[i-1] = float(num_curr_label)/num_samples
    return priorprobs


def pcut_thresholding(scores, prior_probs, x):
    num_labels = scores.shape[1]
    set_length = scores.shape[0]
    pred_labels = np.zeros((set_length, num_labels))
    # Sort data per category (low to high) and get the sorted indices:
    sorted_scores = scores.argsort(axis=0)
    for cat in xrange(num_labels):
        prior = prior_probs[cat]
        k = int(np.floor(prior*x*num_labels))
        # print k
        cat_scores = sorted_scores[:, cat]
        cat_scores = np.flipud(cat_scores)
        indices = cat_scores[:k]
        pred_labels[indices, cat] = 1
    return pred_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates PCut thresholding over the data in the filename.')
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
    # Separate 50% for train, 30% for validation and 20% for testing:
    num_train_samps = int(0.5 * num_samps)
    num_val_samps = int(0.3 * num_samps)
    # num_test_samps = num_samps - num_train_samps - num_val_samps
    train_indices = rand_indices[:num_train_samps]
    val_indices = rand_indices[num_train_samps:(num_train_samps + num_val_samps)]
    test_indices = rand_indices[(num_train_samps + num_val_samps):]
    train_set = scores[train_indices, :]
    train_set_labels = labels[train_indices, :]
    val_set = scores[val_indices, :]
    val_set_labels = labels[val_indices, :]
    test_set = scores[test_indices, :]
    test_set_labels = labels[test_indices, :]
    # Prior probabilities:
    prior_probs = prior_prob_per_cat(train_set_labels, num_cats)
    max_fscore = 0
    x_optim = 0
    for x in xrange(0, num_val_samps, num_val_samps/100):
        pred_val_labels = pcut_thresholding(val_set, prior_probs, x)
        hloss = calculate_hamming_loss(val_set_labels, pred_val_labels)
        acc = calculate_accuracy(val_set_labels, pred_val_labels)
        precision = calculate_macro_precision(val_set_labels, pred_val_labels)
        recall = calculate_macro_recall(val_set_labels, pred_val_labels)
        fscore_macro = calculate_macro_f1_score(val_set_labels, pred_val_labels)
        print "X: %d\tHamming Loss: %f\tAccuracy: %f\tMacro Precision: %f\tMacro Recall: %f\tMacro F1-Score: %f" \
              % (x, hloss, acc, precision, recall, fscore_macro)
        if fscore_macro > max_fscore:
            max_fscore = fscore_macro
            x_optim = x
    # Optimum value for X:
    print "Optimum X: %d" % x_optim
    # Perform prediction on test set:
    pred_testset_labels = pcut_thresholding(test_set, prior_probs, x_optim)
    fscore_micro = calculate_micro_f1_score(test_set_labels, pred_testset_labels)
    print "Test Micro F-Score: %f" % fscore_micro
    fscore_macro = calculate_macro_f1_score(test_set_labels, pred_testset_labels)
    print "Test Macro F-Score: %f" % fscore_macro
