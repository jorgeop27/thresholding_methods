#!/usr/bin/env python

import argparse
import numpy as np
from performance_measures import *


def mcut_thresholding(dataset):
    num_samps = dataset.shape[0]
    sorted_dataset = np.sort(dataset)
    diff_scores = np.diff(sorted_dataset, axis=1)
    # Get the index of the maximum differences:
    maxdiff_1 = np.argmax(diff_scores, axis=1).reshape(1, -1).T
    maxdiff_2 = maxdiff_1 + 1
    row_index = np.array(range(num_samps)).reshape(1, -1).T
    mcut = (sorted_dataset[(row_index, maxdiff_1)] + sorted_dataset[(row_index, maxdiff_2)]) / 2
    mcut = np.reshape(mcut, mcut.shape[0])
    pred_labels = np.greater(dataset, mcut[:, None]).astype(int)
    return pred_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates MCut thresholding over the data in the filename.')
    parser.add_argument('scores', nargs=1, type=file, help='Filename of the data with the scores.')
    parser.add_argument('labels', nargs=1, type=file, help='Filename of the data with the labels.')
    args = parser.parse_args()
    scores_file = args.scores[0]
    labels_file = args.labels[0]
    # Read data from files into a numpy arrays:
    scores = np.loadtxt(scores_file, dtype='float64', delimiter=',')
    labels = np.loadtxt(labels_file, dtype='uint8', delimiter=',')
    num_samps, num_cats = scores.shape

    pred_labels = mcut_thresholding(scores)
    hloss = calculate_hamming_loss(labels, pred_labels)
    acc = calculate_accuracy(labels, pred_labels)
    precision = calculate_macro_precision(labels, pred_labels)
    recall = calculate_macro_recall(labels, pred_labels)
    fscore_micro = calculate_micro_f1_score(labels, pred_labels)
    fscore_macro = calculate_macro_f1_score(labels, pred_labels)
    print "Hamming Loss: %f\tAccuracy: %f\tMacro Precision: %f\tMacro Recall: %f\tMacro F1-Score: %f" \
          % (hloss, acc, precision, recall, fscore_macro)
