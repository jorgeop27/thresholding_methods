#!/usr/bin/env python

import argparse
import numpy as np


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


def compute_fscore_macro(real_labels, pred_labels):
    num_cats = real_labels.shape[1]
    fscore = 0.0
    for cat in xrange(num_cats):
        cat_labels = real_labels[:, cat]
        cat_pred_labels = pred_labels[:, cat]
        sum_labels = cat_labels + cat_pred_labels
        tn = np.count_nonzero(sum_labels == 0)
        tp = np.count_nonzero(sum_labels == 2)
        sub_labels = cat_labels.astype('int') - cat_pred_labels.astype('int')
        fp = np.count_nonzero(sub_labels == -1)
        fn = np.count_nonzero(sub_labels == 1)
        if (tp + fp) == 0:
            prec = 0
        else:
            prec = (1.0 * tp) / (tp + fp)
        if (tp + fn) == 0:
            rec = 0
        else:
            rec = (1.0 * tp) / (tp + fn)
        if (prec + rec) == 0:
            fscore_cat = 0
        else:
            fscore_cat = (2 * prec * rec) / (prec + rec)
        fscore += fscore_cat
    fscore /= num_cats
    return fscore


def compute_fscore_micro(real_labels, pred_labels):
    num_cats = real_labels.shape[1]
    prec_num = 0.0
    prec_den = 0.0
    rec_num = 0.0
    rec_den = 0.0
    for cat in xrange(num_cats):
        cat_labels = real_labels[:, cat]
        cat_pred_labels = pred_labels[:, cat]
        sum_labels = cat_labels + cat_pred_labels
        tn = np.count_nonzero(sum_labels == 0)
        tp = np.count_nonzero(sum_labels == 2)
        sub_labels = cat_labels.astype('int') - cat_pred_labels.astype('int')
        fp = np.count_nonzero(sub_labels == -1)
        fn = np.count_nonzero(sub_labels == 1)
        prec_num += tp
        prec_den += (tp + fp)
        rec_num += tp
        rec_den += (tp + fn)
    if prec_den != 0:
        prec = (1.0 * prec_num) / prec_den
    else:
        prec = 0
    if rec_den != 0:
        rec = (1.0 * rec_num) / rec_den
    else:
        rec = 0
    if (prec + rec) != 0:
        fscore = (2 * prec * rec) / (prec + rec)
    else:
        fscore = 0
    return fscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates MCut thresholding over the data in the filename.')
    parser.add_argument('filename', nargs=1, type=file, help='Filename of the data with the scores.')
    parser.add_argument('num_categories', nargs=1, type=int, help='Number of categories in the previous data set')
    args = parser.parse_args()
    f = args.filename[0]
    num_cats = args.num_categories[0]
    # Read data from file f into a new numpy array:
    data_array = np.loadtxt(f, dtype=float, delimiter=',')
    num_samps = data_array.shape[0]
    dataset = data_array[:, :num_cats]
    truelabs = data_array[:, num_cats:].astype('uint8')
    true_labels = np.zeros((num_samps, num_cats + 1), dtype='uint8')
    row_index = np.array(range(num_samps)).reshape(1, -1).T
    true_labels[(row_index, truelabs)] = 1
    true_labels = true_labels[:, 1:]
    pred_labels = mcut_thresholding(dataset)
    # Compute Fscore (macro and micro):
    fscore_micro = compute_fscore_micro(true_labels, pred_labels)
    print "Test Micro F-Score: %f" % fscore_micro
    fscore_macro = compute_fscore_macro(true_labels, pred_labels)
    print "Test Macro F-Score: %f" % fscore_macro
