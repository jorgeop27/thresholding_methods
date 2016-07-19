#!/usr/bin/env python

import argparse
import sys
import numpy as np


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
        # print "TN: %d, TP: %d, FN: %d, FP: %d" % (tn, tp, fn, fp)
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


def rcut_thresholding(scores, threshold):
    num_labels = scores.shape[1]
    set_length = scores.shape[0]
    pred_labels = np.zeros((set_length, num_labels), dtype='uint8')
    # Get the indices of the sorted array of data:
    sort_index = np.argsort(scores, axis=1)
    # Reverse the order of the columns to get High to Low order:
    sort_index = np.fliplr(sort_index)
    labels = sort_index[:, :threshold]
    row_index = np.array(range(set_length)).reshape(1, -1).T
    pred_labels[(row_index, labels)] = 1
    return pred_labels

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculates Rank Cut thresholding over the data in the filename.')
    parser.add_argument('filename', nargs=1, type=file, help='Filename of the data with the scores.')
    parser.add_argument('num_categories', nargs=1, type=int, help='Number of categories in the previous data set')
    args = parser.parse_args()
    f = args.filename[0]
    num_cats = args.num_categories[0]
    # Read data from file f into a new numpy array:
    data_array = np.loadtxt(f, dtype=float, delimiter=',')
    # Shuffle rows in data_array:
    np.random.seed(2706)
    np.random.shuffle(data_array)
    num_samps = data_array.shape[0]
    dataset = data_array[:, :num_cats]
    truelabs = data_array[:, num_cats:].astype('uint8')
    true_labels = np.zeros((num_samps, num_cats + 1), dtype='uint8')
    row_index = np.array(range(num_samps)).reshape(1, -1).T
    true_labels[(row_index, truelabs)] = 1
    true_labels = true_labels[:, 1:]
    # Separate 70% for test and 30% for validation:
    num_test_samps = int(0.7 * num_samps)
    num_val_samps = num_samps - num_test_samps
    test_set = dataset[:num_test_samps, :]
    test_set_labels = true_labels[:num_test_samps, :]
    val_set = dataset[num_test_samps:, :]
    val_set_labels = true_labels[num_test_samps:, :]
    max_fscore = 0
    t_optim = 0
    for t in xrange(1, num_cats+1):
        pred_val_labels = rcut_thresholding(val_set, t)
        fscore_macro = compute_fscore_macro(val_set_labels, pred_val_labels)
        if fscore_macro > max_fscore:
            max_fscore = fscore_macro
            t_optim = t
        print t, fscore_macro
    # Optimum value for T:
    print "Optimum T: %d" % t_optim
    # Perform prediction on test set:
    pred_testset_labels = rcut_thresholding(test_set, t_optim)
    fscore = compute_fscore_macro(test_set_labels, pred_testset_labels)
    print "Test F-Score: %f" % fscore