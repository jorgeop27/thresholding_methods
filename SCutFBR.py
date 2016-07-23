#!/usr/bin/env python

import argparse
import numpy as np


# scut_thresholding calculates the thresholds for each category
def scut_thresholding(scores, labels, min_threshold = 0.1, scut_type = 0):
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
    if scut_type == 0:
        thresholds[thresholds < min_threshold] = np.Inf
    else:
        max_score_per_cat = np.max(scores, axis=0)
        thresholds[thresholds < min_threshold] = max_score_per_cat[thresholds < min_threshold]
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
    parser = argparse.ArgumentParser(description='Calculates SCutFBR thresholding over the data in the filename.')
    parser.add_argument('filename', nargs=1, type=file, help='Filename of the data with the scores.')
    parser.add_argument('num_categories', nargs=1, type=int, help='Number of categories in the previous data set')
    parser.add_argument('min_threshold', nargs=1, type=float, help='Minimum threshold value to all categories')
    parser.add_argument('scut_type', nargs=1, type=int, choices=(0, 1),
                        help='Minimum threshold value to all categories')
    args = parser.parse_args()
    f = args.filename[0]
    num_cats = args.num_categories[0]
    min_threshold = args.min_threshold[0]
    scut_type = args.scut_type[0]
    # Read data from file f into a new numpy array:
    data_array = np.loadtxt(f, dtype=float, delimiter=',')
    # Shuffle rows in data_array:
    np.random.seed(2706)
    np.random.shuffle(data_array)
    num_samps = data_array.shape[0]
    dataset = data_array[:, :num_cats]
    truelabs = data_array[:, num_cats:].astype('uint8')
    true_labels = np.zeros((num_samps, num_cats+1), dtype='uint8')
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
    thresholds = scut_thresholding(val_set, val_set_labels, min_threshold, scut_type)
    print "Thresholds: %s" % thresholds
    # Perform prediction on test set:
    pred_testset_labels = apply_thresholds(test_set, thresholds)
    fscore_micro = compute_fscore_micro(test_set_labels, pred_testset_labels)
    print "Test Micro F-Score: %f" % fscore_micro
    fscore_macro = compute_fscore_macro(test_set_labels, pred_testset_labels)
    print "Test Macro F-Score: %f" % fscore_macro
