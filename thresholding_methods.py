#!/usr/bin/env python

import os
import argparse
# import numpy as np
from performance_measures import *


class Thresholding:
    def __init__(self, score_matrix, label_matrix, random_seed=0):
        assert score_matrix.shape == label_matrix.shape
        self.scores = score_matrix
        self.labels = label_matrix
        self.length_dataset, self.num_labels = self.scores.shape
        np.random.seed(random_seed)
        # Split into training set and testing set:
        # Randomise the indices:
        indices = range(self.length_dataset)
        rand_indices = np.random.choice(indices, self.length_dataset, replace=False)
        self.trainset_length = int(0.7 * self.length_dataset)
        self.testset_length = self.length_dataset - self.trainset_length
        # Trainset:
        self.trainset = self.scores[rand_indices[:self.trainset_length], :]
        self.trainset_labels = self.labels[rand_indices[:self.trainset_length], :]
        # Testset:
        self.testset = self.scores[rand_indices[self.trainset_length:], :]
        self.testset_labels = self.labels[rand_indices[self.trainset_length:], :]
        # Test predictions:
        self.testset_predictions = np.zeros(self.testset_labels.shape, dtype='uint8')

    def clear_prediction(self):
        self.testset_predictions = np.zeros(self.testset_labels.shape, dtype='uint8')

    def __rcut_threshold__(self, threshold, dataset=None):
        if dataset is None:
            # Calculate using the class score set
            pred_labels = np.zeros((self.length_dataset, self.num_labels), dtype='uint8')
            # Get the indices of the sorted array of data:
            sort_index = np.argsort(self.scores, axis=1)
            # Reverse the order of the columns to get High to Low order:
            sort_index = np.fliplr(sort_index)
            labels = sort_index[:, :threshold]
            row_index = np.array(range(self.length_dataset)).reshape(1, -1).T
            pred_labels[(row_index, labels)] = 1
        else:
            length_dtset, num_labs = dataset.shape
            pred_labels = np.zeros((length_dtset, num_labs), dtype='uint8')
            # Get the indices of the sorted array of data:
            sort_index = np.argsort(dataset, axis=1)
            # Reverse the order of the columns to get High to Low order:
            sort_index = np.fliplr(sort_index)
            labels = sort_index[:, :threshold]
            row_index = np.array(range(length_dtset)).reshape(1, -1).T
            pred_labels[(row_index, labels)] = 1
        return pred_labels

    def rcut(self, threshold=None):
        if threshold is not None:
            self.testset_predictions = self.__rcut_threshold__(threshold)
            return threshold
        else:
            # Cross validation for calculating the best possible threshold
            # Split the scores and labels in k (5 or 10) sets
            k = 5
            print "Cross Validation with k = %d" % k
            # optimum thresholds and performance values:
            opt_thresholds = np.zeros(k, dtype='uint8')
            opt_perf_values = np.zeros(k, dtype='float')
            # Randomise the indices from the trainset:
            indices = range(self.trainset_length)
            rand_indices = np.random.choice(indices, self.trainset_length, replace=False)
            # Divide in k set of indices:
            sets = np.array_split(rand_indices, k)
            for kind in xrange(k):
                print "Fold k = %d" % (kind + 1)
                test_ind = sets[kind]
                train_ind = np.setdiff1d(indices, test_ind)
                train_set = self.trainset[train_ind, :]
                train_set_labels = self.trainset_labels[train_ind, :]
                test_set = self.trainset[test_ind, :]
                test_set_labels = self.trainset_labels[test_ind, :]
                max_f1 = 0
                opt_thresh = 0
                max_num_labels = int(np.max(train_set_labels.sum(axis=1)))
                print 'Max. num. of Labels: %d' % max_num_labels
                for t in xrange(1, max_num_labels+1):
                    pred_train_labels = self.__rcut_threshold__(t, train_set)
#                    hloss = calculate_hamming_loss(train_set_labels, pred_train_labels)
                    f1s = calculate_macro_f1_score(train_set_labels, pred_train_labels)
                    # print "F1-score (Macro): %f" % f1s
                    if f1s > max_f1:
                        max_f1 = f1s
                        opt_thresh = t
                print "Opt. Threshold for fold %d: %d" % (kind + 1, opt_thresh)
                pred_test_labels = self.__rcut_threshold__(opt_thresh, test_set)
                opt_f1s = calculate_macro_f1_score(test_set_labels, pred_test_labels)
                opt_thresholds[kind] = opt_thresh
                opt_perf_values[kind] = opt_f1s
            print "Best F1 Score (Macro): %s" % opt_perf_values
            opt_thresh = opt_thresholds[np.argmax(opt_perf_values)]
            self.testset_predictions = self.__rcut_threshold__(opt_thresh, self.testset)
            return opt_thresh

    def scut(self):
        # Identify the best threshold per class using cross validation
        k = 5
        print "Cross Validation with k = %d" % k
        opt_thresholds = np.zeros((k, self.num_labels), dtype='float')
        opt_perf_values = np.zeros((k, self.num_labels), dtype='float')
        # Randomise the indices from the trainset:
        indices = range(self.trainset_length)
        rand_indices = np.random.choice(indices, self.trainset_length, replace=False)
        # Divide in k set of indices:
        sets = np.array_split(rand_indices, k)
        for kind in xrange(k):
            print "Fold k = %d" % (kind + 1)
            test_ind = sets[kind]
            train_ind = np.setdiff1d(indices, test_ind)
            train_set = self.trainset[train_ind, :]
            train_set_labels = self.trainset_labels[train_ind, :]
            test_set = self.trainset[test_ind, :]
            test_set_labels = self.trainset_labels[test_ind, :]
            step = 200
            f1_matrix = np.zeros((step, self.num_labels), dtype='float')
            ind = 0
            possible_thresholds = np.linspace(1.0 / step, 1, step)
            for thr in possible_thresholds:
                pred_lbls = (train_set >= thr)
                f1_matrix[ind, :] = calculate_macro_f1_score(train_set_labels, pred_lbls, per_class=True)
                ind += 1
            thresh_indices = np.argmax(f1_matrix, axis=0)
            thresholds_per_class = possible_thresholds[thresh_indices]
            thresholds_per_class = np.tile(thresholds_per_class, (test_set.shape[0], 1))
            test_preds = (test_set >= thresholds_per_class).astype('uint8')
            opt_perf_values[kind, :] = calculate_macro_f1_score(test_set_labels, test_preds, per_class=True)
            opt_thresholds[kind, :] = thresholds_per_class[0, :]
        thresh_row_indices = np.argmax(opt_perf_values, axis=0)
        thresh_col_indices = np.array(range(self.num_labels))
        optim_thresholds = opt_thresholds[(thresh_row_indices, thresh_col_indices)]
        optim_thresholds = np.tile(optim_thresholds, (self.testset_length, 1))
        self.testset_predictions = (self.testset >= optim_thresholds).astype('uint8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates Rank Cut thresholding over the data in the filename.')
    parser.add_argument('scores', nargs=1, help='Filename of the data with the scores.')
    parser.add_argument('labels', nargs=1, help='Filename of the data with the labels.')
    args = parser.parse_args()
    scores_file = args.scores[0]
    labels_file = args.labels[0]
    # Read data from files into a numpy arrays:
    scores_array = np.loadtxt(scores_file, dtype='float64', delimiter=',')
    labels_array = np.loadtxt(labels_file, dtype='uint8', delimiter=',')
    th = Thresholding(scores_array, labels_array)
    print 'RCut thresholding:'
    optim_th = th.rcut()
    print "Optimum threshold: %d" % optim_th
    print "Performance Measures on Testing Dataset:"
    test_f1_score = calculate_macro_f1_score(th.testset_labels, th.testset_predictions)
    print "Macro F1-Score: %f" % test_f1_score
    pred_file = '%s_predicted_rcut%s' % os.path.splitext(labels_file)
    np.savetxt(pred_file, th.testset_predictions, fmt='%u', delimiter=',')
    print 'SCut thresholding:'
    th.scut()
    print "Performance Measures on Testing Dataset:"
    test_f1_score = calculate_macro_f1_score(th.testset_labels, th.testset_predictions)
    print "Macro F1-Score: %f" % test_f1_score
    pred_file = '%s_predicted_scut%s' % os.path.splitext(labels_file)
    np.savetxt(pred_file, th.testset_predictions, fmt='%u', delimiter=',')
