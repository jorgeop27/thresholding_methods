#!/usr/bin/env python

import os
import argparse
# import numpy as np
from performance_measures import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


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
        # Calculate prior probabilities per category:
        self.prior_probabilities = self.prior_probabilites_per_cat()

    def clear_prediction(self):
        self.testset_predictions = np.zeros(self.testset_labels.shape, dtype='uint8')

    def prior_probabilites_per_cat(self):
        priorprobs = self.trainset_labels.sum(axis=0).astype('float') / self.trainset_length
        return priorprobs

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

    def rcut(self, threshold=None, pmeasure='acc'):
        if threshold is not None:
            self.testset_predictions = self.__rcut_threshold__(threshold, self.testset)
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
                max_pval = 0
                opt_thresh = 0
                max_num_labels = int(np.max(train_set_labels.sum(axis=1)))
                print 'Max. num. of Labels: %d' % max_num_labels
                for t in xrange(1, max_num_labels+1):
                    pred_train_labels = self.__rcut_threshold__(t, train_set)
#                    hloss = calculate_hamming_loss(train_set_labels, pred_train_labels)
                    if pmeasure == 'acc':
                        pval = calculate_accuracy(train_set_labels, pred_train_labels)
                    elif pmeasure == 'macro':
                        pval = calculate_macro_f1_score(train_set_labels, pred_train_labels)
                    elif pmeasure == 'micro':
                        pval = calculate_micro_f1_score(train_set_labels, pred_train_labels)
                    else:
                        print "Error in performance measure"
                        return -1
                    # print "F1-score (Macro): %f" % f1s
                    if pval > max_pval:
                        max_pval = pval
                        opt_thresh = t
                print "Opt. Threshold for fold %d: %d" % (kind + 1, opt_thresh)
                pred_test_labels = self.__rcut_threshold__(opt_thresh, test_set)
                if pmeasure == 'acc':
                    opt_pval = calculate_accuracy(test_set_labels, pred_test_labels)
                elif pmeasure == 'macro':
                    opt_pval = calculate_macro_f1_score(test_set_labels, pred_test_labels)
                elif pmeasure == 'micro':
                    opt_pval = calculate_micro_f1_score(test_set_labels, pred_test_labels)
                else:
                    print "Error in performance measure"
                    return -1
                opt_thresholds[kind] = opt_thresh
                opt_perf_values[kind] = opt_pval
            print "Best F1 Score (Macro): %s" % opt_perf_values
            opt_thresh = opt_thresholds[np.argmax(opt_perf_values)]
            self.testset_predictions = self.__rcut_threshold__(opt_thresh, self.testset)
            return opt_thresh

    def __one_threshold__(self, threshold, dataset=None):
        if dataset is None:
            # Calculate using the class score set
            pred_labels = (self.scores >= threshold).astype('uint8')
        else:
            pred_labels = (dataset >= threshold).astype('uint8')
        return pred_labels

    def one_threshold(self, threshold=None, pmeasure='acc'):
        if threshold is not None:
            self.testset_predictions = self.__one_threshold__(threshold, self.testset)
            return threshold
        else:
            # Cross validation for calculating the best possible threshold
            # Split the scores and labels in k (5 or 10) sets
            k = 5
            print "Cross Validation with k = %d" % k
            # optimum thresholds and performance values:
            opt_thresholds = np.zeros(k, dtype='float')
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
                max_perf_val = 0
                opt_thresh = 0
                num_possible_thresholds = 500
                possible_thresholds = np.linspace(1 / num_possible_thresholds, 1, num_possible_thresholds)
                for t in possible_thresholds:
                    pred_train_labels = self.__one_threshold__(t, train_set)
                    if pmeasure == 'acc':
                        pmeas = calculate_accuracy(train_set_labels, pred_train_labels)
                    elif pmeasure == 'macro':
                        pmeas = calculate_macro_f1_score(train_set_labels, pred_train_labels)
                    elif pmeasure == 'micro':
                        pmeas = calculate_micro_f1_score(train_set_labels, pred_train_labels)
                    if pmeas > max_perf_val:
                        max_perf_val = pmeas
                        opt_thresh = t
                print "Opt. Threshold for fold %d: %f" % (kind + 1, opt_thresh)
                pred_test_labels = self.__one_threshold__(opt_thresh, test_set)
                if pmeasure == 'acc':
                    opt_pval = calculate_accuracy(test_set_labels, pred_test_labels)
                elif pmeasure == 'macro':
                    opt_pval = calculate_macro_f1_score(test_set_labels, pred_test_labels)
                elif pmeasure == 'micro':
                    opt_pval = calculate_micro_f1_score(test_set_labels, pred_test_labels)
                opt_thresholds[kind] = opt_thresh
                opt_perf_values[kind] = opt_pval
            print "Best F1 Score (Macro): %s" % opt_perf_values
            opt_thresh = opt_thresholds[np.argmax(opt_perf_values)]
            self.testset_predictions = self.__one_threshold__(opt_thresh, self.testset)
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
            testset_length = len(test_ind)
            train_ind = np.setdiff1d(indices, test_ind)
            trainset_length = len(train_ind)
            train_set = self.trainset[train_ind, :]
            train_set_labels = self.trainset_labels[train_ind, :]
            test_set = self.trainset[test_ind, :]
            test_set_labels = self.trainset_labels[test_ind, :]
            num_poss_thresholds = 200
            possible_thresholds = np.linspace(1.0 / num_poss_thresholds, 1, num_poss_thresholds)
            for cat in xrange(self.num_labels):
                scores_cat = train_set[:, cat]
                labels_cat = train_set_labels[:, cat]
                perfmeas_values = np.zeros(num_poss_thresholds, 'float')
                ind = 0
                for thr in possible_thresholds:
                    pred_lbls = (scores_cat >= thr).astype('int')
                    perfmeas_values[ind] = (np.power(pred_lbls - labels_cat, 2).sum()).astype('float') / trainset_length
                    ind += 1
                thresh_cat = possible_thresholds[np.argmin(perfmeas_values)]
                opt_thresholds[kind, cat] = thresh_cat
                testset_cat = test_set[:, cat]
                testset_labels_cat = test_set_labels[:, cat]
                pred_test_lbls = (testset_cat >= thresh_cat).astype('int')
                test_mse = (np.power(pred_test_lbls - testset_labels_cat, 2).sum()).astype('float') / testset_length
                opt_perf_values[kind, cat] = test_mse
        thresh_row_indices = np.argmin(opt_perf_values, axis=0)
        thresh_col_indices = np.array(range(self.num_labels))
        optim_thresholds = opt_thresholds[(thresh_row_indices, thresh_col_indices)]
        optim_thresholds = np.tile(optim_thresholds, (self.testset_length, 1))
        self.testset_predictions = (self.testset >= optim_thresholds).astype('uint8')

    def __pcut_threshold__(self, omega, dataset=None):
        if dataset is None:
            # Calculate using the class score set
            k = int(np.floor(omega * self.prior_probabilities * self.length_dataset))
            pred_labels = np.zeros((self.length_dataset, self.num_labels), dtype='uint8')
            sorted_scores = self.scores.argsort(axis=0)
        else:
            k = np.floor(omega * self.prior_probabilities * dataset.shape[0]).astype('uint8')
            pred_labels = np.zeros(dataset.shape, dtype='uint8')
            sorted_scores = dataset.argsort(axis=0)
        for cat in xrange(self.num_labels):
            cat_scores = sorted_scores[:, cat]
            cat_scores = np.flipud(cat_scores)
            indices = cat_scores[:k[cat]]
            pred_labels[indices, cat] = 1
        return pred_labels

    def pcut(self, pmeasure='macro'):
        # Identify the best threshold per class using cross validation
        k = 5
        print "Cross Validation with k = %d" % k
        opt_omega = np.zeros(k, dtype='float')
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
            step = 0.01
            perf_values = np.zeros(1 / step, dtype='float')
            ind = 0
            possible_omega = np.linspace(step, 1, 1 / step)
            for w in possible_omega:
                pred_lbls = self.__pcut_threshold__(w, train_set)
                if pmeasure == 'macro':
                    perf_values[ind] = calculate_macro_f1_score(train_set_labels, pred_lbls)
                elif pmeasure == 'micro':
                    perf_values[ind] = calculate_micro_f1_score(train_set_labels, pred_lbls)
                elif pmeasure == 'acc':
                    perf_values[ind] = calculate_accuracy(train_set_labels, pred_lbls)
                ind += 1
            max_pmeasure_ind = np.argmax(perf_values)
            omega = possible_omega[max_pmeasure_ind]
            test_pred_labels = self.__pcut_threshold__(omega, test_set)
            if pmeasure == 'macro':
                opt_perf_values[kind] = calculate_macro_f1_score(test_set_labels, test_pred_labels)
            elif pmeasure == 'micro':
                opt_perf_values[kind] = calculate_micro_f1_score(test_set_labels, test_pred_labels)
            elif pmeasure == 'acc':
                opt_perf_values[kind] = calculate_accuracy(test_set_labels, test_pred_labels)
            opt_omega[kind] = omega
        best_perf_value_ind = np.argmax(opt_perf_values)
        best_omega = opt_omega[best_perf_value_ind]
        self.testset_predictions = self.__pcut_threshold__(best_omega, self.testset)

    def mcut(self):
        # Apply MCut thresholding over the test dataset
        sorted_dataset = np.sort(self.testset)
        diff_scores = np.diff(sorted_dataset, axis=1)
        # Get the index of the maximum differences:
        maxdiff_1 = np.argmax(diff_scores, axis=1).reshape(1, -1).T
        maxdiff_2 = maxdiff_1 + 1
        row_index = np.array(range(self.testset_length)).reshape(1, -1).T
        mcut = (sorted_dataset[(row_index, maxdiff_1)] + sorted_dataset[(row_index, maxdiff_2)]) / 2
        mcut = np.reshape(mcut, mcut.shape[0])
        self.testset_predictions = np.greater(self.testset, mcut[:, None]).astype('uint8')

    def metalabeler(self):
        metadataset = self.trainset
        metalabels = self.trainset_labels.sum(axis=1)
        clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(metadataset, metalabels)
        k_pred = clf.predict(self.testset)
        print k_pred
        print self.testset_labels.sum(axis=1)
        sorted_dataset = np.fliplr(np.argsort(self.testset))
        self.clear_prediction()
        for ind in xrange(self.testset_length):
            k = k_pred[ind]
            lbl_inds = sorted_dataset[ind,:k]
            self.testset_predictions[ind, lbl_inds] = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates different thresholdings over the datas in the directory.')
    parser.add_argument('directory', nargs=1, help='Path with the Set directory.')
    parser.add_argument('file_basename', nargs=1, help='Common filename to all the files in the set.')
    args = parser.parse_args()
    set_dir = args.directory[0]
    basename = args.file_basename[0]
    extensions = [['biological_labels', 'biological_scores'],
                  ['molecular_labels', 'molecular_scores'],
                  ['cellular_labels', 'cellular_scores']]
    set_output = os.path.join(set_dir, 'output_file3.csv')
    with open(set_output, 'w') as outf:
        for ix in range(len(extensions)):
            # files:
            ext = extensions[ix]
            ontology_cat = ext[0].split('_')[0].upper()
            labels_file = os.path.join(set_dir, '%s.%s' % (basename, ext[0]))
            scores_file = os.path.join(set_dir, '%s.%s' % (basename, ext[1]))
            print labels_file
            print scores_file
            # Read data from files into a numpy arrays:
            scores_array = np.loadtxt(scores_file, dtype='float64', delimiter=',')
            labels_array = np.loadtxt(labels_file, dtype='uint8', delimiter=',')
            th = Thresholding(scores_array, labels_array)
            print "One-Threshold:"
            line_arr = [ontology_cat, 'OT']
            th.one_threshold(pmeasure='acc')
            print "Performance Measures on Testing Dataset - Sample-based"
            pm1 = calculate_accuracy(th.testset_labels, th.testset_predictions)
            pm2 = calculate_hamming_loss(th.testset_labels, th.testset_predictions)
            pm3 = calculate_precision(th.testset_labels, th.testset_predictions)
            pm4 = calculate_recall(th.testset_labels, th.testset_predictions)
            pm5 = calculate_f1_score(th.testset_labels, th.testset_predictions)
            line_arr.extend([str(pm1), str(pm2), str(pm3), str(pm4), str(pm5)])
            outf.write(','.join(line_arr) + '\n')
            th.one_threshold(pmeasure='macro')
            print "Performance Measures on Testing Dataset - Class-based"
            pm6 = calculate_macro_precision(th.testset_labels, th.testset_predictions)
            pm7 = calculate_macro_recall(th.testset_labels, th.testset_predictions)
            pm8 = calculate_macro_f1_score(th.testset_labels, th.testset_predictions)
            line_arr = [ontology_cat, 'OT', str(pm6), str(pm7), str(pm8)]
            outf.write(','.join(line_arr) + '\n')
            th.one_threshold(pmeasure='micro')
            print "Performance Measures on Testing Dataset - Overall"
            pm9 = calculate_micro_precision(th.testset_labels, th.testset_predictions)
            pm10 = calculate_micro_recall(th.testset_labels, th.testset_predictions)
            pm11 = calculate_micro_f1_score_from_precision_recall(pm9, pm10)
            line_arr = [ontology_cat, 'OT', str(pm9), str(pm10), str(pm11)]
            outf.write(','.join(line_arr) + '\n')

            print '\nRCut thresholding:'
            line_arr = [ontology_cat, 'RCut']
            th.rcut(pmeasure='acc')
            print "Performance Measures on Testing Dataset - Sample-based"
            pm1 = calculate_accuracy(th.testset_labels, th.testset_predictions)
            pm2 = calculate_hamming_loss(th.testset_labels, th.testset_predictions)
            pm3 = calculate_precision(th.testset_labels, th.testset_predictions)
            pm4 = calculate_recall(th.testset_labels, th.testset_predictions)
            pm5 = calculate_f1_score(th.testset_labels, th.testset_predictions)
            line_arr.extend([str(pm1), str(pm2), str(pm3), str(pm4), str(pm5)])
            outf.write(','.join(line_arr) + '\n')
            th.rcut(pmeasure='macro')
            print "Performance Measures on Testing Dataset - Class-based"
            pm6 = calculate_macro_precision(th.testset_labels, th.testset_predictions)
            pm7 = calculate_macro_recall(th.testset_labels, th.testset_predictions)
            pm8 = calculate_macro_f1_score(th.testset_labels, th.testset_predictions)
            line_arr = [ontology_cat, 'RCut', str(pm6), str(pm7), str(pm8)]
            outf.write(','.join(line_arr) + '\n')
            th.rcut(pmeasure='micro')
            print "Performance Measures on Testing Dataset - Overall"
            pm9 = calculate_micro_precision(th.testset_labels, th.testset_predictions)
            pm10 = calculate_micro_recall(th.testset_labels, th.testset_predictions)
            pm11 = calculate_micro_f1_score_from_precision_recall(pm9, pm10)
            line_arr = [ontology_cat, 'RCut', str(pm9), str(pm10), str(pm11)]
            outf.write(','.join(line_arr) + '\n')

            print '\nPCut thresholding:'
            line_arr = [ontology_cat, 'PCut']
            th.pcut(pmeasure='acc')
            print "Performance Measures on Testing Dataset - Sample-based"
            pm1 = calculate_accuracy(th.testset_labels, th.testset_predictions)
            pm2 = calculate_hamming_loss(th.testset_labels, th.testset_predictions)
            pm3 = calculate_precision(th.testset_labels, th.testset_predictions)
            pm4 = calculate_recall(th.testset_labels, th.testset_predictions)
            pm5 = calculate_f1_score(th.testset_labels, th.testset_predictions)
            line_arr.extend([str(pm1), str(pm2), str(pm3), str(pm4), str(pm5)])
            outf.write(','.join(line_arr) + '\n')
            th.pcut(pmeasure='macro')
            print "Performance Measures on Testing Dataset - Class-based"
            pm6 = calculate_macro_precision(th.testset_labels, th.testset_predictions)
            pm7 = calculate_macro_recall(th.testset_labels, th.testset_predictions)
            pm8 = calculate_macro_f1_score(th.testset_labels, th.testset_predictions)
            line_arr = [ontology_cat, 'PCut', str(pm6), str(pm7), str(pm8)]
            outf.write(','.join(line_arr) + '\n')
            th.pcut(pmeasure='micro')
            print "Performance Measures on Testing Dataset - Overall"
            pm9 = calculate_micro_precision(th.testset_labels, th.testset_predictions)
            pm10 = calculate_micro_recall(th.testset_labels, th.testset_predictions)
            pm11 = calculate_micro_f1_score_from_precision_recall(pm9, pm10)
            line_arr = [ontology_cat, 'PCut', str(pm9), str(pm10), str(pm11)]
            outf.write(','.join(line_arr) + '\n')

            print '\nSCut thresholding:'
            line_arr = [ontology_cat, 'SCut']
            th.scut()
            print "Performance Measures on Testing Dataset - Sample-based"
            pm1 = calculate_accuracy(th.testset_labels, th.testset_predictions)
            pm2 = calculate_hamming_loss(th.testset_labels, th.testset_predictions)
            pm3 = calculate_precision(th.testset_labels, th.testset_predictions)
            pm4 = calculate_recall(th.testset_labels, th.testset_predictions)
            pm5 = calculate_f1_score(th.testset_labels, th.testset_predictions)
            line_arr.extend([str(pm1), str(pm2), str(pm3), str(pm4), str(pm5)])
            outf.write(','.join(line_arr) + '\n')
            print "Performance Measures on Testing Dataset - Class-based"
            pm6 = calculate_macro_precision(th.testset_labels, th.testset_predictions)
            pm7 = calculate_macro_recall(th.testset_labels, th.testset_predictions)
            pm8 = calculate_macro_f1_score(th.testset_labels, th.testset_predictions)
            line_arr = [ontology_cat, 'SCut', str(pm6), str(pm7), str(pm8)]
            outf.write(','.join(line_arr) + '\n')
            print "Performance Measures on Testing Dataset - Overall"
            pm9 = calculate_micro_precision(th.testset_labels, th.testset_predictions)
            pm10 = calculate_micro_recall(th.testset_labels, th.testset_predictions)
            pm11 = calculate_micro_f1_score_from_precision_recall(pm9, pm10)
            line_arr = [ontology_cat, 'SCut', str(pm9), str(pm10), str(pm11)]
            outf.write(','.join(line_arr) + '\n')

            print '\nMCut thresholding:'
            line_arr = [ontology_cat, 'MCut']
            th.mcut()
            print "Performance Measures on Testing Dataset - Sample-based"
            pm1 = calculate_accuracy(th.testset_labels, th.testset_predictions)
            pm2 = calculate_hamming_loss(th.testset_labels, th.testset_predictions)
            pm3 = calculate_precision(th.testset_labels, th.testset_predictions)
            pm4 = calculate_recall(th.testset_labels, th.testset_predictions)
            pm5 = calculate_f1_score(th.testset_labels, th.testset_predictions)
            line_arr.extend([str(pm1), str(pm2), str(pm3), str(pm4), str(pm5)])
            outf.write(','.join(line_arr) + '\n')
            print "Performance Measures on Testing Dataset - Class-based"
            pm6 = calculate_macro_precision(th.testset_labels, th.testset_predictions)
            pm7 = calculate_macro_recall(th.testset_labels, th.testset_predictions)
            pm8 = calculate_macro_f1_score(th.testset_labels, th.testset_predictions)
            line_arr = [ontology_cat, 'MCut', str(pm6), str(pm7), str(pm8)]
            outf.write(','.join(line_arr) + '\n')
            print "Performance Measures on Testing Dataset - Overall"
            pm9 = calculate_micro_precision(th.testset_labels, th.testset_predictions)
            pm10 = calculate_micro_recall(th.testset_labels, th.testset_predictions)
            pm11 = calculate_micro_f1_score_from_precision_recall(pm9, pm10)
            line_arr = [ontology_cat, 'MCut', str(pm9), str(pm10), str(pm11)]
            outf.write(','.join(line_arr) + '\n')

            print "\nMetalabeler:"
            line_arr = [ontology_cat, 'ML']
            th.metalabeler()
            print "Performance Measures on Testing Dataset - Sample-based"
            pm1 = calculate_accuracy(th.testset_labels, th.testset_predictions)
            pm2 = calculate_hamming_loss(th.testset_labels, th.testset_predictions)
            pm3 = calculate_precision(th.testset_labels, th.testset_predictions)
            pm4 = calculate_recall(th.testset_labels, th.testset_predictions)
            pm5 = calculate_f1_score(th.testset_labels, th.testset_predictions)
            line_arr.extend([str(pm1), str(pm2), str(pm3), str(pm4), str(pm5)])
            outf.write(','.join(line_arr) + '\n')
            print "Performance Measures on Testing Dataset - Class-based"
            pm6 = calculate_macro_precision(th.testset_labels, th.testset_predictions)
            pm7 = calculate_macro_recall(th.testset_labels, th.testset_predictions)
            pm8 = calculate_macro_f1_score(th.testset_labels, th.testset_predictions)
            line_arr = [ontology_cat, 'ML', str(pm6), str(pm7), str(pm8)]
            outf.write(','.join(line_arr) + '\n')
            print "Performance Measures on Testing Dataset - Overall"
            pm9 = calculate_micro_precision(th.testset_labels, th.testset_predictions)
            pm10 = calculate_micro_recall(th.testset_labels, th.testset_predictions)
            pm11 = calculate_micro_f1_score_from_precision_recall(pm9, pm10)
            line_arr = [ontology_cat, 'ML', str(pm9), str(pm10), str(pm11)]
            outf.write(','.join(line_arr) + '\n')
