#!/usr/bin/env python

import sys
import csv
import numpy as np

def generate_random_sample(filename, num_rows, num_cols, num_labels):
    # Creates a ramdon file of num_rows samples (num_cols possible labels) separated by comas (csv)
    # and for each sample, it assigns a number of labels between 1 and num_labels in the last columns:
    with open(filename,'wb') as f:
        csvf = csv.writer(f)
        # Generates random data num_rows x num_cols and normalises the rows to sum 1:
        data = np.random.rand(num_rows, num_cols)
        row_sums = data.sum(1)[:,None]
        data = data/row_sums
        # Generates a new matrix with the indices of the sorted data by column:
        sort_index = np.argsort(data,1)
        # Take the latest num_labels as the possible labels indices
        # Then we will shuffle these indices and select randomly
        # between 1 and num_labels labels:
        possible_labels_indices = sort_index[:,-num_labels:]
        np.random.shuffle(possible_labels_indices.T)
        output_data = np.zeros([num_rows, (num_cols+num_labels)])
        for rw_index in xrange(possible_labels_indices.shape[0]):
            data_line = data[rw_index,:]
            ln = possible_labels_indices[rw_index,:]
            # rand_num_labels is the number of labels to create (between 1 and num_labels)
            rand_num_labels = np.random.randint(1,num_labels+1)
            possible_labels = ln[:rand_num_labels] + 1
            ln = np.concatenate((data_line, possible_labels), axis=1)
            ln = np.pad(ln, [0, (num_cols+num_labels-len(ln))], 'constant', constant_values=0)
            output_data[rw_index,:] = ln
        csvf.writerows(output_data)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.stderr.write("Usage: %s filename number_of_samples number_of_columns, max_number_of_labels\n"%sys.argv[0])
        raise SystemExit(1)
    ## Main Function:
    filename = sys.argv[1]
    num_rows = sys.argv[2]
    num_cols = sys.argv[3]
    max_num_labs = sys.argv[4]
    if num_rows.isdigit() and num_cols.isdigit() and max_num_labs.isdigit():
        generate_random_sample(filename, int(num_rows), int(num_cols), int(max_num_labs))
    else:
        sys.stderr.write("Usage: %s filename number_of_samples number_of_columns number_of_labels\n"%sys.argv[0])
        sys.stderr.write("number_of_samples, number_of_columns and number_of_labels must be valid numbers!\n")
        raise SystemExit(1)
