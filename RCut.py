#!/usr/bin/env python

import sys
import csv
import numpy as np

def rcut_thresholding(filename, threshold):
    data = np.genfromtxt(filename, delimiter=',')
    # Remove labels from data array:
    # threshold should be equal to the number of labels
    labels = data[:,-threshold:]
    data = data[:,:(data.shape[1]-threshold)]
    # Get the indices of the sorted array of data:
    sort_index = np.argsort(data,1)
    thresh_range = xrange(-1,-1-threshold,-1)
    predicted_labels = sort_index[:,thresh_range]
    return predicted_labels

if __name__=='__main__':
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: %s filename threshold\n"%sys.argv[0])
        raise SystemExit(1)
    filename = sys.argv[1]
    th = sys.argv[2]
    if th.isdigit():
        th = int(th)
        if th>0:
            pred_labels = rcut_thresholding(filename, int(th))
        else:
            sys.stderr.write("Usage: %s filename threshold\n"%sys.argv[0])
            sys.stderr.write("Threshold should be an integer larger than 0.\n")
            raise SystemExit(1)
    else:
        sys.stderr.write("Usage: %s filename threshold\n"%sys.argv[0])
        sys.stderr.write("Threshold should be an integer larger than 0.\n")
        raise SystemExit(1)
