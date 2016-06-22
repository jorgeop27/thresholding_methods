#!/usr/bin/env python

import sys
import csv
import numpy as np
import pandas as pd

def rcut_thresholding(filename):
    data = pd.read_csv(filename, sep=',', header=None)
    


if __name__=='__main__':
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s filename\n"%sys.argv[0])
        raise SystemExit(1)
    filename = sys.argv[1]
    rcut_thresholding(filename)