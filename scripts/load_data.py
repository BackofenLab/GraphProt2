#!/usr/bin/env python3

from lib import gp2lib

# Input data folder to load data from.
data_folder = "test_data"

graphs, seqs_1h, labels = gp2lib.load_data(data_folder,
                                           use_up=True, 
                                           bpp_cutoff=0.2)
