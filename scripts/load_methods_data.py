#!/usr/bin/env python3

from lib import gp2lib

# Input data folder to load data from.
data_folder = "HNRNPK_K562_eCLIP_test_extlr30_extcon150_thr2.2_m0_out"


ids, labels, sequences, features = gp2lib.load_dlprb_data(data_folder
                                                          use_vp_ext=False,
                                                          vp_ext=100,
                                                          fix_vp_len=True)


"""
gp2lib.load_dlprb_data()
========================

ids : list of sequence IDs
labels : list of class labels, with 1's (positives) and 0's (negatives)
sequences : list of (extended) viewpoint sequences
features : list of feature matrices, where each sequence has a feature 
           matrix with dimension nr_of_features * length_of_sequence
           For dlprb, nr_of_features = 5

Function arguments:
use_vp_ext : Instead of extracting viewpoint sequence and get features for 
             this sequence part only, extend viewpoint by vp_ext and get 
             features for extended viewpoint sequence
             This is the default when constructing the graphs, but not 
             for the one-hot encoding CNN.
vp_ext : Define upstream + downstream viewpoint extension
         Usually set equal to used plfold_L (default: 100)
fix_vp_len : Use only viewpoint regions with same length (= max length)
             i.e. shortened viewpoint regions will be discarded

"""
