#!/usr/bin/env python3

from lib import gp2lib

# Input data folder to load data from.
data_folder = "EWSR1_K562_rep1_test_extlr30_extcon150_thr3_m0_out"


ids, labels, sequences, features = gp2lib.load_dlprb_data(data_folder,
                                                          use_vp_ext=True,
                                                          vp_ext=10,
                                                          fix_vp_len=True)


"""

gp2lib.load_dlprb_data()
========================

ids : list of sequence IDs
labels : list of class labels, with 1's (positives) and 0's (negatives)
sequences : list of (extended) viewpoint sequences, converted to uppercase
features : list of feature matrices, where each sequence has a feature 
           matrix with dimension nr_of_features * length_of_sequence
           For dlprb, nr_of_features = 5
           Order of features for i-th sequence:
           [i][0] : S (paired prob)
           [i][1] : H (hairpin prob)
           [i][2] : I (internal loop prob)
           [i][3] : M (multi loop prob)
           [i][4] : E (external loop prob)

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