#!/usr/bin/env python3

from lib import gp2lib

# Input data folder to load data from.
data_folder = "HNRNPK_K562_eCLIP_test_extlr30_extcon150_thr2.2_m0_out"

# Use_sf and use_entr are parameters corresponding to site features, be careful when using use_entr since they are biased
# use_up: contains 1 feature (position based feature)
# use_con: 2 features (position based features)
# use_str_elem_up: 4 features (position based features)
# bpp: 1 feature, only for forming graphs
graphs, seqs_1h, sfv_list, labels = gp2lib.load_data(data_folder,
                                           use_up=True,
                                           use_con=True,
                                           use_sf=True,
                                           use_entr=False,
                                           onehot2d=True,
                                           mean_norm=True,
                                           add_1h_to_g=True,
                                           use_str_elem_up=True,
                                           bpp_cutoff=0.2)

