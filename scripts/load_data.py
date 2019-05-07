#!/usr/bin/env python3

from lib import gp2lib

# Input data folder to load data from.
data_folder = "test_data"

# Use_sf and use_entr are parameters corresponding to site features, be careful when using use_entr since they are biased
# use_up: contains 1 feature
# use_con: 2 features
# use_str_elem_up: 4 features
# bpp: 1 feature
graphs, seqs_1h, sfv_list, labels = gp2lib.load_data(data_folder,
                                           use_up=True,
                                           use_con=True,
                                           use_sf=True,
                                           use_entr=False,
                                           use_str_elem_up=True,
                                           bpp_cutoff=0.2)

