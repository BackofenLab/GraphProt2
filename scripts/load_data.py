#!/usr/bin/env python3

from lib import gp2lib

# Input data folder to load data from.
data_folder = "test_data"

graphs, seqs_1h, labels = gp2lib.load_data(data_folder,
                                           use_up=True,
                                           use_con=True,
                                           use_sf=True,
                                           use_entr=True,
                                           use_str_elem_up=True,
                                           bpp_cutoff=0.2)

