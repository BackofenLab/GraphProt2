#!/usr/bin/env python3

from lib import gp2lib

# Input data folder to load data from.
data_folder = "K562_eCLIP_rep1_hg38_rbp_names_test5_centerpos_extlr30_thr4_strctovlp0.6_maxnoovlp4000_extlr30_conext150_out"

graphs, seqs_1h, sfv_list, labels = gp2lib.load_ml_data(data_folder,
                                           use_up=True,
                                           use_con=True,
                                           use_sf=True,
                                           use_entr=False,
                                           use_str_elem_up=True,
                                           bpp_cutoff=0.2)

