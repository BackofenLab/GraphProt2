#!/usr/bin/env python3

from lib import gp2lib
import re

# Input data folder to load data from.
data_folder = "/home/uhlm/scratch_0_uhlm/Data/cluster/gp2_out_merged_results"
# /home/uhlm/scratch_0_uhlm/Data/cluster/gp2_out_merged_results
# test10_generic_set_extlr30_extcon150_thr2_m0_out

graphs, seqs_1h, sfv_list, labels = gp2lib.load_data(data_folder,
                                           gm_data=True,
                                           use_str_elem_up=False,
                                           use_str_elem_1h=False,
                                           use_us_ds_labels=False,
                                           use_region_labels=False,
                                           use_up=False,
                                           center_vp=False,
                                           vp_ext=0,
                                           con_ext=20,
                                           use_con=False,
                                           use_sf=False,
                                           use_entr=False,
                                           add_1h_to_g=False,
                                           all_nt_uc=False,
                                           fix_vp_len=False,
                                           bpp_mode=2,
                                           bpp_cutoff=0.5)


# Get RBP label to numeric label ID (stored in labels list) mapping.
label_dic = {}
for i,g in enumerate(graphs):
    m = re.search("(.+?)_", g.graph["id"])
    label_dic[m.group(1)] = labels[i]
print("RBP label to label ID mapping:")
for label_id, label_idx in sorted(label_dic.items()):
    print("%s -> %i" %(label_id, label_idx))

"""
graphs: list of graphs
seqs_1h : list of one-hot matrices plus additional position-wise features
sfv_list : list of site feature vectors
labels : list of site labels (0 : negatives, 1-n : protein labels for n proteins)

IMPORTANT:
For generic model data, use gm_data=True, i.e. function will assign labels to 
each site based on first part of site ID (before first underscore). So 
e.g. id1_001 would be interpreted as belonging to class "id1" and so on.

Features + Options are described in load_data.py / load_ml_data.py


"""
