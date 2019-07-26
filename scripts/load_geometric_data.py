#!/usr/bin/env python3

from lib import gp2lib

# Input data folder to load data from.
data_folder = "EWSR1_K562_rep1_GRCh38_peaks_extlr30_extcon150_thr3.4_m0_out"

anl, agi, ae, ana, labels, sfvl = gp2lib.load_geometric_data(data_folder,
                                         use_str_elem_up=True,
                                         use_str_elem_1h=False,
                                         use_us_ds_labels=False,
                                         use_region_labels=True,
                                         use_con=True,
                                         use_sf=True,
                                         use_entr=False,
                                         add_1h_to_g=True,
                                         bpp_mode=1,
                                         vp_ext=100,
                                         bpp_cutoff=0.5)


"""

Returned geometric lists:
anl :  all_node_labels
agi :  all_graph_indicators
ae  :  all_edges
ana :  all_nodes_attributes
Others:
labels : graph class labels
sfvl   : site feature vectors list

    all_nodes_labels       Nucleotide indices (dict_label_idx)
    all_graph_indicators   Graph indices, each node of a graph 
                           gets same index
    all_edges              Indices of edges
    all_nodes_attributes   Node vectors

Options:

use_up            Add position-wise total unpaired probabilities to one-hot 
                  matrix and graph node feature vectors
                  NOTE that this feature will not be added if use_str_elem_up=True
use_str_elem_up   Add position-wise probabilities for structural elements 
                  (E, H, I, M, S)
use_con           Add position-wise conservation scores to one-hot matrix and 
                  graph node feature vectors
use_sf            Use site features (one feature vector per binding site)
use_entr          Use entropy features (see below), added to site feature vector
                  Advised not to use for binary classification with random 
                  negatives (naturally biased towards RBP binding sites)
bpp_cutoff        Base pair probability threshold for adding base pairs to graph 
                  base pairs with prob. >= bpp_cutoff will be added
add_1h_to_g       Add sequence one-hot encoding to graph node feature vectors
use_str_elem_1h   use one-hot encoding of structural elements (E,H,I,M,S) 
                  instead of probabilities (use_str_elem_up)
                  To use probabilities: use_str_elem_1h=False, use_str_elem_up=True
                  To use one-hot encodings: use_str_elem_1h=True, use_str_elem_up=True
use_us_ds_labels  add upstream downstream labeling for context regions in graph (node labels)
                  e.g. upstream "a" becomes "ua", downstream "g" becomes "dg", 
                  while viewpoint region nucleotide labels stay same
use_region_labels use exon intron position-wise labels, 
                  encode one-hot (= 2 channels) and add to CNN and graphs.
vp_ext            Define upstream + downstream viewpoint extension for graphs
                  Usually set equal to used plfold_L (default: 100)
bpp_mode          Defines what base pair edges to add to graph.
                  1 : all bps in extended vp region vp_s-vp_lr_ext - vp_e+vp_lr_ext
                  2 : bps in extended vp region with start or end in non-extended vp region
                  3 : only bps with start+end in base vp

"""

