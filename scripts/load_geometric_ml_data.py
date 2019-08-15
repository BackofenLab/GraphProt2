#!/usr/bin/env python3

from lib import gp2lib

# Input data folder to load data from.
data_folder = "K562_eCLIP_rep1_hg38_rbp_names_test5_centerpos_extlr30_thr3.5_strctovlp0.6_maxnoovlp4000_extlr30_conext150_out"

anl, agi, ae, ana, s1h, sfvl, labels, labels_1h = gp2lib.load_geom_ml_data(data_folder,
                                                   use_str_elem_up=True,
                                                   use_str_elem_1h=False,
                                                   use_us_ds_labels=False,
                                                   use_region_labels=False,
                                                   use_con=True,
                                                   use_sf=True,
                                                   use_entr=False,
                                                   use_up=False,
                                                   all_nt_uc=False,
                                                   center_vp=False,
                                                   vp_ext=0,
                                                   add_1h_to_g=False,
                                                   bpp_mode=2,
                                                   con_ext=50,
                                                   bpp_cutoff=0.5)

"""

Returned geometric lists:
anl :  all_node_labels
agi :  all_graph_indicators
ae  :  all_edges
ana :  all_nodes_attributes
Others:
s1h       : one-hot CNN matrices (plus position-wise features) list
sfvl      : site feature vectors list
labels    : list of label vectors (= multi label labels)
labels_1h : list of label one-hot vectors, 
            i.e. no overlap information (= simple multi class labels)

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
use_region_labels use exon intron position-wise labels, 
                  encode one-hot (= 2 channels) and add to CNN and graphs.
use_con           Add position-wise conservation scores to one-hot matrix and 
                  graph node feature vectors
use_sf            Use site features (one feature vector per binding site)
use_entr          Use entropy features (see below), added to site feature vector
                  Advised not to use for binary classification with random 
                  negatives (naturally biased towards RBP binding sites)
onehot2d          Keep one-matrix 2D
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
vp_ext            Define upstream + downstream viewpoint extension for graphs
                  Usually set equal to used plfold_L (default: 100)


Position-wise features:
4 from nucleotide sequences (one-hot)
1 unpaired probability from RNAplfold
2 conservation features (phastCons + phyloP score)
5 unpaired probabilities for single structural elements
    p_external, p_hairpin, p_internal, p_multiloop, p_paired

Currently used site features:

These are calculated based on the viewpoint region sequence.

    -a            A content
    -c            C content
    -g            G content
    -t            T content
    -at           AT content
    -ags          AG skew content
    -cts          CT skew content
    -gcs          GC skew content
    -gts          GT skew content
    -ks           keto skew content
    -ps           purine skew content
    -cpg          CpG content
    -ce           Shannon entropy
    -cz           compression factor (using gzip)
    -lct          Calculate linguistic complexity (Trifonov)
                  This makes use of -word-size,
                  word-sizes 1,2,3 are used
    -mfe          MFE of extended viewpoint region 
                  (currently 121 nt for vp_len=61 nt)
    -fmfe         Frequency of the MFE
    -ed           Ensemble diversity of extended viewpoint region
    -zsc          Thermodynamic z-score of extended viewpoint region
    -pv           p-value of -zsc


Entropy /RBP occupancy features (part of site feature vector):

These are calculated based all combined eCLIP peaks from clipper, 
HepG2, K562, rep1+rep2, in total 446 sets of bed files.

    avg_rbpc : average dataset overlap count for viewpoint region
               i.e. how many binding sites (for each dataset only 
               counted once) overlap with this region, divided by 
               total number of datasets
    avg_cov:   Average crosslink count or fold change associated with 
               this region. If several binding sites overlap with 
               region in one dataset, average fold change of sites 
               or sum of crosslink counts is taken. Then sum up 
               these counts/changes for all overlapping datasets, 
               and average by total number of datasets
    entr:      Entropy of the viewpoint region, calculated based on 
               coverage values for each dataset. 

"""

