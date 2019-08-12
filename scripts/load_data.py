#!/usr/bin/env python3

from lib import gp2lib

# Input data folder to load data from.
data_folder = "CAPRIN1_Baltz2012_original_gp1_data_out"

graphs, seqs_1h, sfv_list, labels = gp2lib.load_data(data_folder,
                                           use_str_elem_up=False,
                                           use_str_elem_1h=False,
                                           use_us_ds_labels=False,
                                           use_region_labels=False,
                                           center_vp=False,
                                           vp_ext=False,
                                           use_con=False,
                                           use_sf=False,
                                           use_entr=False,
                                           onehot2d=False,
                                           add_1h_to_g=False,
                                           all_nt_uc=False,
                                           con_ext=20,
                                           fix_vp_len=False,
                                           bpp_cutoff=0.5)


# Min-max normalize (norm_mode=0) all feature vector values (apart from 1-hot features).
gp2lib.normalize_graph_feat_vectors(graphs, norm_mode=0)

"""

NOTE:
If you have variable center (=viewpoint or vp) region data, 
leave center_vp=False, vp_ext=False and fix_vp_len=False
If you want to have same viewpoint lengths also in case of variable vp length 
input data, use center_vp=True and e.g. vp_ext=30, meaning every center region 
will have length of 61 nt.


graphs: list of graphs
seqs_1h : list of one-hot matrices plus additional position-wise features
sfv_list : list of site feature vectors
labels : list of site labels (1 : positives, 0 : negatives)

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
use_region_labels use exon intron position-wise labels, 
                  encode one-hot (= 2 channels) and add to CNN and graphs.
con_ext           Define upstream + downstream viewpoint extension for graphs
                  The extensions on both sides are called the context regions 
                  and are usually in lowercase nucleotide characters
                  Usually set equal to used plfold_L (default: 100)
vp_ext            Define upstream + downstream viewpoint extension for graphs
                  This extends the viewpoint, ie extending the center region 
                  with uppercase nucleotides (as opposed to con_ext)
                  Best combined with center_vp=True, to make variable length
                  viewpoint regions the same length. E.g. setting center_vp=True 
                  and vp_ext=30 results in viewpoint regions of 61 nt.
all_nt_uc         Convert all graph node characters into uppercase, regardless 
                  of viewpoint or context region.

Position-wise features (in order):
4 from nucleotide sequences (one-hot)  (you have to set add_1h_to_g=True to get these)
5 unpaired probabilities for single structural elements
    p_external, p_hairpin, p_internal, p_multiloop, p_paired
2 conservation features (phastCons + phyloP score)
2 from exon intron region labels (one-hot)

graphs[0].node[0]['feat_vector']
graphs[0].graph["id"]
graphs[0].nodes[0]['label']
list(graphs[].edges())


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
