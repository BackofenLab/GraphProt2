#!/usr/bin/env python3

from lib import gp2lib

# Input data folder to load data from.
data_folder = "test10_generic_set_extlr30_extcon150_thr2_m0_out"

graphs, seqs_1h, sfv_list, labels = gp2lib.load_data(data_folder,
                                           use_up=True,
                                           use_con=True,
                                           use_sf=True,
                                           use_entr=False,
                                           gm_data=True,
                                           onehot2d=False,
                                           mean_norm=True,
                                           add_1h_to_g=True,
                                           use_str_elem_up=True,
                                           bpp_cutoff=0.2)


# Get RBP label to numeric label ID (stored in labels list) mapping.
label_dic = {}
for i,g in enumerate(graphs):
    m = re.search("(.+?)_", g.graph["id"])
    label_dic[m.group(1)] = labels[i]
print ("RBP label to label ID mapping:")
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

Position-wise features:
4 from nucleotide sequences (one-hot)
1 unpaired probability from RNAplfold
2 conservation features (phastCons + phyloP score)
4 unpaired probabilities for single structural elements
    p_external, p_hairpin, p_internal, p_multiloop

graphs[0].node[1]['feat_vector']
graphs[0].graph["id"]
graphs[0].nodes[0]['label']

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
