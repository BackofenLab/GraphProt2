#!/usr/bin/env python3

import sys
import re
import networkx as nx
import numpy as np
import os

"""
GraphProt2 Python3 function library

Run doctests from base directory:
python3 -m doctest -v lib/gp2lib.py


2DO:
check whether methods also work with sequences without 
lowercase context.

"""

def load_geometric_data(data_folder,
                        use_up=False,
                        use_con=False,
                        use_entr=False,
                        use_str_elem_up=False,
                        use_sf=False,
                        use_str_elem_1h=False,
                        use_us_ds_labels=False,
                        use_region_labels=False,
                        disable_bpp=False,
                        bpp_cutoff=0.2,
                        bpp_mode=1,
                        gm_data=False,
                        vp_ext = 100,
                        sf_norm=True,
                        add_1h_to_g=False,
                        fix_vp_len=True):

    """
    Load function for PyTorch geometric data, instead of loading  networkx 
    graphs.
    
    Load data from data_folder.

    Returns the following lists:

    all_nodes_labels       Nucleotide indices (dict_label_idx)
    all_graph_indicators   Graph indices, each node of a graph 
                           gets same index
    all_edges              Indices of edges
    all_nodes_attributes   Node vectors
    graph_labels           Class labels (e.g. 0,1 for binary classification)
                           Length of list = #positives+#negatives
    sfv_list               Site feature vectors list

    Function parameters:
        use_up : if true add unpaired probabilities to graph + one-hot
        use_con : if true add conservation scores to graph + one-hot
        use_str_elem_up: add str elements unpaired probs to graph + one-hot
        use_sf: add site features, store in additional vector for each sequence
        use_entr: use RBP occupancy / entropy features for each sequence
        use_str_elem_1h: use structural elements chars as 1h (in 1h + graph)
                         instead of probabilities
        use_us_ds_labels: add upstream downstream labeling for context 
                          regions in graph (node labels)
        use_region_labels: use exon intron position-wise labels, 
                           encode one-hot (= 2 channels) and add to 
                           CNN and graphs.
        sf_norm:      Normalize site features
        disable_bpp : disables adding of base pair information
        bpp_cutoff : bp probability threshold when adding bp probs.
        bpp_mode : see ext_mode in convert_seqs_to_graphs for details
        vp_ext : Define upstream + downstream viewpoint extension for graphs
                 Usually set equal to used plfold_L (default: 100)
        gm_data : If data is in format for generic model generation
        add_1h_to_g : add one-hot encodings to graph node vectors
        fix_vp_len : Use only viewpoint regions with same length (= max length)

    """

    # Input files.
    pos_fasta_file = "%s/positives.fa" % (data_folder)
    neg_fasta_file = "%s/negatives.fa" % (data_folder)
    pos_up_file = "%s/positives.up" % (data_folder)
    neg_up_file = "%s/negatives.up" % (data_folder)
    pos_bpp_file = "%s/positives.bpp" % (data_folder)
    neg_bpp_file = "%s/negatives.bpp" % (data_folder)
    pos_con_file = "%s/positives.con" % (data_folder)
    neg_con_file = "%s/negatives.con" % (data_folder)
    pos_str_elem_up_file = "%s/positives.str_elem.up" % (data_folder)
    neg_str_elem_up_file = "%s/negatives.str_elem.up" % (data_folder)
    pos_sf_file = "%s/positives.sf" % (data_folder)
    neg_sf_file = "%s/negatives.sf" % (data_folder)
    pos_entr_file = "%s/positives.entr" % (data_folder)
    neg_entr_file = "%s/negatives.entr" % (data_folder)
    pos_region_labels_file = "%s/positives.exon_intron_labels" % (data_folder)
    neg_region_labels_file = "%s/negatives.exon_intron_labels" % (data_folder)

    # Check inputs.
    if not os.path.isdir(data_folder):
        print("INPUT_ERROR: Input data folder \"%s\" not found" % (data_folder))
        sys.exit()
    if not os.path.isfile(pos_fasta_file):
        print("INPUT_ERROR: missing \"%s\"" % (pos_fasta_file))
        sys.exit()
    if not os.path.isfile(neg_fasta_file):
        print("INPUT_ERROR: missing \"%s\"" % (neg_fasta_file))
        sys.exit()
    if use_up:
        if not os.path.isfile(pos_up_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_up_file))
            sys.exit()
        if not os.path.isfile(neg_up_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_up_file))
            sys.exit()
    if use_str_elem_up:
        if not os.path.isfile(pos_str_elem_up_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_str_elem_up_file))
            sys.exit()
        if not os.path.isfile(neg_str_elem_up_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_str_elem_up_file))
            sys.exit()
    if use_con:
        if not os.path.isfile(pos_con_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_con_file))
            sys.exit()
        if not os.path.isfile(neg_con_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_con_file))
            sys.exit()
    if use_entr:
        if not os.path.isfile(pos_entr_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_entr_file))
            sys.exit()
        if not os.path.isfile(neg_entr_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_entr_file))
            sys.exit()
    if use_sf:
        if not os.path.isfile(pos_sf_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_sf_file))
            sys.exit()
        if not os.path.isfile(neg_sf_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_sf_file))
            sys.exit()
    if not disable_bpp:
        if not os.path.isfile(pos_bpp_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_bpp_file))
            sys.exit()
        if not os.path.isfile(neg_bpp_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_bpp_file))
            sys.exit()
    if use_region_labels:
        if not os.path.isfile(pos_region_labels_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_region_labels_file))
            sys.exit()
        if not os.path.isfile(neg_region_labels_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_region_labels_file))
            sys.exit()

    # Read in FASTA sequences.
    pos_seqs_dic = read_fasta_into_dic(pos_fasta_file)
    neg_seqs_dic = read_fasta_into_dic(neg_fasta_file)
    # Get viewpoint regions.
    pos_vp_s, pos_vp_e = extract_viewpoint_regions_from_fasta(pos_seqs_dic)
    neg_vp_s, neg_vp_e = extract_viewpoint_regions_from_fasta(neg_seqs_dic)
    # Extract most prominent (max) viewpoint length from data.
    max_vp_l = 0
    if fix_vp_len:
        for seq_id in pos_seqs_dic:
            vp_l = pos_vp_e[seq_id] - pos_vp_s[seq_id] + 1  # +1 since 1-based.
            if vp_l > max_vp_l:
                max_vp_l = vp_l
        if not max_vp_l:
            print("ERROR: viewpoint length extraction failed")
            sys.exit()

    # Remove sequences that do not pass fix_vp_len.
    if fix_vp_len:
        filter_seq_dic_fixed_vp_len(pos_seqs_dic, pos_vp_s, pos_vp_e, max_vp_l)
        filter_seq_dic_fixed_vp_len(neg_seqs_dic, neg_vp_s, neg_vp_e, max_vp_l)

    # Init dictionaries.
    pos_up_dic = False
    neg_up_dic = False
    pos_bpp_dic = False
    neg_bpp_dic = False
    # con_dic: seq_id to 2xn matrix (storing both phastcons+phylop)
    pos_con_dic = False 
    neg_con_dic = False
    pos_str_elem_up_dic = False
    neg_str_elem_up_dic = False
    # Site features dictionary also includes RNP occupancy / entropy features.
    # Each site gets a vector of feature values.
    pos_sf_dic = False
    neg_sf_dic = False
    # Region labels (exon intron).
    pos_region_labels_dic = False
    neg_region_labels_dic = False

    print("Read in dataset dictionaries ... ")

    # Extract additional annotations.
    if use_up:
        pos_up_dic = read_up_into_dic(pos_up_file)
        neg_up_dic = read_up_into_dic(neg_up_file)
    if not disable_bpp:
        pos_bpp_dic = read_bpp_into_dic(pos_bpp_file, pos_vp_s, pos_vp_e, 
                                        vp_lr_ext=vp_ext)
        neg_bpp_dic = read_bpp_into_dic(neg_bpp_file, neg_vp_s, neg_vp_e, 
                                        vp_lr_ext=vp_ext)
    if use_con:
        pos_con_dic = read_con_into_dic(pos_con_file)
        neg_con_dic = read_con_into_dic(neg_con_file)
    if use_str_elem_up:
        pos_str_elem_up_dic = read_str_elem_up_into_dic(pos_str_elem_up_file)
        neg_str_elem_up_dic = read_str_elem_up_into_dic(neg_str_elem_up_file)
    if use_sf:
        pos_sf_dic = read_sf_into_dic(pos_sf_file, sf_dic=pos_sf_dic)
        neg_sf_dic = read_sf_into_dic(neg_sf_file, sf_dic=neg_sf_dic)
    if use_entr:
        pos_sf_dic = read_entr_into_dic(pos_entr_file, entr_dic=pos_sf_dic)
        neg_sf_dic = read_entr_into_dic(neg_entr_file, entr_dic=neg_sf_dic)
    if use_region_labels:
        pos_region_labels_dic = read_region_labels_into_dic(pos_region_labels_file)
        neg_region_labels_dic = read_region_labels_into_dic(neg_region_labels_file)

    # Normalize site features.
    if sf_norm:
        if use_sf or use_entr:
            pos_sf_dic, neg_sf_dic = normalize_pos_neg_sf_dic(pos_sf_dic, neg_sf_dic)

    print("Generate PyTorch Geometric lists  ... ")

    # Convert input sequences to sequence or structure graphs.
    pos_anl, pos_agi, pos_ae, pos_ana, g_idx, n_idx = generate_geometric_data(pos_seqs_dic, 
                                                                pos_vp_s,
                                                                pos_vp_e,
                                                                up_dic=pos_up_dic, 
                                                                con_dic=pos_con_dic, 
                                                                region_labels_dic=pos_region_labels_dic,
                                                                use_str_elem_1h=use_str_elem_1h,
                                                                str_elem_up_dic=pos_str_elem_up_dic, 
                                                                use_us_ds_labels=use_us_ds_labels,
                                                                bpp_dic=pos_bpp_dic, 
                                                                vp_lr_ext=vp_ext, 
                                                                ext_mode=bpp_mode,
                                                                add_1h_to_g=add_1h_to_g,
                                                                plfold_bpp_cutoff=bpp_cutoff)
    neg_anl, neg_agi, neg_ae, neg_ana, g_idx, n_idx = generate_geometric_data(neg_seqs_dic, 
                                                                neg_vp_s,
                                                                neg_vp_e,
                                                                up_dic=neg_up_dic, 
                                                                con_dic=neg_con_dic, 
                                                                region_labels_dic=neg_region_labels_dic,
                                                                use_str_elem_1h=use_str_elem_1h,
                                                                str_elem_up_dic=neg_str_elem_up_dic, 
                                                                use_us_ds_labels=use_us_ds_labels,
                                                                bpp_dic=neg_bpp_dic, 
                                                                vp_lr_ext=vp_ext, 
                                                                ext_mode=bpp_mode,
                                                                add_1h_to_g=add_1h_to_g,
                                                                g_idx=g_idx,
                                                                n_idx=n_idx,
                                                                plfold_bpp_cutoff=bpp_cutoff)


    # Create labels.
    labels = [1]*len(pos_seqs_dic) + [0]*len(neg_seqs_dic)
    # If data is generic model data, use n labels for n proteins, 
    # + "0" label for negatives.
    if gm_data:
        # Seen labels dictionary.
        label_dic = {}
        # Site ID to label dictionary.
        id2l_dic = {}
        # Label index.
        li = 0
        for seq_id, seq in sorted(pos_seqs_dic.items()):
            # Get RBP ID from seq_id.
                m = re.search("(.+?)_", seq_id)
                if m:
                    label = m.group(1)
                    if not label in label_dic:
                        li += 1
                    label_dic[label] = li
                    id2l_dic[seq_id] = li
                else:
                    print ("ERROR: viewpoint extraction failed for \"%s\"" % (seq_id))
                    sys.exit()
        # Construct positives label vector.
        labels = []
        for seq_id, seq in sorted(seqs_dic.items()):
            label = id2l_dic[seq_id]
            labels.append(label)
        # Add negatives to label vector.
        labels = labels + [0]*len(neg_seqs_dic)
        
    # Concatenate geometric lists.
    anl = pos_anl + neg_anl
    agi = pos_agi + neg_agi
    ae = pos_ae + neg_ae
    ana = pos_ana + neg_ana

    # From site feature dictionaries to list of site feature vectors.
    site_feat_v = []
    if pos_sf_dic:
        for site_id, site_v in sorted(pos_sf_dic.items()):
            if site_id in pos_seqs_dic:
                site_feat_v.append(site_v)
    else:
        for l in [0]*len(pos_seq_1h):
            site_feat_v.append([0])
    if neg_sf_dic:
        for site_id, site_v in sorted(neg_sf_dic.items()):
            if site_id in neg_seqs_dic:
                site_feat_v.append(site_v)
    else:
        for l in [0]*len(neg_seq_1h):
            site_feat_v.append([0])

    # Return geometric lists, label list, and site feature vectors list.
    return anl, agi, ae, ana, labels, site_feat_v


################################################################################

def load_sf_data(data_folder,
                 sf_norm=True):
    """
    Load site feature vectors, store as list of vectors.
    Return list of vectors and labels list.

    """
    # Input files.
    pos_sf_file = "%s/positives.sf" % (data_folder)
    neg_sf_file = "%s/negatives.sf" % (data_folder)

    # Check inputs.
    if not os.path.isfile(pos_sf_file):
        print("INPUT_ERROR: missing \"%s\"" % (pos_sf_file))
        sys.exit()
    if not os.path.isfile(neg_sf_file):
        print("INPUT_ERROR: missing \"%s\"" % (neg_sf_file))
        sys.exit()

    # Site features dictionaries.
    pos_sf_dic = False
    neg_sf_dic = False

    print("Read in dataset dictionaries ... ")

    pos_sf_dic = read_sf_into_dic(pos_sf_file, sf_dic=pos_sf_dic)
    neg_sf_dic = read_sf_into_dic(neg_sf_file, sf_dic=neg_sf_dic)

    # Normalize site features.
    if sf_norm:
        pos_sf_dic, neg_sf_dic = normalize_pos_neg_sf_dic(pos_sf_dic, neg_sf_dic,
                                                          norm_mode=0)

    # Create labels.
    labels = [1]*len(pos_sf_dic) + [0]*len(neg_sf_dic)

    # From site feature dictionaries to list of site feature vectors.
    sf_list = []
    for site_id, site_v in sorted(pos_sf_dic.items()):
        sf_list.append(site_v)
    for site_id, site_v in sorted(neg_sf_dic.items()):
        sf_list.append(site_v)

    # Return list of site feature vectors and label vector.
    return sf_list, labels


################################################################################

def load_ideeps_data(data_folder,
                vp_ext=20,
                use_vp_ext=True,
                fix_vp_len=True):
    """
    Prepare data for iDeepS method.
    Use 101 nt long sequences just like in provided data (fixed length!).
    Return following lists: ids, labels, sequences
    ids : list of sequence ids
          IMPORTANT: id needs to be in format: "id; class:0" for negatives 
          and "id; class:1" for positives in order for iDeepS to recognize
          the class labels.
    vp_ext : Define upstream + downstream viewpoint extension for graphs
             Usually set equal to used plfold_L (default: 20)
             vp_ext=20 + 61 nt of viewpoint = 101 nt sequences
    add_1h_to_g : add one-hot encodings to graph node vectors
    fix_vp_len : Use only viewpoint regions with same length (= max length)

    """

    # Input files.
    pos_fasta_file = "%s/positives.fa" % (data_folder)
    neg_fasta_file = "%s/negatives.fa" % (data_folder)

    # Check inputs.
    if not os.path.isdir(data_folder):
        print("INPUT_ERROR: Input data folder \"%s\" not found" % (data_folder))
        sys.exit()
    if not os.path.isfile(pos_fasta_file):
        print("INPUT_ERROR: missing \"%s\"" % (pos_fasta_file))
        sys.exit()
    if not os.path.isfile(neg_fasta_file):
        print("INPUT_ERROR: missing \"%s\"" % (neg_fasta_file))
        sys.exit()

    # Read in FASTA sequences.
    pos_seqs_dic = read_fasta_into_dic(pos_fasta_file)
    neg_seqs_dic = read_fasta_into_dic(neg_fasta_file)
    # Get viewpoint regions.
    pos_vp_s, pos_vp_e = extract_viewpoint_regions_from_fasta(pos_seqs_dic)
    neg_vp_s, neg_vp_e = extract_viewpoint_regions_from_fasta(neg_seqs_dic)
    # Extract most prominent (max) viewpoint length from data.
    max_vp_l = 0
    if fix_vp_len:
        for seq_id in pos_seqs_dic:
            vp_l = pos_vp_e[seq_id] - pos_vp_s[seq_id] + 1  # +1 since 1-based.
            if vp_l > max_vp_l:
                max_vp_l = vp_l
        if not max_vp_l:
            print("ERROR: viewpoint length extraction failed")
            sys.exit()

    # Remove sequences that do not pass fix_vp_len.
    if fix_vp_len:
        filter_seq_dic_fixed_vp_len(pos_seqs_dic, pos_vp_s, pos_vp_e, max_vp_l)
        filter_seq_dic_fixed_vp_len(neg_seqs_dic, neg_vp_s, neg_vp_e, max_vp_l)

    ids_list = []
    label_list = []
    sequence_list = []

    # Expected sequence length.
    exp_seq_length = max_vp_l
    if use_vp_ext:
        exp_seq_length = max_vp_l + vp_ext*2

    # Process positives.
    for seq_id, seq in sorted(pos_seqs_dic.items()):
        new_seq, new_s, new_e = extract_vp_seq(pos_seqs_dic, seq_id,
                                               use_vp_ext=use_vp_ext,
                                               vp_ext=vp_ext)
        if len(new_seq) != exp_seq_length:
            print("ERROR: new_seq length != exp_seq_length length (%i != %i) for seq_id %s" % (len(new_seq), exp_seq_length, seq_id))
            sys.exit()
        new_seq_id = seq_id + "; class:1"
        ids_list.append(new_seq_id)
        sequence_list.append(new_seq.upper())

        label_list.append(1) # one label positives.

    # Process negatives.
    for seq_id, seq in sorted(neg_seqs_dic.items()):
        new_seq, new_s, new_e = extract_vp_seq(neg_seqs_dic, seq_id,
                                               use_vp_ext=use_vp_ext,
                                               vp_ext=vp_ext)
        if len(new_seq) != exp_seq_length:
            print("ERROR: new_seq length != exp_seq_length length (%i != %i) for seq_id %s" % (len(new_seq), exp_seq_length, seq_id))
            sys.exit()
        new_seq_id = seq_id + "; class:0"
        ids_list.append(new_seq_id)
        sequence_list.append(new_seq.upper())
        label_list.append(0) # zero label negatives.

    # Check lengths.
    l_seqs = len(sequence_list)
    l_labs = len(label_list)
    if l_seqs != l_labs:
        print("ERROR: sequence_list length != label_list length (%i != %i)" % (l_seqs, l_labs))
        sys.exit()
    # Return lists.
    return ids_list, label_list, sequence_list


################################################################################

def load_dlprb_data(data_folder,
                vp_ext=100,
                use_vp_ext=False,
                fix_vp_len=True):
    """
    Prepare data for DLPRB method.
    Return following lists: ids, labels, sequences, features
    ids : list of sequence ids
    features : 5 structural element probabilities for each sequence 
               position, resulting in 2d array for each sequence
    From DLPRB github readme:
    ""Every row corresponds to a different structural context: 
    paired, hairpin loop, internal loop, multiloop, and external loop"
    So return for each sequence matrix of size 5*n with n=length(sequence)
    sequences : return viewpoint sequences, or set use_vp_ext=True to 
    get features for extended viewpoint sequences (extension set by vp_ext)
    labels : the sequence labels 1 or 0 (1 : positives, 0 : negatives)
    Arguments:
    vp_ext : Define upstream + downstream viewpoint extension for graphs
             Usually set equal to used plfold_L (default: 100)
    add_1h_to_g : add one-hot encodings to graph node vectors
    fix_vp_len : Use only viewpoint regions with same length (= max length)

    """

    # Input files.
    pos_fasta_file = "%s/positives.fa" % (data_folder)
    neg_fasta_file = "%s/negatives.fa" % (data_folder)
    pos_str_elem_up_file = "%s/positives.str_elem.up" % (data_folder)
    neg_str_elem_up_file = "%s/negatives.str_elem.up" % (data_folder)

    # Check inputs.
    if not os.path.isdir(data_folder):
        print("INPUT_ERROR: Input data folder \"%s\" not found" % (data_folder))
        sys.exit()
    if not os.path.isfile(pos_fasta_file):
        print("INPUT_ERROR: missing \"%s\"" % (pos_fasta_file))
        sys.exit()
    if not os.path.isfile(neg_fasta_file):
        print("INPUT_ERROR: missing \"%s\"" % (neg_fasta_file))
        sys.exit()
    if not os.path.isfile(pos_str_elem_up_file):
        print("INPUT_ERROR: missing \"%s\"" % (pos_str_elem_up_file))
        sys.exit()
    if not os.path.isfile(neg_str_elem_up_file):
        print("INPUT_ERROR: missing \"%s\"" % (neg_str_elem_up_file))
        sys.exit()

    # Read in FASTA sequences.
    pos_seqs_dic = read_fasta_into_dic(pos_fasta_file)
    neg_seqs_dic = read_fasta_into_dic(neg_fasta_file)
    # Get viewpoint regions.
    pos_vp_s, pos_vp_e = extract_viewpoint_regions_from_fasta(pos_seqs_dic)
    neg_vp_s, neg_vp_e = extract_viewpoint_regions_from_fasta(neg_seqs_dic)
    # Extract most prominent (max) viewpoint length from data.
    max_vp_l = 0
    if fix_vp_len:
        for seq_id in pos_seqs_dic:
            vp_l = pos_vp_e[seq_id] - pos_vp_s[seq_id] + 1  # +1 since 1-based.
            if vp_l > max_vp_l:
                max_vp_l = vp_l
        if not max_vp_l:
            print("ERROR: viewpoint length extraction failed")
            sys.exit()

    # Remove sequences that do not pass fix_vp_len.
    if fix_vp_len:
        filter_seq_dic_fixed_vp_len(pos_seqs_dic, pos_vp_s, pos_vp_e, max_vp_l)
        filter_seq_dic_fixed_vp_len(neg_seqs_dic, neg_vp_s, neg_vp_e, max_vp_l)

    # Read in structural elements probabilities.
    str_elem_up_dic = read_str_elem_up_into_dic(pos_str_elem_up_file)
    str_elem_up_dic = read_str_elem_up_into_dic(neg_str_elem_up_file, str_elem_up_dic=str_elem_up_dic)

    ids_list = []
    label_list = []
    sequence_list = []
    feat_matrix_list = []

    # Process positives.
    for seq_id, seq in sorted(pos_seqs_dic.items()):
        new_seq, new_s, new_e = extract_vp_seq(pos_seqs_dic, seq_id,
                                               use_vp_ext=use_vp_ext,
                                               vp_ext=vp_ext)
        # Start position to extract probabilities.
        i=new_s-1
        # Init feature matrix.
        feat_matrix = [[],[],[],[],[]] # 5 empty feature rows.
        for nt in new_seq:
            p_e = str_elem_up_dic[seq_id][0][i]
            p_h = str_elem_up_dic[seq_id][1][i]
            p_i = str_elem_up_dic[seq_id][2][i]
            p_m = str_elem_up_dic[seq_id][3][i]
            p_s = str_elem_up_dic[seq_id][4][i]
            # DLPRB order: S H I M E
            feat_matrix[0].append(p_s)
            feat_matrix[1].append(p_h)
            feat_matrix[2].append(p_i)
            feat_matrix[3].append(p_m)
            feat_matrix[4].append(p_e)
            i+=1
        ids_list.append(seq_id)
        feat_matrix_list.append(feat_matrix)
        sequence_list.append(new_seq.upper())
        label_list.append(1) # one label positives.

    # Process negatives.
    for seq_id, seq in sorted(neg_seqs_dic.items()):
        new_seq, new_s, new_e = extract_vp_seq(neg_seqs_dic, seq_id,
                                               use_vp_ext=use_vp_ext,
                                               vp_ext=vp_ext)
        # Start position to extract probabilities.
        i=new_s-1
        # Init feature matrix.
        feat_matrix = [[],[],[],[],[]] # 5 empty feature rows.
        for nt in new_seq:
            p_e = str_elem_up_dic[seq_id][0][i]
            p_h = str_elem_up_dic[seq_id][1][i]
            p_i = str_elem_up_dic[seq_id][2][i]
            p_m = str_elem_up_dic[seq_id][3][i]
            p_s = str_elem_up_dic[seq_id][4][i]
            # DLPRB order: S H I M E
            feat_matrix[0].append(p_s)
            feat_matrix[1].append(p_h)
            feat_matrix[2].append(p_i)
            feat_matrix[3].append(p_m)
            feat_matrix[4].append(p_e)
            i+=1
        ids_list.append(seq_id)
        feat_matrix_list.append(feat_matrix)
        sequence_list.append(new_seq.upper())
        label_list.append(0) # zero label negatives.

    # Check lengths.
    l_feat = len(feat_matrix_list)
    l_seqs = len(sequence_list)
    l_labs = len(label_list)
    if l_feat != l_seqs:
        print("ERROR: feat_matrix_list length != sequence_list length (%i != %i)" % (l_feat, l_seqs))
        sys.exit()
    if l_seqs != l_labs:
        print("ERROR: sequence_list length != label_list length (%i != %i)" % (l_seqs, l_labs))
        sys.exit()
    # Return lists.
    return ids_list, label_list, sequence_list, feat_matrix_list


################################################################################

def extract_vp_seq(seqs_dic, seq_id,
                   use_vp_ext=False,
                   vp_ext=100):
    """
    Extract viewpoint part (uppercase chars) from sequence with 
    given sequence ID seq_id.
    If use_vp_ext is set, viewpoint region will be extended by 
    vp_ext. Thus total length of returned sequence will be 
    len(vp_region)+2*len(vp_ext).
    Return sequence, start position + end position (both one-based)
    of extracted sequence.
    
    >>> seqs_dic = {"CLIP_01" : "acguACGUacgu", "CLIP_02" : "CCCCgggg"}
    >>> seq, s, e = extract_vp_seq(seqs_dic, "CLIP_01")
    >>> print(seq, s, e)
    ACGU 5 8
    >>> seq, s, e = extract_vp_seq(seqs_dic, "CLIP_01", use_vp_ext=True, vp_ext=2)
    >>> print(seq, s, e)
    guACGUac 3 10
    >>> seq, s, e = extract_vp_seq(seqs_dic, "CLIP_02", use_vp_ext=True, vp_ext=2)
    >>> print(seq, s, e)
    CCCCgg 1 6

    """
    # Check.
    if not seq_id in seqs_dic:
        print ("ERROR: seq_id \"%s\" not found in seqs_dic" % (seq_id))
        sys.exit()
    seq = seqs_dic[seq_id]
    m = re.search("([acgun]*)([ACGUN]+)([acgun]*)", seq)
    if m:
        us_seq = m.group(1)
        vp_seq = m.group(2)
        ds_seq = m.group(3)
        l_us = len(us_seq)
        l_vp = len(vp_seq)
        l_ds = len(ds_seq)
        # Viewpoint start + end.
        new_s = l_us+1
        new_e = l_us+l_vp
        new_seq = vp_seq
        if use_vp_ext:
            new_us_seq = us_seq[-vp_ext:]
            new_ds_seq = ds_seq[:vp_ext]
            l_new_us = len(new_us_seq)
            l_new_ds = len(new_ds_seq)
            new_s = l_us-l_new_us+1
            new_e = l_us+l_vp+l_new_ds
            new_seq = new_us_seq+vp_seq+new_ds_seq
        return new_seq, new_s, new_e
    else:
        print ("ERROR: extract_vp_seq() viewpoint extraction failed for \"%s\"" % (seq_id))
        sys.exit()


################################################################################

def load_ml_data(data_folder, 
              use_up=False,
              use_con=False,
              use_entr=False,
              use_str_elem_up=False,
              use_sf=False,
              use_str_elem_1h=False,
              use_us_ds_labels=False,
              disable_bpp=False,
              bpp_cutoff=0.2,
              bpp_mode=1,
              vp_ext=100,
              add_1h_to_g=False,
              onehot2d=False,
              fix_vp_len=True):
    """
    Load multi label data from folder.
    Loop over all files, load data in.
    After looping, construct one-hot-lists and graphs from 
    feature dictionaries.
    NOTE that file names are treated as labels, 
    e.g. file name: DGCR8.bed, thus label: DGCR8

    Function parameters:
        use_up : if true add unpaired probabilities to graph + one-hot
        use_con : if true add conservation scores to graph + one-hot
        use_str_elem_up: add str elements unpaired probs to graph + one-hot
        use_sf: add site features, store in additional vector for each sequence
        use_entr: use RBP occupancy / entropy features for each sequence
        use_str_elem_1h: use structural elements chars as 1h (in 1h + graph)
                         instead of probabilities
        use_us_ds_labels: add upstream downstream labeling for context 
                          regions in graph (node labels)
        disable_bpp : disables adding of base pair information
        bpp_cutoff : bp probability threshold when adding bp probs.
        bpp_mode : see ext_mode in convert_seqs_to_graphs for details
        onehot2d : Do not convert one-hot to 3d
        vp_ext : Define upstream + downstream viewpoint extension for graphs
                 Usually set equal to used plfold_L (default: 100)
        add_1h_to_g : add one-hot encodings to graph node vectors
        fix_vp_len : Use only viewpoint regions with same length (= max length)

    """

    # Check input.
    if not os.path.isdir(data_folder):
        print("INPUT_ERROR: Input data folder \"%s\" not found" % (data_folder))
        sys.exit()
    # Sequences dictionary.
    total_seqs_dic = {}
    # Viewpoint coordinate dictionaries.
    vp_s_dic = {}
    vp_e_dic = {}
    # Feature dictionaries.
    up_dic = False
    bpp_dic = False
    con_dic = False 
    str_elem_up_dic = False
    neg_str_elem_up_dic = False
    sf_dic = False
    max_vp_l = 0
    seq_1h_list = False
    graphs_list = False
    # Label to index dictionary.
    l2i_dic = {}
    # Site ID to protein dictionary.
    id2l_dic = {}
    # Label index.
    li = 0
    
    print("Read in datasets ... ")
    
    # Get dataset IDs.
    cmd = "ls " + data_folder + "/*.fa | sort"
    set_list = os.popen(cmd).readlines()
    # Go over datasets.
    l_sl = len(set_list)
    for l in set_list:
        m = re.search(".+\/(.+?)\.fa", l.strip())
        data_id = m.group(1)
        l2i_dic[data_id] = li
        li += 1
        # Input files for dataset.
        fasta_file = "%s/%s.fa" % (data_folder, data_id)
        up_file = "%s/%s.up" % (data_folder, data_id)
        bpp_file = "%s/%s.bpp" % (data_folder, data_id)
        con_file = "%s/%s.con" % (data_folder, data_id)
        str_elem_up_file = "%s/%s.str_elem.up" % (data_folder, data_id)
        sf_file = "%s/%s.sf" % (data_folder, data_id)
        entr_file = "%s/%s.entr" % (data_folder, data_id)
        # Check if files exist.
        if use_up:
            if not os.path.isfile(up_file):
                print("INPUT_ERROR: missing \"%s\"" % (up_file))
                sys.exit()
        if use_str_elem_up:
            if not os.path.isfile(str_elem_up_file):
                print("INPUT_ERROR: missing \"%s\"" % (str_elem_up_file))
                sys.exit()
        if use_con:
            if not os.path.isfile(con_file):
                print("INPUT_ERROR: missing \"%s\"" % (con_file))
                sys.exit()
        if use_entr:
            if not os.path.isfile(entr_file):
                print("INPUT_ERROR: missing \"%s\"" % (entr_file))
                sys.exit()
        if use_sf:
            if not os.path.isfile(sf_file):
                print("INPUT_ERROR: missing \"%s\"" % (sf_file))
                sys.exit()
        if not disable_bpp:
            if not os.path.isfile(bpp_file):
                print("INPUT_ERROR: missing \"%s\"" % (bpp_file))
                sys.exit()

        print("Read in dataset %s dictionaries ... " % (data_id))

        # Read in FASTA sequences for this round only.
        seqs_dic = read_fasta_into_dic(fasta_file)
        # Assign site IDs to dataset label.
        for seq_id in seqs_dic:
            id2l_dic[seq_id] = data_id
        # Get viewpoint regions.
        vp_s_dic, vp_e_dic = extract_viewpoint_regions_from_fasta(seqs_dic,
                                                                  vp_s_dic=vp_s_dic,
                                                                  vp_e_dic=vp_e_dic)
        # Extract most prominent (max) viewpoint length from data.
        if not max_vp_l: # extract only first dataset.
            if fix_vp_len:
                for seq_id in seqs_dic:
                    vp_l = vp_e_dic[seq_id] - vp_s_dic[seq_id] + 1  # +1 since 1-based.
                    if vp_l > max_vp_l:
                        max_vp_l = vp_l
                if not max_vp_l:
                    print("ERROR: viewpoint length extraction failed")
                    sys.exit()

        # Remove sequences that do not pass fix_vp_len.
        if fix_vp_len:
            filter_seq_dic_fixed_vp_len(seqs_dic, vp_s_dic, vp_e_dic, max_vp_l)

        # Extract additional annotations.
        if use_up:
            up_dic = read_up_into_dic(up_file, up_dic=up_dic)
        if not disable_bpp:
            bpp_dic = read_bpp_into_dic(bpp_file, vp_s_dic, vp_e_dic, 
                                        bpp_dic=bpp_dic,
                                        vp_lr_ext=vp_ext)
        if use_con:
            con_dic = read_con_into_dic(con_file, 
                                        con_dic=con_dic)
        if use_str_elem_up:
            str_elem_up_dic = read_str_elem_up_into_dic(str_elem_up_file,
                                                        str_elem_up_dic=str_elem_up_dic)
        if use_sf:
            sf_dic = read_sf_into_dic(sf_file,
                                      sf_dic=sf_dic)
        if use_entr:
            sf_dic = read_entr_into_dic(entr_file,
                                        entr_dic=sf_dic)
        # Merge to total seqs dic.
        total_seqs_dic = add_dic2_to_dic1(total_seqs_dic, seqs_dic)

    # Error if not data was read in.
    if not total_seqs_dic:
        print("INPUT_ERROR: empty total_seqs_dic")
        sys.exit()

    print("Generate one-hot encodings ... ")

    # Create / extend one-hot encoding list.
    seq_1h_list = convert_seqs_to_one_hot(total_seqs_dic, vp_s_dic, vp_e_dic,
                                          up_dic=up_dic,
                                          con_dic=con_dic,
                                          use_str_elem_1h=use_str_elem_1h,
                                          str_elem_up_dic=str_elem_up_dic)

    print("Generate graphs ... ")

    # Create / extend graphs list.
    graphs_list = convert_seqs_to_graphs(total_seqs_dic, vp_s_dic, vp_e_dic, 
                                         up_dic=up_dic, 
                                         con_dic=con_dic, 
                                         use_str_elem_1h=use_str_elem_1h,
                                         use_us_ds_labels=use_us_ds_labels,
                                         str_elem_up_dic=str_elem_up_dic, 
                                         bpp_dic=bpp_dic, 
                                         vp_lr_ext=vp_ext, 
                                         ext_mode=bpp_mode,
                                         add_1h_to_g=add_1h_to_g,
                                         plfold_bpp_cutoff=bpp_cutoff)

    # Check.
    if li != l_sl:
        print("ERROR: li != l_sl (%i != %i)" % (li, l_sl))
        sys.exit()

    # Label vectors list, in sorted seqs_dic order like other lists.
    label_vect_list = []
    # Label vectors list, where only main binding site gets "1".
    label_1h_vect_list = []
    # Site-level feature vectors list.
    site_feat_vect_list = []
    for seq_id, seq in sorted(total_seqs_dic.items()):
        # Get data_id of seq_id.
        data_id = id2l_dic[seq_id]
        label_1h_list = [0]*li
        data_id_i = l2i_dic[data_id]
        label_1h_list[data_id_i] = 1
        # Generate site label vectors.
        m = re.search(".+;(.+),", seq_id)
        labels = m.group(1).split(",")
        label_list = [0]*li
        # For each label from binding site.
        for l in labels:
            i = l2i_dic[l] # get index of label.
            label_list[i] = 1
        label_vect_list.append(label_list)
        label_1h_vect_list.append(label_1h_list)
        # Generate site-level feature vectors.
        if sf_dic:
            site_feat_vect_list.append(sf_dic[seq_id])
        else:
            site_feat_vect_list.append([0])

    # Convert 1h list to np array, transpose matrices and make each entry 3d (1,number_of_features,vp_length).
    new_seq_1h_list = []
    for i in range(len(seq_1h_list)):
        M = np.array(seq_1h_list[i]).transpose()
        if not onehot2d:
            M = np.reshape(M, (1, M.shape[0], M.shape[1]))
        new_seq_1h_list.append(M)

    # Check for equal lengths of graphs, new_seq_1h and site_feat_v.
    l_g = len(graphs_list)
    l_1h = len(new_seq_1h_list)
    l_sfv = len(site_feat_vect_list)
    l_lbl = len(label_vect_list)
    if l_g != l_1h:
        print("ERROR: graphs list length != one-hot list length (%i != %i)" % (l_g, l_1h))
        sys.exit()
    if l_1h != l_sfv:
        print("ERROR: one-hot list length != site feature vector list length (%i != %i)" % (l_1h, l_sfv))
        sys.exit()
    if l_sfv != l_lbl:
        print("ERROR: site feature vector list length != labels list length (%i != %i)" % (l_sfv, l_lbl))
        sys.exit()
    # Return graphs list, one-hot np.array, and label vector.
    return graphs_list, new_seq_1h_list, site_feat_vect_list, label_vect_list, label_1h_vect_list


################################################################################

def add_dic2_to_dic1(dic1, dic2):
    """
    Add dictionary 1 to dictionary 2.
    """
    for dic2_k, dic2_v in dic2.items():
        dic1[dic2_k] = dic2_v
    return dic1


################################################################################

def load_data(data_folder, 
              use_up=False,
              use_con=False,
              use_entr=False,
              use_str_elem_up=False,
              use_sf=False,
              use_str_elem_1h=False,
              use_us_ds_labels=False,
              use_region_labels=False,
              disable_bpp=False,
              bpp_cutoff=0.2,
              bpp_mode=1,
              gm_data=False,
              vp_ext = 100,
              sf_norm=True,
              add_1h_to_g=False,
              onehot2d=False,
              fix_vp_len=True):
    """
    Load data from data_folder.
    Return list of structure graphs, one-hot encoded sequences np array, 
    and label vector.
    
    Function parameters:
        use_up : if true add unpaired probabilities to graph + one-hot
        use_con : if true add conservation scores to graph + one-hot
        use_str_elem_up: add str elements unpaired probs to graph + one-hot
        use_sf: add site features, store in additional vector for each sequence
        use_entr: use RBP occupancy / entropy features for each sequence
        use_str_elem_1h: use structural elements chars as 1h (in 1h + graph)
                         instead of probabilities
        use_us_ds_labels: add upstream downstream labeling for context 
                          regions in graph (node labels)
        use_region_labels: use exon intron position-wise labels, 
                           encode one-hot (= 2 channels) and add to 
                           CNN and graphs.
        sf_norm:      Normalize site features
        disable_bpp : disables adding of base pair information
        bpp_cutoff : bp probability threshold when adding bp probs.
        bpp_mode : see ext_mode in convert_seqs_to_graphs for details
        vp_ext : Define upstream + downstream viewpoint extension for graphs
                 Usually set equal to used plfold_L (default: 100)
        onehot2d : Do not convert one-hot to 3d
        gm_data : If data is in format for generic model generation
        add_1h_to_g : add one-hot encodings to graph node vectors
        fix_vp_len : Use only viewpoint regions with same length (= max length)

    """
    # Input files.
    pos_fasta_file = "%s/positives.fa" % (data_folder)
    neg_fasta_file = "%s/negatives.fa" % (data_folder)
    pos_up_file = "%s/positives.up" % (data_folder)
    neg_up_file = "%s/negatives.up" % (data_folder)
    pos_bpp_file = "%s/positives.bpp" % (data_folder)
    neg_bpp_file = "%s/negatives.bpp" % (data_folder)
    pos_con_file = "%s/positives.con" % (data_folder)
    neg_con_file = "%s/negatives.con" % (data_folder)
    pos_str_elem_up_file = "%s/positives.str_elem.up" % (data_folder)
    neg_str_elem_up_file = "%s/negatives.str_elem.up" % (data_folder)
    pos_sf_file = "%s/positives.sf" % (data_folder)
    neg_sf_file = "%s/negatives.sf" % (data_folder)
    pos_entr_file = "%s/positives.entr" % (data_folder)
    neg_entr_file = "%s/negatives.entr" % (data_folder)
    pos_region_labels_file = "%s/positives.exon_intron_labels" % (data_folder)
    neg_region_labels_file = "%s/negatives.exon_intron_labels" % (data_folder)

    # Check inputs.
    if not os.path.isdir(data_folder):
        print("INPUT_ERROR: Input data folder \"%s\" not found" % (data_folder))
        sys.exit()
    if not os.path.isfile(pos_fasta_file):
        print("INPUT_ERROR: missing \"%s\"" % (pos_fasta_file))
        sys.exit()
    if not os.path.isfile(neg_fasta_file):
        print("INPUT_ERROR: missing \"%s\"" % (neg_fasta_file))
        sys.exit()
    if use_up:
        if not os.path.isfile(pos_up_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_up_file))
            sys.exit()
        if not os.path.isfile(neg_up_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_up_file))
            sys.exit()
    if use_str_elem_up:
        if not os.path.isfile(pos_str_elem_up_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_str_elem_up_file))
            sys.exit()
        if not os.path.isfile(neg_str_elem_up_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_str_elem_up_file))
            sys.exit()
    if use_con:
        if not os.path.isfile(pos_con_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_con_file))
            sys.exit()
        if not os.path.isfile(neg_con_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_con_file))
            sys.exit()
    if use_entr:
        if not os.path.isfile(pos_entr_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_entr_file))
            sys.exit()
        if not os.path.isfile(neg_entr_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_entr_file))
            sys.exit()
    if use_sf:
        if not os.path.isfile(pos_sf_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_sf_file))
            sys.exit()
        if not os.path.isfile(neg_sf_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_sf_file))
            sys.exit()
    if not disable_bpp:
        if not os.path.isfile(pos_bpp_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_bpp_file))
            sys.exit()
        if not os.path.isfile(neg_bpp_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_bpp_file))
            sys.exit()
    if use_region_labels:
        if not os.path.isfile(pos_region_labels_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_region_labels_file))
            sys.exit()
        if not os.path.isfile(neg_region_labels_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_region_labels_file))
            sys.exit()

    #print("Read in sequences ... ")

    # Read in FASTA sequences.
    pos_seqs_dic = read_fasta_into_dic(pos_fasta_file)
    neg_seqs_dic = read_fasta_into_dic(neg_fasta_file)
    # Get viewpoint regions.
    pos_vp_s, pos_vp_e = extract_viewpoint_regions_from_fasta(pos_seqs_dic)
    neg_vp_s, neg_vp_e = extract_viewpoint_regions_from_fasta(neg_seqs_dic)
    # Extract most prominent (max) viewpoint length from data.
    max_vp_l = 0
    if fix_vp_len:
        for seq_id in pos_seqs_dic:
            vp_l = pos_vp_e[seq_id] - pos_vp_s[seq_id] + 1  # +1 since 1-based.
            if vp_l > max_vp_l:
                max_vp_l = vp_l
        if not max_vp_l:
            print("ERROR: viewpoint length extraction failed")
            sys.exit()

    # Remove sequences that do not pass fix_vp_len.
    if fix_vp_len:
        filter_seq_dic_fixed_vp_len(pos_seqs_dic, pos_vp_s, pos_vp_e, max_vp_l)
        filter_seq_dic_fixed_vp_len(neg_seqs_dic, neg_vp_s, neg_vp_e, max_vp_l)

    # Init dictionaries.
    pos_up_dic = False
    neg_up_dic = False
    pos_bpp_dic = False
    neg_bpp_dic = False
    # con_dic: seq_id to 2xn matrix (storing both phastcons+phylop)
    pos_con_dic = False 
    neg_con_dic = False
    pos_str_elem_up_dic = False
    neg_str_elem_up_dic = False
    # Site features dictionary also includes RNP occupancy / entropy features.
    # Each site gets a vector of feature values.
    pos_sf_dic = False
    neg_sf_dic = False
    # Region labels (exon intron).
    pos_region_labels_dic = False
    neg_region_labels_dic = False

    print("Read in dataset dictionaries ... ")

    # Extract additional annotations.
    if use_up:
        pos_up_dic = read_up_into_dic(pos_up_file)
        neg_up_dic = read_up_into_dic(neg_up_file)
    if not disable_bpp:
        pos_bpp_dic = read_bpp_into_dic(pos_bpp_file, pos_vp_s, pos_vp_e, 
                                        vp_lr_ext=vp_ext)
        neg_bpp_dic = read_bpp_into_dic(neg_bpp_file, neg_vp_s, neg_vp_e, 
                                        vp_lr_ext=vp_ext)
    if use_con:
        pos_con_dic = read_con_into_dic(pos_con_file)
        neg_con_dic = read_con_into_dic(neg_con_file)
    if use_str_elem_up:
        pos_str_elem_up_dic = read_str_elem_up_into_dic(pos_str_elem_up_file)
        neg_str_elem_up_dic = read_str_elem_up_into_dic(neg_str_elem_up_file)
    if use_sf:
        pos_sf_dic = read_sf_into_dic(pos_sf_file, sf_dic=pos_sf_dic)
        neg_sf_dic = read_sf_into_dic(neg_sf_file, sf_dic=neg_sf_dic)
    if use_entr:
        pos_sf_dic = read_entr_into_dic(pos_entr_file, entr_dic=pos_sf_dic)
        neg_sf_dic = read_entr_into_dic(neg_entr_file, entr_dic=neg_sf_dic)
    if use_region_labels:
        pos_region_labels_dic = read_region_labels_into_dic(pos_region_labels_file)
        neg_region_labels_dic = read_region_labels_into_dic(neg_region_labels_file)

    # Normalize site features.
    if sf_norm:
        if use_sf or use_entr:
            pos_sf_dic, neg_sf_dic = normalize_pos_neg_sf_dic(pos_sf_dic, neg_sf_dic)

    print("Generate one-hot encodings ... ")

    # Convert input sequences to one-hot encoding (optionally with unpaired probabilities vector).
    pos_seq_1h = convert_seqs_to_one_hot(pos_seqs_dic, pos_vp_s, pos_vp_e, 
                                         up_dic=pos_up_dic,
                                         con_dic=pos_con_dic,
                                         region_labels_dic=pos_region_labels_dic,
                                         use_str_elem_1h=use_str_elem_1h,
                                         str_elem_up_dic=pos_str_elem_up_dic)
    neg_seq_1h = convert_seqs_to_one_hot(neg_seqs_dic, neg_vp_s, neg_vp_e, 
                                         up_dic=neg_up_dic, 
                                         con_dic=neg_con_dic,
                                         region_labels_dic=neg_region_labels_dic,
                                         use_str_elem_1h=use_str_elem_1h,
                                         str_elem_up_dic=neg_str_elem_up_dic)

    print("Generate graphs ... ")

    # Convert input sequences to sequence or structure graphs.
    pos_graphs = convert_seqs_to_graphs(pos_seqs_dic, pos_vp_s, pos_vp_e, 
                                        up_dic=pos_up_dic, 
                                        con_dic=pos_con_dic, 
                                        region_labels_dic=pos_region_labels_dic,
                                        use_str_elem_1h=use_str_elem_1h,
                                        str_elem_up_dic=pos_str_elem_up_dic, 
                                        use_us_ds_labels=use_us_ds_labels,
                                        bpp_dic=pos_bpp_dic, 
                                        vp_lr_ext=vp_ext, 
                                        ext_mode=bpp_mode,
                                        add_1h_to_g=add_1h_to_g,
                                        plfold_bpp_cutoff=bpp_cutoff)
    neg_graphs = convert_seqs_to_graphs(neg_seqs_dic, neg_vp_s, neg_vp_e, 
                                        up_dic=neg_up_dic, 
                                        con_dic=neg_con_dic,
                                        region_labels_dic=neg_region_labels_dic,
                                        use_str_elem_1h=use_str_elem_1h,
                                        str_elem_up_dic=neg_str_elem_up_dic, 
                                        use_us_ds_labels=use_us_ds_labels,
                                        bpp_dic=neg_bpp_dic, 
                                        vp_lr_ext=vp_ext,  
                                        ext_mode=bpp_mode,
                                        add_1h_to_g=add_1h_to_g,
                                        plfold_bpp_cutoff=bpp_cutoff)

    # Create labels.
    labels = [1]*len(pos_seq_1h) + [0]*len(neg_seq_1h)
    # If data is generic model data, use n labels for n proteins, 
    # + "0" label for negatives.
    if gm_data:
        # Seen labels dictionary.
        label_dic = {}
        # Site ID to label dictionary.
        id2l_dic = {}
        # Label index.
        li = 0
        for seq_id, seq in sorted(pos_seqs_dic.items()):
            # Get RBP ID from seq_id.
                m = re.search("(.+?)_", seq_id)
                if m:
                    label = m.group(1)
                    if not label in label_dic:
                        li += 1
                    label_dic[label] = li
                    id2l_dic[seq_id] = li
                else:
                    print ("ERROR: viewpoint extraction failed for \"%s\"" % (seq_id))
                    sys.exit()
        # Construct positives label vector.
        labels = []
        for g in pos_graphs:
            seq_id = g.graph["id"]
            label = id2l_dic[seq_id]
            labels.append(label)
        # Add negatives to label vector.
        labels = labels + [0]*len(neg_seq_1h)
    # Concatenate pos+neg graph lists.
    graphs = pos_graphs + neg_graphs
    # Concatenate pos+neg one-hot lists.
    seq_1h = pos_seq_1h + neg_seq_1h
    # Convert 1h list to np array, transpose matrices and make each entry 3d (1,number_of_features,vp_length).
    new_seq_1h = []
    for idx in range(len(seq_1h)):
        M = np.array(seq_1h[idx]).transpose()
        if not onehot2d:
            M = np.reshape(M, (1, M.shape[0], M.shape[1]))
        new_seq_1h.append(M)
    # From site feature dictionaries to list of site feature vectors.
    site_feat_v = []
    if pos_sf_dic:
        for site_id, site_v in sorted(pos_sf_dic.items()):
            if site_id in pos_seqs_dic:
                site_feat_v.append(site_v)
    else:
        for l in [0]*len(pos_seq_1h):
            site_feat_v.append([0])
    if neg_sf_dic:
        for site_id, site_v in sorted(neg_sf_dic.items()):
            if site_id in neg_seqs_dic:
                site_feat_v.append(site_v)
    else:
        for l in [0]*len(neg_seq_1h):
            site_feat_v.append([0])
    # Check for equal lengths of graphs, new_seq_1h and site_feat_v.
    l_g = len(graphs)
    l_1h = len(new_seq_1h)
    l_sfv = len(site_feat_v)
    l_lbl = len(labels)
    if l_g != l_1h:
        print("ERROR: graphs list length != one-hot list length (%i != %i)" % (l_g, l_1h))
        sys.exit()
    if l_1h != l_sfv:
        print("ERROR: one-hot list length != site feature vector list length (%i != %i)" % (l_1h, l_sfv))
        sys.exit()
    if l_sfv != l_lbl:
        print("ERROR: site feature vector list length != labels list length (%i != %i)" % (l_sfv, l_lbl))
        sys.exit()
    # Return graphs list, one-hot np.array, and label vector.
    return graphs, new_seq_1h, site_feat_v, labels


################################################################################

def read_fasta_into_dic(fasta_file,
                        seqs_dic=False,
                        skip_n_seqs=True):
    """
    Read in FASTA sequences, store in dictionary and return dictionary.
    
    >>> test_fasta = "test_data/test.fa"
    >>> d = read_fasta_into_dic(test_fasta)
    >>> print(d)
    {'seq1': 'acguACGUacgu', 'seq2': 'ugcaUGCAugcaACGUacgu'}
    >>> test_fasta = "test_data/test2.fa"
    >>> d = read_fasta_into_dic(test_fasta)
    >>> print(d)
    {}

    """
    if not seqs_dic:
        seqs_dic = {}
    seq_id = ""
    seq = ""
    # Go through FASTA file, extract sequences.
    with open(fasta_file) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                if seq_id in seqs_dic:
                    print ("ERROR: non-unique FASTA header \"%s\" in \"%s\"" % (seq_id, fasta_file))
                    sys.exit()
                else:
                    seqs_dic[seq_id] = ""
            elif re.search("[ACGTUN]+", line, re.I):
                m = re.search("([ACGTUN]+)", line, re.I)
                # If sequences with N nucleotides should be skipped.
                if skip_n_seqs:
                    if "n" in m.group(1) or "N" in m.group(1):
                        print ("WARNING: sequence with seq_id \"%s\" in file \"%s\" contains N nucleotides. Discarding sequence ... " % (seq_id, fasta_file))
                        del seqs_dic[seq_id]
                        continue
                if seq_id in seqs_dic:
                    # Convert to RNA, concatenate sequence.
                    seqs_dic[seq_id] += m.group(1).replace("T","U").replace("t","u")
    f.closed
    return seqs_dic


################################################################################

def read_region_labels_into_dic(region_labels_file,
                                region_labels_dic=False):
    """
    Read in position-wise labels for each site into dictionary of lists.
    Key: site ID, value: list of labels
    E.g. from input:
    CLIP_1	EEIIIIEEEE
    CLIP_3	IIIIIIIIII
    CLIP_2	EEEEIIIIEE
    Generate lists:
    ['E', 'E', 'I', 'I', 'I', 'I', 'E', 'E', 'E', 'E']
    ['I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I']
    ['E', 'E', 'E', 'E', 'I', 'I', 'I', 'I', 'E', 'E']

    >>> region_labels_test = "test_data/test.region_labels"
    >>> read_region_labels_into_dic(region_labels_test)
    {'CLIP_1': ['E', 'I', 'I', 'E'], 'CLIP_2': ['I', 'I', 'I', 'I']}

    """
    if not region_labels_dic:
        region_labels_dic = {}
    # Read in file content.
    with open(region_labels_file) as f:
        for line in f:
            cols = line.strip().split("\t")
            site_id = cols[0]
            region_labels_dic[site_id] = list(cols[1])
    return region_labels_dic


################################################################################

def string_vectorizer(seq, 
                      s=False,
                      e=False,
                      empty_vectors=False,
                      custom_alphabet=False):
    """
    Take string sequence, look at each letter and convert to one-hot-encoded
    vector. Optionally define start and end index (1-based) for extracting 
    sub-sequences.
    Return array of one-hot encoded vectors.
    If empty_vectors=True, return list of empty vectors.

    >>> string_vectorizer("ACGU")
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    >>> string_vectorizer("")
    []
    >>> string_vectorizer("XX")
    [[0, 0, 0, 0], [0, 0, 0, 0]]
    >>> string_vectorizer("ABC", empty_vectors=True)
    [[], [], []]

    """
    alphabet=['A','C','G','U']
    if custom_alphabet:
        alphabet = custom_alphabet
    seq_l = len(seq)
    if empty_vectors:
        vector = []
        for letter in seq:
            vector.append([])
    else:
        vector = [[1 if char == letter else 0 for char in alphabet] for letter in seq]
    if s and e:
        if len(seq) < e or s < 1:
            print ("ERROR: invalid indices passed to string_vectorizer (s:\"%s\", e:\"%s\")" % (s, e))
            sys.exit()
        vector = vector[s-1:e]
    return vector


################################################################################

def char_vectorizer(char,
                    custom_alphabet=False):
    """
    Vectorize given nucleotide character. Convert to uppercase before 
    vectorizing.

    >>> char_vectorizer("C")
    [0, 1, 0, 0]
    >>> char_vectorizer("g")
    [0, 0, 1, 0]
    >>> char_vectorizer("M", ['E', 'H', 'I', 'M', 'S'])
    [0, 0, 0, 1, 0]

    """
    alphabet = ['A','C','G','U']
    if custom_alphabet:
        alphabet = custom_alphabet
    char = char.upper()
    l = len(char)
    vector = []
    if not l == 1:
        print ("ERROR: given char length != 1 (given char: \"%s\")" % (l))
        sys.exit()
    for c in alphabet:
        if c == char:
            vector.append(1)
        else:
            vector.append(0)
    return vector


################################################################################

def filter_seq_dic_fixed_vp_len(seqs_dic, vp_s_dic, vp_e_dic, vp_l):
    """
    Remove sequences from sequence dictionary that do have a viewpoint 
    length != given vp_l.
    
    >>> seqs_dic = {"CLIP_01" : "acgtACGTacgt", "CLIP_02" : "aaaCCCggg"}
    >>> vp_s_dic = {"CLIP_01" : 5, "CLIP_02" : 4}
    >>> vp_e_dic = {"CLIP_01" : 8, "CLIP_02" : 6}
    >>> filter_seq_dic_fixed_vp_len(seqs_dic, vp_s_dic, vp_e_dic, 4)
    >>> seqs_dic
    {'CLIP_01': 'acgtACGTacgt'}

    """
    seqs2del = {}
    for seq_id in seqs_dic:
        # Get viewpoint start+end of sequence.
        vp_s = vp_s_dic[seq_id]
        vp_e = vp_e_dic[seq_id]
        l_vp = vp_e - vp_s + 1
        if vp_l != l_vp:
            seqs2del[seq_id] = 1
    for seq_id in seqs2del:
        del seqs_dic[seq_id]


################################################################################

def extract_viewpoint_regions_from_fasta(seqs_dic,
                                         vp_s_dic=False,
                                         vp_e_dic=False):
    """
    Extract viewpoint start end end positions from FASTA dictionary.
    Return dictionaries for start+end (1-based indices, key:fasta_id).

    >>> seqs_dic = {"id1": "acguACGUacgu", "id2": "ACGUacgu"}
    >>> vp_s, vp_e = extract_viewpoint_regions_from_fasta(seqs_dic)
    >>> vp_s["id1"] == 5
    True
    >>> vp_e["id1"] == 8
    True
    >>> vp_s["id2"] == 1
    True
    >>> vp_e["id2"] == 4
    True

    """
    if not vp_s_dic:
        vp_s_dic = {}
    if not vp_e_dic:
        vp_e_dic = {}
    for seq_id, seq in sorted(seqs_dic.items()):
        m = re.search("([acgun]*)([ACGUN]+)", seq)
        if m:
            l_us = len(m.group(1))
            l_vp = len(m.group(2))
            vp_s = l_us+1
            vp_e = l_us+l_vp
            vp_s_dic[seq_id] = vp_s
            vp_e_dic[seq_id] = vp_e
        else:
            print ("ERROR: viewpoint extraction failed for \"%s\"" % (seq_id))
            sys.exit()
    return vp_s_dic, vp_e_dic


################################################################################

def read_str_elem_up_into_dic(str_elem_up_file,
                              str_elem_up_dic=False):

    """
    Read in structural elements unpaired probabilities for each sequence 
    position. Available structural elements:
    p_unpaired, p_external, p_hairpin, p_internal, p_multiloop, p_paired
    Input Format:
    >sequence_id
    pos(1-based)<t>p_unpaired<t>p_external<t>p_hairpin<t>p_internal<t>p_multiloop<t>p_paired
    Extract: p_external, p_hairpin, p_internal, p_multiloop
    thus getting 4xn matrix for sequence with length n
    Return dictionary with matrix for each sequence.
    (key: sequence id, value: ups matrix)

    >>> str_elem_up_test = "test_data/test.str_elem.up"
    >>> read_str_elem_up_into_dic(str_elem_up_test)
    {'CLIP_01': [[0.1, 0.2], [0.2, 0.3], [0.4, 0.2], [0.2, 0.1], [0.1, 0.2]]}

    """
    if not str_elem_up_dic:
        str_elem_up_dic = {}
    seq_id = ""
    # Go through .str_elem.up file, extract p_external, p_hairpin, p_internal, p_multiloop.
    with open(str_elem_up_file) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                str_elem_up_dic[seq_id] = [[],[],[],[],[]]
            else:
                m = re.search("\d+\t.+?\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t(.+?)\n", line)
                p_external = float(m.group(1))
                p_hairpin = float(m.group(2))
                p_internal = float(m.group(3))
                p_multiloop = float(m.group(4))
                p_stack = float(m.group(5))
                str_elem_up_dic[seq_id][0].append(p_external)
                str_elem_up_dic[seq_id][1].append(p_hairpin)
                str_elem_up_dic[seq_id][2].append(p_internal)
                str_elem_up_dic[seq_id][3].append(p_multiloop)
                str_elem_up_dic[seq_id][4].append(p_stack)
    f.closed
    return str_elem_up_dic


################################################################################

def read_entr_into_dic(entr_file,
                       entr_dic=False,
                       mean_norm=False):

    """
    Read RBP occupancy+entropy scores for each sequence into dictionary.
    RBP occupancy+entropy scores are calculated for viewpoint regions only.
    
    (key: sequence id, value: vector of feature values)

    test.entr file content:
    id	avg_rbpc	avg_cov	entr
    CLIP_01	0.03	6.05	3.18
    CLIP_02	0.02	4.37	3.27

    Features:
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

    >>> entr_test = "test_data/test.entr"
    >>> read_entr_into_dic(entr_test)
    {'CLIP_01': [0.03, 6.05, 3.18], 'CLIP_02': [0.02, 4.37, 3.27]}

    """
    if not entr_dic:
        entr_dic = {}
    seq_id = ""
    max_v = [-1000, -1000, -1000]
    min_v = [1000, 1000, 1000]
    avg_v = [0, 0, 0]
    # Go through .up file, extract unpaired probs for each position.
    with open(entr_file) as f:
        for line in f:
            f_list = line.strip().split("\t")
            # Skip header line(s).
            if f_list[0] == "id":
                continue
            seq_id = f_list[0]
            f_list.pop(0)
            if not seq_id in entr_dic:
                entr_dic[seq_id] = []
            for i,v in enumerate(f_list):
                v = float(v)
                if v > max_v[i]:
                    max_v[i] = v
                if v < min_v[i]:
                    min_v[i] = v
                avg_v[i] += v
                entr_dic[seq_id].append(v)
    f.closed
    if mean_norm:
        for i,v in enumerate(avg_v):
            avg_v[i] = avg_v[i] / len(entr_dic)
        for seq_id in entr_dic:
            for i in range(len(entr_dic[seq_id])):
                entr_dic[seq_id][i] = mean_normalize(entr_dic[seq_id][i], avg_v[i], max_v[i], min_v[i])
    return entr_dic


################################################################################

def read_sf_into_dic(sf_file,
                     sf_dic=False,
                     mean_norm=False):

    """
    Read site features into dictionary.
    (key: sequence id, value: vector of site feature values)

    >>> sf_test = "test_data/test.sf"
    >>> read_sf_into_dic(sf_test)
    {'CLIP_01': [0.2, 0.3, 0.3, 0.2], 'CLIP_02': [0.1, 0.2, 0.4, 0.3]}

    """
    if not sf_dic:
        sf_dic = {}
    seq_id = ""
    max_v = False
    min_v = False
    avg_v = False
    # Go through .up file, extract unpaired probs for each position.
    with open(sf_file) as f:
        for line in f:
            f_list = line.strip().split("\t")
            # Init min, max, avg lists.
            if not max_v:
                max_v = [-1000]*(len(f_list)-1)
                min_v = [1000]*(len(f_list)-1)
                avg_v = [0]*(len(f_list)-1)
            # Skip header line(s).
            if f_list[0] == "id":
                continue
            seq_id = f_list[0]
            f_list.pop(0)
            if not seq_id in sf_dic:
                sf_dic[seq_id] = []
            for i,v in enumerate(f_list):
                v = float(v)
                if v > max_v[i]:
                    max_v[i] = v
                if v < min_v[i]:
                    min_v[i] = v
                avg_v[i] += v
                sf_dic[seq_id].append(v)
    f.closed
    if mean_norm:
        for i,v in enumerate(avg_v):
            avg_v[i] = avg_v[i] / len(sf_dic)
        for seq_id in sf_dic:
            for i in range(len(sf_dic[seq_id])):
                sf_dic[seq_id][i] = mean_normalize(sf_dic[seq_id][i], avg_v[i], max_v[i], min_v[i])
    return sf_dic


################################################################################

def normalize_sf_dic(sf_dic,
                     norm_mode=0):
    """
    Mean normalize sf_dic values or any dictinary with value=vector of feature 
    values.

    norm_mode : normalization mode
                norm_mode=0 : Min-max normalization
                norm_mode=1 : Mean normalization

    >>> test_dic = {"id1": [0.5, 2.5], "id2": [1, 3], "id3": [1.5, 3.5]}
    >>> normalize_sf_dic(test_dic, norm_mode=1)
    {'id1': [-0.5, -0.5], 'id2': [0.0, 0.0], 'id3': [0.5, 0.5]}

    """
    max_v = False
    min_v = False
    avg_v = False
    # Get min, max, averaging sum for each feature.
    for seq_id in sf_dic:
        l_v = len(sf_dic[seq_id])
        if not max_v:
            max_v = [-1000]*l_v
            min_v = [1000]*l_v
            avg_v = [0]*l_v
        for i,v in enumerate(sf_dic[seq_id]):
            if v > max_v[i]:
                max_v[i] = v
            if v < min_v[i]:
                min_v[i] = v
            avg_v[i] += v
    # Mean normalize.
    for i,v in enumerate(avg_v):
        avg_v[i] = avg_v[i] / len(sf_dic)
    for seq_id in sf_dic:
        for i in range(len(sf_dic[seq_id])):
            if norm_mode == 0:
                sf_dic[seq_id][i] = min_max_normalize(sf_dic[seq_id][i], max_v[i], min_v[i])
            elif norm_mode == 1:
                sf_dic[seq_id][i] = mean_normalize(sf_dic[seq_id][i], avg_v[i], max_v[i], min_v[i])
            else:
                print("ERROR: invalid norm_mode \"%i\" set in normalize_sf_dic()" % (norm_mode))
                sys.exit()
    return sf_dic


################################################################################

def normalize_pos_neg_sf_dic(pos_sf_dic, neg_sf_dic,
                             norm_mode=0):
    """
    Mean normalize pos+neg sf_dic values.

    norm_mode : normalization mode
                norm_mode=0 : Min-max normalization
                norm_mode=1 : Mean normalization
    """
    max_v = False
    min_v = False
    avg_v = False
    # Get min, max, averaging sum for each feature, positives.
    for seq_id in pos_sf_dic:
        l_v = len(pos_sf_dic[seq_id])
        if not max_v:
            max_v = [-1000]*l_v
            min_v = [1000]*l_v
            avg_v = [0]*l_v
        for i,v in enumerate(pos_sf_dic[seq_id]):
            if v > max_v[i]:
                max_v[i] = v
            if v < min_v[i]:
                min_v[i] = v
            avg_v[i] += v
    # Negatives.
    for seq_id in neg_sf_dic:
        for i,v in enumerate(neg_sf_dic[seq_id]):
            if v > max_v[i]:
                max_v[i] = v
            if v < min_v[i]:
                min_v[i] = v
            avg_v[i] += v
    # Number of pos+neg instances.
    c_v = len(pos_sf_dic) + len(neg_sf_dic)
    # Mean normalize.
    for i,v in enumerate(avg_v):
        avg_v[i] = avg_v[i] / c_v
    for seq_id in pos_sf_dic:
        for i in range(len(pos_sf_dic[seq_id])):
            if norm_mode == 0:
                pos_sf_dic[seq_id][i] = min_max_normalize(pos_sf_dic[seq_id][i], max_v[i], min_v[i])
            elif norm_mode == 1:
                pos_sf_dic[seq_id][i] = mean_normalize(pos_sf_dic[seq_id][i], avg_v[i], max_v[i], min_v[i])
            else:
                print("ERROR: invalid norm_mode \"%i\" set in normalize_pos_neg_sf_dic()" % (norm_mode))
                sys.exit()
    for seq_id in neg_sf_dic:
        for i in range(len(neg_sf_dic[seq_id])):
            if norm_mode == 0:
                neg_sf_dic[seq_id][i] = min_max_normalize(neg_sf_dic[seq_id][i], max_v[i], min_v[i])
            elif norm_mode == 1:
                neg_sf_dic[seq_id][i] = mean_normalize(neg_sf_dic[seq_id][i], avg_v[i], max_v[i], min_v[i])
            else:
                print("ERROR: invalid norm_mode \"%i\" set in normalize_pos_neg_sf_dic()" % (norm_mode))
                sys.exit()
    return pos_sf_dic, neg_sf_dic


################################################################################

def read_up_into_dic(up_file,
                     up_dic=False):

    """
    Read in unpaired probabilities and store probability value list for 
    each sequence.
    Return dictionary with unpaired probability list for each sequence
    (key: sequence id, value: unpaired probability list), so length of 
    list == sequence length.

    >>> up_test = "test_data/test.up"
    >>> read_up_into_dic(up_test)
    {'CLIP_01': [0.11, 0.22], 'CLIP_02': [0.33, 0.44]}
    >>> up_test = "test_data/test2.up"
    >>> read_up_into_dic(up_test)
    {}

    """
    if not up_dic:
        up_dic = {}
    seq_id = ""
    # Go through .up file, extract unpaired probs for each position.
    with open(up_file) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                up_dic[seq_id] = []
            else:
                m = re.search("\d+\t(.+)", line)
                up = float(m.group(1))
                up_dic[seq_id].append(up)
    f.closed
    return up_dic


################################################################################

def normalize_con_dic(con_dic):
    """
    Mean normalize con_dic values (only phyloP scores).

    >>> test_dic = {"id1": [[0.5, 1, 1.5], [2.5, 3, 3.5]], "id2": [[0.5, 1, 1.5], [2.5, 3, 3.5]]}
    >>> normalize_con_dic(test_dic)
    {'id1': [[0.5, 1, 1.5], [-0.5, 0.0, 0.5]], 'id2': [[0.5, 1, 1.5], [-0.5, 0.0, 0.5]]}

    """
    # Mean normalization for phyloP scores.
    pp_max = -1000
    pp_min = 1000
    pp_sum = 0
    pp_c = 0
    # Get min, max, averaging sum for each feature.
    for seq_id in con_dic:
        for i,v in enumerate(con_dic[seq_id][1]):
            if v > pp_max:
                pp_max = v
            if v < pp_min:
                pp_min = v
            pp_sum += v
            pp_c += 1
    pp_mean = pp_sum / pp_c
    for seq_id in con_dic:
        for i in range(len(con_dic[seq_id][1])):
            con_dic[seq_id][1][i] = mean_normalize(con_dic[seq_id][1][i], pp_mean, pp_max, pp_min)
    return con_dic


################################################################################

def normalize_pos_neg_con_dic(pos_con_dic, neg_con_dic):
    """
    Mean normalize pos+neg con_dic values (only phyloP scores).
    """
    # Mean normalization for phyloP scores.
    pp_max = -1000
    pp_min = 1000
    pp_sum = 0
    pp_c = 0
    # Get min, max, averaging sum for each feature.
    for seq_id in pos_con_dic:
        for i,v in enumerate(pos_con_dic[seq_id][1]):
            if v > pp_max:
                pp_max = v
            if v < pp_min:
                pp_min = v
            pp_sum += v
            pp_c += 1
    # Negatives.
    for seq_id in neg_con_dic:
        for i,v in enumerate(neg_con_dic[seq_id][1]):
            if v > pp_max:
                pp_max = v
            if v < pp_min:
                pp_min = v
            pp_sum += v
            pp_c += 1
    # Mean.
    pp_mean = pp_sum / pp_c
    for seq_id in pos_con_dic:
        for i in range(len(pos_con_dic[seq_id][1])):
            pos_con_dic[seq_id][1][i] = mean_normalize(pos_con_dic[seq_id][1][i], pp_mean, pp_max, pp_min)
    for seq_id in neg_con_dic:
        for i in range(len(neg_con_dic[seq_id][1])):
            neg_con_dic[seq_id][1][i] = mean_normalize(neg_con_dic[seq_id][1][i], pp_mean, pp_max, pp_min)
    return pos_con_dic, neg_con_dic


################################################################################

def read_con_into_dic(con_file,
                      con_dic=False,
                      mean_norm=False):
    """
    Read in conservation scores (phastCons+phyloP) and store scores as 
    2xn matrix for sequence with length n. 
    Return dictionary with matrix for each sequence
    (key: sequence id, value: scores matrix)
    Entry format: [[1,2,3],[4,5,6]] : 2x3 format (2 rows, 3 columns)
    mean_norm  :  mean normalize phyloP scores.
    phastCons scores already normalized since probabilities 0 .. 1

    >>> con_test = "test_data/test.con"
    >>> read_con_into_dic(con_test)
    {'CLIP_01': [[0.1, 0.2], [0.3, -0.4]], 'CLIP_02': [[0.4, 0.5], [0.6, 0.7]]}

    """
    if not con_dic:
        con_dic = {}
    seq_id = ""
    # Mean normalization for phyloP scores.
    pp_max = -1000
    pp_min = 1000
    pp_sum = 0
    pp_c = 0
    # Go through .con file, extract phastCons, phyloP scores for each position.
    with open(con_file) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                con_dic[seq_id] = [[],[]]
            else:
                m = re.search("\d+\t(.+?)\t(.+)", line)
                phastcons_sc = float(m.group(1))
                phylop_sc = float(m.group(2))
                con_dic[seq_id][0].append(phastcons_sc)
                con_dic[seq_id][1].append(phylop_sc)
                pp_c += 1
                pp_sum += phylop_sc
                if phylop_sc > pp_max:
                    pp_max = phylop_sc
                if phylop_sc < pp_min:
                    pp_min = phylop_sc
    f.closed
    # Mean normalize phylop scores.
    if mean_norm:
        pp_mean = pp_sum / pp_c
        for seq_id in con_dic:
            for i in range(len(con_dic[seq_id][1])):
                con_dic[seq_id][1][i] = mean_normalize(con_dic[seq_id][1][i], pp_mean, pp_max, pp_min)
    return con_dic


################################################################################

def mean_normalize(x, mean_x, max_x, min_x):
    """
    Mean normalization of input x, given dataset mean, max, and min.
    
    >>> mean_normalize(10, 10, 15, 5)
    0.0
    >>> mean_normalize(15, 20, 30, 10)
    -0.25
    
    Formula from:
    https://en.wikipedia.org/wiki/Feature_scaling
    
    """
    # If min=max, all values the same, so return x.
    if (max_x - min_x) == 0:
        return x
    else:
        return ( (x-mean_x) / (max_x - min_x) )


################################################################################

def min_max_normalize(x, max_x, min_x):
    """
    Min-max normalization of input x, given dataset max and min.
    
    >>> min_max_normalize(20, 30, 10)
    0.5
    >>> min_max_normalize(30, 30, 10)
    1.0
    >>> min_max_normalize(10, 30, 10)
    0.0
    
    Formula from:
    https://en.wikipedia.org/wiki/Feature_scaling
    
    """
    # If min=max, all values the same, so return x.
    if (max_x - min_x) == 0:
        return x
    else:
        return ( (x-min_x) / (max_x - min_x) )


################################################################################

def normalize_graph_feat_vectors(graphs,
                                 norm_mode=0):
    """
    Normalize graph feature vector values. Automatically check for 
    one-hot encoded features (only "0" or "1" values), do not normalize these.
    
    graph     : List of graphs
    norm_mode : normalization mode
                norm_mode=0 : Min-max normalization
                norm_mode=1 : Mean normalization

    >>> seqs_dic = {"CLIP_01" : "aCGu"}
    >>> up_dic = {"CLIP_01" : [0.8]*4}
    >>> vp_s = {"CLIP_01" : 2}
    >>> vp_e = {"CLIP_01" : 3}
    >>> region_labels_dic = {"CLIP_01" : ['E', 'E', 'I', 'I']}
    >>> con_dic = {"CLIP_01" : [[0.3, 0.5, 0.7, 0.2], [0.2, 0.8, 0.9, 0.1]]}
    >>> str_elem_up_dic = {"CLIP_01": [[0.1, 0.3, 0.25, 0.2], [0.15, 0.1, 0.3, 0.25], [0.2, 0.15, 0.1, 0.3], [0.25, 0.2, 0.15, 0.1], [0.3, 0.25, 0.2, 0.15]]}
    >>> g_list = convert_seqs_to_graphs(seqs_dic, vp_s, vp_e, up_dic=up_dic, con_dic=con_dic, str_elem_up_dic=str_elem_up_dic, region_labels_dic=region_labels_dic)
    >>> g_list[0].node[0]['feat_vector']
    [0.3, 0.1, 0.15, 0.2, 0.25, 0.5, 0.8, 1, 0]
    >>> g_list[0].node[1]['feat_vector']
    [0.25, 0.3, 0.1, 0.15, 0.2, 0.7, 0.9, 0, 1]
    >>> normalize_graph_feat_vectors(g_list, norm_mode=0)
    >>> g_list[0].node[0]['feat_vector']
    [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1, 0]
    >>> g_list[0].node[1]['feat_vector']
    [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, 1]

    """
    # Get number of vector elements.
    c_ve = len(graphs[0].node[0]['feat_vector'])
    # Identify elements with one-hot encoding (use first graph for that).
    norm_i = [] # vector indices to normalize.
    for i in range(c_ve):
        one_hot = 1
        for n in graphs[0].nodes:
            v = graphs[0].node[n]['feat_vector'][i]
            if str(v) != "0" and str(v) != "1":
                one_hot = 0
        if not one_hot:
            norm_i.append(i)
    # Vectors of max, min, avg values for each features.
    max_v = [-1000]*c_ve
    min_v = [1000]*c_ve
    avg_v = [0]*c_ve
    c_v = [0]*c_ve
    for i in norm_i: # For each normalization index.
        for g in graphs:
            for n in g.nodes: # For each node in graph.
                v = g.node[n]['feat_vector'][i]
                c_v[i] += 1
                if v > max_v[i]:
                    max_v[i] = v
                if v < min_v[i]:
                    min_v[i] = v
                avg_v[i] += v
    # Calculate means.
    for i in norm_i:
        avg_v[i] = avg_v[i] / c_v[i]
    # Normalize all vector values.
    for i in norm_i: # For each normalization index.
        for g in graphs:
            for n in g.nodes: # For each node in graph.
                if norm_mode == 0:
                    g.node[n]['feat_vector'][i] = min_max_normalize(g.node[n]['feat_vector'][i], max_v[i], min_v[i])
                elif norm_mode == 1:
                    g.node[n]['feat_vector'][i] = mean_normalize(g.node[n]['feat_vector'][i], avg_v[i], max_v[i], min_v[i])
                else:
                    print("ERROR: invalid norm_mode \"%i\" set in normalize_graph_feat_vectors()" % (norm_mode))
                    sys.exit()


################################################################################

def read_bpp_into_dic(bpp_file, vp_s, vp_e,
                      bpp_dic=False,
                      vp_lr_ext=100,
                      ext_mode=1):
    """
    Read in base pair probabilities and store information in list for 
    each sequence, where region to extract values from is defined by 
    viewpoint (vp) start+end (+ vp_lr_ext == plfold L parameter, assuming L=100)
    Return dictionary with base pair+probability list for each sequence
    (key: sequence id, value: "bp_start-bp_end,bp_prob").
    ext_mode: define which base pairs get extracted.
    ext_mode=1 : all bps in extended vp region vp_s-vp_lr_ext - vp_e+vp_lr_ext
    ext_mode=2 : bps in extended vp region with start or end in base vp
    ext_mode=3 : only bps with start+end in base vp

    >>> bpp_test = "test_data/test.bpp"
    >>> vp_s = {"CLIP_01": 150}
    >>> vp_e = {"CLIP_01": 250}
    >>> d = read_bpp_into_dic(bpp_test, vp_s, vp_e, vp_lr_ext=50, ext_mode=1)
    >>> print(d)
    {'CLIP_01': ['110-140,0.22', '130-150,0.33', '160-200,0.44', '240-260,0.55', '270-290,0.66']}
    >>> d = read_bpp_into_dic(bpp_test, vp_s, vp_e, vp_lr_ext=50, ext_mode=2)
    >>> print(d)
    {'CLIP_01': ['130-150,0.33', '160-200,0.44', '240-260,0.55']}
    >>> d = read_bpp_into_dic(bpp_test, vp_s, vp_e, vp_lr_ext=50, ext_mode=3)
    >>> print(d)
    {'CLIP_01': ['160-200,0.44']}
    >>> bpp_test = "test_data/test2.bpp"
    >>> d = read_bpp_into_dic(bpp_test, vp_s, vp_e, vp_lr_ext=50, ext_mode=1)
    >>> print(d)
    {}

    """
    if not bpp_dic:
        bpp_dic = {}
    seq_id = ""
    # Go through FASTA file, extract sequences.
    with open(bpp_file) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                if not seq_id in vp_s:
                    print ("ERROR: bpp_file \"%s\" id \"%s\" not found in vp_s  in \"%s\"" % (bpp_file, seq_id))
                    sys.exit()
                bpp_dic[seq_id] = []
            else:
                m = re.search("(\d+)\t(\d+)\t(.+)", line)
                s = int(m.group(1))
                e = int(m.group(2))
                bpp = float(m.group(3))
                bpp_se = "%s-%s,%s" % (s,e,bpp)
                if ext_mode == 1:
                    if s >= (vp_s[seq_id]-vp_lr_ext) and e <= (vp_e[seq_id]+vp_lr_ext):
                        bpp_dic[seq_id].append(bpp_se)
                elif ext_mode == 2:
                    if (s >= (vp_s[seq_id]-vp_lr_ext) and (e <= (vp_e[seq_id]) and e >= vp_s[seq_id])) or (e <= (vp_e[seq_id]+vp_lr_ext) and (s <= (vp_e[seq_id]) and s >= vp_s[seq_id])):
                        bpp_dic[seq_id].append(bpp_se)
                elif ext_mode == 3:
                    if s >= vp_s[seq_id] and e <= vp_e[seq_id]:
                        bpp_dic[seq_id].append(bpp_se)
                else:
                    print ("ERROR: invalid ext_mode given (valid values: 1,2,3)")
                    sys.exit()
    f.closed
    return bpp_dic


################################################################################

def generate_geometric_data(seqs_dic, vp_s_dic, vp_e_dic, 
                            g_list=False,
                            up_dic=False,
                            bpp_dic=False,
                            con_dic=False,
                            str_elem_up_dic=False,
                            use_str_elem_1h=False,
                            use_us_ds_labels=False,
                            region_labels_dic=False,
                            plfold_bpp_cutoff=0.2,
                            vp_lr_ext=100, 
                            fix_vp_len=False,
                            add_1h_to_g=False,
                            g_idx=False,
                            n_idx=False,
                            ext_mode=1):
    """
    Generate PyTorch Geometric graph format data.
    Return the following lists:
    all_nodes_labels       Nucleotide indices (dict_label_idx)
    all_graph_indicators   Graph indices, each node of a graph 
                           gets same index
    all_edges              Indices of edges
    all_nodes_attributes   Node vectors
    """

    # Label to idx dictionary.
    dict_label_idx = {'a': '1', 
                      'c': '2', 
                      'g': '3', 
                      'u': '4', 
                      'A': '5', 
                      'C': '6', 
                      'G': '7', 
                      'U': '8', 
                      'ua': '9', 
                      'uc': '10', 
                      'ug': '11', 
                      'uu': '12', 
                      'da': '13', 
                      'dc': '14', 
                      'dg': '15', 
                      'du': '16'}
    # Init lists.
    all_nodes_labels = []
    all_graph_indicators = []
    all_edges = []
    all_nodes_attributes = []
    if not g_idx:
        g_idx = 0
    if not n_idx:
        n_idx = 1
    for seq_id, seq in sorted(seqs_dic.items()):
        # Get viewpoint start+end of sequence.
        vp_s = vp_s_dic[seq_id]
        vp_e = vp_e_dic[seq_id]
        l_vp = vp_e - vp_s + 1
        # If fixed vp length given, only store sites with this vp length.
        if fix_vp_len:
            if not l_vp == fix_vp_len:
                continue
        l_seq = len(seq)
        # Subsequence extraction start + end (1-based).
        ex_s = vp_s
        ex_e = vp_e
        # If base pair probabilities given, adjust extraction s+e.
        if bpp_dic:
            if not seq_id in bpp_dic:
                print ("ERROR: seq_id \"%s\" not in bpp_dic" % (seq_id))
                sys.exit()
            ex_s = ex_s-vp_lr_ext
            if ex_s < 1:
                ex_s = 1
            ex_e = ex_e+vp_lr_ext
            if ex_e > l_seq:
                ex_e = l_seq
        # Length of graph.
        n_nodes = ex_e - ex_s + 1
        # Graph indicator.
        all_graph_indicators.extend([g_idx+1]*n_nodes)
        # Check up_dic.
        if up_dic:
            if not seq_id in up_dic:
                print ("ERROR: seq_id \"%s\" not in up_dic" % (seq_id))
                sys.exit()
            if len(up_dic[seq_id]) != l_seq:
                print ("ERROR: up_dic[seq_id] length != sequence length for seq_id \"%s\"" % (seq_id))
                sys.exit()
        # Check str_elem_up_dic:
        if str_elem_up_dic:
            if not seq_id in str_elem_up_dic:
                print ("ERROR: seq_id \"%s\" not in str_elem_up_dic" % (seq_id))
                sys.exit()
            if len(str_elem_up_dic[seq_id][0]) != l_seq:
                print ("ERROR: str_elem_up_dic[seq_id] length != sequence length for seq_id \"%s\"" % (seq_id))
                sys.exit()
        # Check con_dic.
        if con_dic:
            if not seq_id in con_dic:
                print ("ERROR: seq_id \"%s\" not in con_dic" % (seq_id))
                sys.exit()
            if len(con_dic[seq_id][0]) != l_seq:
                print ("ERROR: con_dic[seq_id] length != sequence length for seq_id \"%s\"" % (seq_id))
                sys.exit()
        # Check region_labels_dic.
        if region_labels_dic:
            if not seq_id in region_labels_dic:
                print ("ERROR: seq_id \"%s\" not in region_labels_dic" % (seq_id))
                sys.exit()
        # Add feature values per position.
        g_i = 0
        seen_vp = False
        for i,c in enumerate(seq): # i from 0.. l-1
            # Skip if outside region of interest.
            if i < (ex_s-1) or i > (ex_e-1):
                continue
            # Label upstream / downstream nucleotides differently.
            if use_us_ds_labels:
                lc = ["a", "c", "g", "u"]
                uc = ["A", "C", "G", "U"]
                new_c = c
                if c in lc:
                    if seen_vp:
                        new_c = "d"+c
                    else:
                        new_c = "u"+c
                else:
                    seen_vp = True
                all_nodes_labels.append(dict_label_idx[new_c])
            else:
                # Add nucleotide node.
                all_nodes_labels.append(dict_label_idx[c])
            # Make feature vector for each graph node.
            if up_dic or str_elem_up_dic or con_dic or region_labels_dic:
                feat_vector = []
                if add_1h_to_g:
                    feat_vector = char_vectorizer(c)
                if up_dic:
                    if not str_elem_up_dic:
                        feat_vector.append(up_dic[seq_id][i])
                if str_elem_up_dic:
                    # Structural elements unpaired probabilities.
                    p_e = str_elem_up_dic[seq_id][0][i]
                    p_h = str_elem_up_dic[seq_id][1][i]
                    p_i = str_elem_up_dic[seq_id][2][i]
                    p_m = str_elem_up_dic[seq_id][3][i]
                    p_s = str_elem_up_dic[seq_id][4][i]
                    # Unpaired probabilities as one-hot (max_prob_elem=1, else 0).
                    if use_str_elem_1h:
                        str_c = "E"
                        p_max = p_e
                        if p_h > p_max:
                            str_c = "H"
                            p_max = p_h
                        if p_i > p_max:
                            str_c = "I"
                            p_max = p_i
                        if p_m > p_max:
                            str_c = "M"
                            p_max = p_m
                        if p_s > p_max:
                            str_c = "S"
                        str_c_1h = char_vectorizer(str_c, 
                                       custom_alphabet = ["E", "H", "I", "M", "S"])
                        for v in str_c_1h:
                            feat_vector.append(v)
                    else:
                        # Add probabilities to vector.
                        feat_vector.append(p_e) # E
                        feat_vector.append(p_h) # H
                        feat_vector.append(p_i) # I
                        feat_vector.append(p_m) # M
                        feat_vector.append(p_s) # S
                # Conservation scores.
                if con_dic:
                    feat_vector.append(con_dic[seq_id][0][i])
                    feat_vector.append(con_dic[seq_id][1][i])
                # Region labels (exon intron).
                if region_labels_dic:
                    label = region_labels_dic[seq_id][i]
                    label_1h = char_vectorizer(label,
                                               custom_alphabet = ["E", "I"])
                    for v in label_1h:
                        feat_vector.append(v)
                        
                node_attribute = [str(att) for att in feat_vector]
                #g.node[g_i]['feat_vector'] = feat_vector
                all_nodes_attributes.append(",".join(node_attribute))
            # Add backbone edge.
            if g_i > 0:
                all_edges.append((g_i-1+n_idx, g_i+n_idx))
                all_edges.append((g_i+n_idx, g_i-1+n_idx))
            # Increment graph node index.
            g_i += 1
        # Add base pair edges to graph.
        if bpp_dic:
            for entry in bpp_dic[seq_id]:
                m = re.search("(\d+)-(\d+),(.+)", entry)
                p1 = int(m.group(1))
                p2 = int(m.group(2))
                bpp_value = float(m.group(3))
                g_p1 = p1 - ex_s # 0-based base pair p1.
                g_p2 = p2 - ex_s # 0-based base pair p2.
                # Filter.
                if bpp_value < plfold_bpp_cutoff: continue
                # Add edge if bpp value >= threshold.
                if ext_mode == 1:
                    if p1 >= ex_s and p2 <= ex_e:
                        all_edges.append((g_p1+n_idx, g_p2+n_idx))
                        all_edges.append((g_p2+n_idx, g_p1+n_idx))
                elif ext_mode == 2:
                    if (p1 >= ex_s and p2 <= vp_e and p2 >= vp_s) or (p2 <= ex_e and p1 <= vp_e and p1 >= vp_s):
                        all_edges.append((g_p1+n_idx, g_p2+n_idx))
                        all_edges.append((g_p2+n_idx, g_p1+n_idx))
                elif ext_mode == 3:
                    if p1 >= vp_s and p2 <= vp_e:
                        all_edges.append((g_p1+n_idx, g_p2+n_idx))
                        all_edges.append((g_p2+n_idx, g_p1+n_idx))
                else:
                    print ("ERROR: invalid ext_mode given (valid values: 1,2,3)")
                    sys.exit()
        n_idx += n_nodes
        # Append graph to list.
        g_idx += 1
    return all_nodes_labels, all_graph_indicators, all_edges, all_nodes_attributes, g_idx, n_idx


################################################################################

def convert_seqs_to_graphs(seqs_dic, vp_s_dic, vp_e_dic, 
                           g_list=False,
                           up_dic=False,
                           bpp_dic=False,
                           con_dic=False,
                           str_elem_up_dic=False,
                           use_str_elem_1h=False,
                           use_us_ds_labels=False,
                           region_labels_dic=False,
                           plfold_bpp_cutoff=0.2,
                           vp_lr_ext=100, 
                           fix_vp_len=False,
                           add_1h_to_g=False,
                           ext_mode=1):
    """
    Convert dictionary of sequences into list of networkx graphs.
    Input: sequence dictionary, viewpoint dictionaries (start+end), 
    and optionally base pair + unpaired probability dictionaries (for 
    creating structure graphs).
    Unpaired probabilities are added per node in the graph.
    Format bpp entry string: "bp_start-bp_end,bp_prob"
    Probability cutoff can be set via plfold_bpp_cutoff (i.e. if bpp is 
    lower the base pair edge will not be added to graph).
    ext_mode: define which base pairs get extracted.
    ext_mode=1 : all bps in extended vp region vp_s-vp_lr_ext - vp_e+vp_lr_ext
    ext_mode=2 : bps in extended vp region with start or end in base vp
    ext_mode=3 : only bps with start+end in non-extended vp region

    >>> seqs_dic = {"CLIP_01" : "acguaacguaACGUACGUACGacguaacgua", "CLIP_02" : "cguaACGUACGUACGacgua"}
    >>> up_dic = {"CLIP_01" : [0.5]*31, "CLIP_02" : [0.4]*20}
    >>> id1_bp = ["2-5,0.3", "6-8,0.3", "7-12,0.3", "13-18,0.4", "19-23,0.3", "24-30,0.3"]
    >>> id2_bp = ["7-11,0.5"]
    >>> bpp_dic = {"CLIP_01" : id1_bp, "CLIP_02" : id2_bp}
    >>> vp_s = {"CLIP_01": 11, "CLIP_02": 5}
    >>> vp_e = {"CLIP_01": 21, "CLIP_02": 15}
    >>> g_list = convert_seqs_to_graphs(seqs_dic, vp_s, vp_e, up_dic=up_dic, vp_lr_ext=5, ext_mode=1, bpp_dic=bpp_dic)
    >>> convert_graph_to_string(g_list[0])
    '0-1,0-2,1-2,1-6,2-3,3-4,4-5,5-6,6-7,7-8,7-12,8-9,9-10,10-11,11-12,12-13,13-14,13-17,14-15,15-16,16-17,17-18,18-19,19-20,'
    >>> convert_graph_to_string(g_list[1])
    '0-1,1-2,2-3,3-4,4-5,5-6,6-7,6-10,7-8,8-9,9-10,10-11,11-12,12-13,13-14,14-15,15-16,16-17,17-18,18-19,'
    >>> seqs_dic = {"CLIP_01" : "aCGu"}
    >>> up_dic = {"CLIP_01" : [0.8]*4}
    >>> vp_s = {"CLIP_01" : 2}
    >>> vp_e = {"CLIP_01" : 3}
    >>> region_labels_dic = {"CLIP_01" : ['E', 'E', 'I', 'I']}
    >>> con_dic = {"CLIP_01" : [[0.3, 0.5, 0.7, 0.2], [0.2, 0.8, 0.9, 0.1]]}
    >>> str_elem_up_dic = {"CLIP_01": [[0.1]*4, [0.2]*4, [0.4]*4, [0.2]*4, [0.1]*4]}
    >>> g_list = convert_seqs_to_graphs(seqs_dic, vp_s, vp_e, up_dic=up_dic, con_dic=con_dic, str_elem_up_dic=str_elem_up_dic, region_labels_dic=region_labels_dic)
    >>> g_list[0].node[0]['feat_vector']
    [0.1, 0.2, 0.4, 0.2, 0.1, 0.5, 0.8, 1, 0]
    >>> g_list[0].node[1]['feat_vector']
    [0.1, 0.2, 0.4, 0.2, 0.1, 0.7, 0.9, 0, 1]
    >>> bpp_dic = {"CLIP_01" : ["1-4,0.5"]}
    >>> g_list = convert_seqs_to_graphs(seqs_dic, vp_s, vp_e, vp_lr_ext=1, ext_mode=1, bpp_dic=bpp_dic, use_us_ds_labels=True)
    >>> convert_graph_to_string(g_list[0])
    '0-1,0-3,1-2,2-3,'
    >>> g_list[0].node[0]['label']
    'ua'
    >>> g_list[0].node[1]['label']
    'C'
    >>> g_list[0].node[2]['label']
    'G'
    >>> g_list[0].node[3]['label']
    'du'

    """
    if not g_list:
        g_list = []
    for seq_id, seq in sorted(seqs_dic.items()):
        # Get viewpoint start+end of sequence.
        vp_s = vp_s_dic[seq_id]
        vp_e = vp_e_dic[seq_id]
        l_vp = vp_e - vp_s + 1
        # If fixed vp length given, only store sites with this vp length.
        if fix_vp_len:
            if not l_vp == fix_vp_len:
                continue
        # Construct the sequence graph.
        g = nx.Graph()
        g.graph["id"] = seq_id
        l_seq = len(seq)
        # Subsequence extraction start + end (1-based).
        ex_s = vp_s
        ex_e = vp_e
        # If base pair probabilities given, adjust extraction s+e.
        if bpp_dic:
            if not seq_id in bpp_dic:
                print ("ERROR: seq_id \"%s\" not in bpp_dic" % (seq_id))
                sys.exit()
            ex_s = ex_s-vp_lr_ext
            if ex_s < 1:
                ex_s = 1
            ex_e = ex_e+vp_lr_ext
            if ex_e > l_seq:
                ex_e = l_seq
        # Check up_dic.
        if up_dic:
            if not seq_id in up_dic:
                print ("ERROR: seq_id \"%s\" not in up_dic" % (seq_id))
                sys.exit()
            if len(up_dic[seq_id]) != l_seq:
                print ("ERROR: up_dic[seq_id] length != sequence length for seq_id \"%s\"" % (seq_id))
                sys.exit()
        # Check str_elem_up_dic:
        if str_elem_up_dic:
            if not seq_id in str_elem_up_dic:
                print ("ERROR: seq_id \"%s\" not in str_elem_up_dic" % (seq_id))
                sys.exit()
            if len(str_elem_up_dic[seq_id][0]) != l_seq:
                print ("ERROR: str_elem_up_dic[seq_id] length != sequence length for seq_id \"%s\"" % (seq_id))
                sys.exit()
        # Check con_dic.
        if con_dic:
            if not seq_id in con_dic:
                print ("ERROR: seq_id \"%s\" not in con_dic" % (seq_id))
                sys.exit()
            if len(con_dic[seq_id][0]) != l_seq:
                print ("ERROR: con_dic[seq_id] length != sequence length for seq_id \"%s\"" % (seq_id))
                sys.exit()
        # Check region_labels_dic.
        if region_labels_dic:
            if not seq_id in region_labels_dic:
                print ("ERROR: seq_id \"%s\" not in region_labels_dic" % (seq_id))
                sys.exit()
        # Add feature values per position.
        g_i = 0
        seen_vp = False
        for i,c in enumerate(seq): # i from 0.. l-1
            # Skip if outside region of interest.
            if i < (ex_s-1) or i > (ex_e-1):
                continue
            # Label upstream / downstream nucleotides differently.
            if use_us_ds_labels:
                lc = ["a", "c", "g", "u"]
                uc = ["A", "C", "G", "U"]
                new_c = c
                if c in lc:
                    if seen_vp:
                        new_c = "d"+c
                    else:
                        new_c = "u"+c
                else:
                    seen_vp = True
                g.add_node(g_i, label=new_c)
            else:
                # Add nucleotide node.
                g.add_node(g_i, label=c) # zero-based graph node index.
            # Make feature vector for each graph node.
            if up_dic or str_elem_up_dic or con_dic or region_labels_dic:
                feat_vector = []
                if add_1h_to_g:
                    feat_vector = char_vectorizer(c)
                if up_dic:
                    if not str_elem_up_dic:
                        feat_vector.append(up_dic[seq_id][i])
                if str_elem_up_dic:
                    # Structural elements unpaired probabilities.
                    p_e = str_elem_up_dic[seq_id][0][i]
                    p_h = str_elem_up_dic[seq_id][1][i]
                    p_i = str_elem_up_dic[seq_id][2][i]
                    p_m = str_elem_up_dic[seq_id][3][i]
                    p_s = str_elem_up_dic[seq_id][4][i]
                    # Unpaired probabilities as one-hot (max_prob_elem=1, else 0).
                    if use_str_elem_1h:
                        str_c = "E"
                        p_max = p_e
                        if p_h > p_max:
                            str_c = "H"
                            p_max = p_h
                        if p_i > p_max:
                            str_c = "I"
                            p_max = p_i
                        if p_m > p_max:
                            str_c = "M"
                            p_max = p_m
                        if p_s > p_max:
                            str_c = "S"
                        str_c_1h = char_vectorizer(str_c, 
                                       custom_alphabet = ["E", "H", "I", "M", "S"])
                        for v in str_c_1h:
                            feat_vector.append(v)
                    else:
                        # Add probabilities to vector.
                        feat_vector.append(p_e) # E
                        feat_vector.append(p_h) # H
                        feat_vector.append(p_i) # I
                        feat_vector.append(p_m) # M
                        feat_vector.append(p_s) # S
                # Conservation scores.
                if con_dic:
                    feat_vector.append(con_dic[seq_id][0][i])
                    feat_vector.append(con_dic[seq_id][1][i])
                # Region labels (exon intron).
                if region_labels_dic:
                    label = region_labels_dic[seq_id][i]
                    label_1h = char_vectorizer(label,
                                               custom_alphabet = ["E", "I"])
                    for v in label_1h:
                        feat_vector.append(v)
                g.node[g_i]['feat_vector'] = feat_vector
            # Add backbone edge.
            if g_i > 0:
                g.add_edge(g_i-1, g_i, label = '-',type='backbone')
            # Increment graph connode index.
            g_i += 1
        # Add base pair edges to graph.
        if bpp_dic:
            for entry in bpp_dic[seq_id]:
                m = re.search("(\d+)-(\d+),(.+)", entry)
                p1 = int(m.group(1))
                p2 = int(m.group(2))
                bpp_value = float(m.group(3))
                g_p1 = p1 - ex_s # 0-based base pair p1.
                g_p2 = p2 - ex_s # 0-based base pair p2.
                # Filter.
                if bpp_value < plfold_bpp_cutoff: continue
                # Add edge if bpp value >= threshold.
                if ext_mode == 1:
                    if p1 >= ex_s and p2 <= ex_e:
                        g.add_edge(g_p1, g_p2, label='=', bpp=bpp_value, type='basepair')
                elif ext_mode == 2:
                    if (p1 >= ex_s and p2 <= vp_e and p2 >= vp_s) or (p2 <= ex_e and p1 <= vp_e and p1 >= vp_s):
                        g.add_edge(g_p1, g_p2, label='=', bpp=bpp_value, type='basepair')
                elif ext_mode == 3:
                    if p1 >= vp_s and p2 <= vp_e:
                        g.add_edge(g_p1, g_p2, label='=', bpp=bpp_value, type='basepair')
                else:
                    print ("ERROR: invalid ext_mode given (valid values: 1,2,3)")
                    sys.exit()
        # Append graph to list.
        g_list.append(g)
    return g_list


################################################################################

def convert_graph_to_string(g):
    """
    Convert graph into string of graph edges for string comparisons.
    E.g. "1-2,2-3,3-5,,4-5,4-8,5-6,6-7,7-8,"
    This graph has backbone edges from 1 to 8 plus one basepair edge from 4-8.

    Expected output:
Oc:clodxdl'.,;:cclooddxxkkkkkOOOOOOOOOOOOOkkkkO00O00000000O0KNNNNNNNNNNNNNNNNNNN
Kl:cloddo;.';;:cloodddxxkkkOOOOOOOO00OOOOOkkkOO00000000000000KNNNNNNNNNNNNNNNNNN
Xd:clool:..,;:clloodddxxkkOOOOOOOO000OOO0OOkkkO000000000000000KNNNNNNNNNNNNNNNNN
NOccclc:'.',;::cllooddxxkkkOOOOOOOO00O0000OOOOO0000000000000000KNNNNNNNNNNNNNNNN
WKoccc:,..,;;;::ccllooddxxkkOOOOOOOOOO00000OOO000000000000OkxxdxOXNNNNNNNNNNNNNN
NNkcc:;'.',........',,;:cclodxkkkxkkkkkkkkkxxxxxxddoolcc;,'.....'oXNNNNNNNNNNNNN
NN0o::;'..               ...........................        ...'';kNNNNNNNNNNNNN
NNNxc:;'.  ..',,;;;,,'...                                .,:ldkOOOOKNNNNNNNNNNNN
NNNKo:;'.';:cloodxxkkkkkxol:,'.                       .;lx00KKKKKKKKKXNNNNNNNNNN
NNNNkc:;;:::clooddxxkkOOO0000Oxoc,.          .  ..  .cx0KKKKKKKKKKXKKKXNNNNNNNNN
NNNNKdcc:;::cclloodxxxkkOOO0000KK0xl'.          . .;xO0KKKKKKKKKKKKKKKKXNNNNNNNN
NNNNNOo:;;;;::cclooddxxkkkOOO000KKKK0d,.         .cxO00KKKKKKKKKKKKKK00KNNNNNNNN
NNNNNXx:;;;;;::ccloooddxxkkkOO000KKKKK0l,.      .:xkO00KKKKKKKKKKKK00000XNNNNNNN
NNNNNXd;;;;;;:::clloooddxxxkkOOO00KKKKK00d.    .;dxkO00KKKKKKKKK00000000KXNNNNNN
NNNNNKl;;;;;;:::cclloooddxxxkkkOO000KKKKK0o.   'cdxkOO0000000000000000000XNNNNNN
NNNNNO:;;;;;;:::ccllloodddxxxkkkOO000KKKKK0:  .;ldxkkOOO00000000000000000KXNNNNN
NNNNXd;;;;;;;;::cclllooodddxxxkkkOOO0000K00d..':lodxkkkOOO00000OOOOO0OO000XNNNNN
NNNN0l;;;;;;;;::ccclllooodddxxxkkkOOO000000k,.;:loddxxkkkOOOOOOOOOOOOOOOO0KNNNNN
NNNNx:;;;;;;;;::cccllloooddddxxxkkkOOOOO000k:';:clodddxxkkkOOOOOOOOOOOOOOO0XNNNN
NNNKl;;;;;;;;;:::ccllllooodddxxxxkkkkkOOOOOx:';:ccloodddxxkkOOOOOOOOOOOOOO0KNNNN
NNNk:;;;;;;;;;:::cclllloooddddxxxxxkkkkkkkkd;';::ccllooddxxkkkOOOOOOOOOOOOOKNNNN
NWKo;;;;;;;;;;:::ccllllooooddddxxxxxkkkkkxxc'',;::ccllooddxxkkkOOOOOOOOOOOO0XNNN
NN0c;;;;;;;;;;:::cclllllooooddddxxxxxxxxxxo,.',,;::ccllooddxxkkkOOOOOOOOOOO0KNNN
NNk:;;;;;;;;;;;:::ccllllloooooddddxxdddddo;..'',,;::ccclooddxxxkkkkkkOkkOOOO0XNN
WXd;;;;;::::::::::ccclllllooooodddddddooodl;''',,;;:::cclooddxxxkkkkkkkkOOOO0XNN
WKo;;;;:::::::::::cccclllloooooooooooloxOKN0c,,,,;;;:::cclloodddxxxkkkkkOOOOOKNN
W0l;;;;:::ccccccccccccclllllllllllllox0XNNWXo'',;;;;;;::cccllooddxxkkkkkOOOOO0XW
W0c;;;;:::ccccccccclllllllllllllcc:oKNNNNNWXc.....'',;;;:::ccloodxxxkkkOOOOOO0XN
W0c;;;;:::ccccllllllllooooooollccc:dXWNNNNWK:........'',;;:clloddxxkkkkOOOOOOOKN
WKl;;;;:::ccllllloooooddddddddddddoxKWNNNNWKc...'',,,;;::cllooddxxkkkkOOOOOOOOKN
WKo;;;;;::ccllllooodddddxxxxxxxxxxdxKWNWNNWNo'''',;::cccclloooddxxkkkkOOOOOOOOKN
WXd;;;;;::cclllloooddddxxxxxxxxxxxdkXWNWWWWWO:'',,;:ccllllooodddxxkkkkkOOOOOOOKW

    """
    g_string = ""
    for s,e in g.edges:
        g_string += "%i-%i," % (s,e)
    return g_string


################################################################################

def convert_seqs_to_bppms(seqs_dic, vp_s, vp_e, bpp_dic,
                          plfold_bpp_cutoff=0.2,
                          vp_size=101,
                          vp_lr_ext=100):
    """
    Convert sequences to base pair probability matrices, given base pair 
    information in bpp_dic (key: sequence id, value: list of base pair 
    strings in format: "bp_start-bp_end,bp_prob")
    vp_lr_ext sets sequence region to use for base pair probability matrix.
    len(vp_lr_ext) upstream context + viewpoint region + len(vp_lr_ext) down-
    stream context. If sequence is shorter than context extensions, 
    still generate same size matrix, filling up missing regions with "0" 
    entries. Resulting matrix is symmetric.
    Return list of base pair probability matrices (2d lists).

    >>> seqs_dic = {"CLIP_01" : "acguACGUacgu"}
    >>> id1_bp = ["3-10,0.3", "4-9,0.5", "5-8,0.4"]
    >>> bpp_dic = {"CLIP_01" : id1_bp}
    >>> vp_s = {"CLIP_01": 5}
    >>> vp_e = {"CLIP_01": 8}
    >>> bppms_list = convert_seqs_to_bppms(seqs_dic, vp_s, vp_e, bpp_dic, vp_size=4, vp_lr_ext=2)
    >>> print(bppms_list[0])
    [[0.  0.  0.  0.  0.  0.  0.  0.3]
     [0.  0.  0.  0.  0.  0.  0.5 0. ]
     [0.  0.  0.  0.  0.  0.4 0.  0. ]
     [0.  0.  0.  0.  0.  0.  0.  0. ]
     [0.  0.  0.  0.  0.  0.  0.  0. ]
     [0.  0.  0.4 0.  0.  0.  0.  0. ]
     [0.  0.5 0.  0.  0.  0.  0.  0. ]
     [0.3 0.  0.  0.  0.  0.  0.  0. ]]

    """
    bppms_list = []
    # Fixed matrix size, defined by vp_size + vp_lr_ext.
    m_size = vp_size + vp_lr_ext*2
    for seq_id, seq in sorted(seqs_dic.items()):
        l_seq = len(seq)
        # Define region with base pair information.
        ex_s = vp_s[seq_id]-vp_lr_ext
        if ex_s < 1:
            ex_s = 1
        ex_e = vp_e[seq_id]+vp_lr_ext
        if ex_e > l_seq:
            ex_e = l_seq
        # Size of actual region (can be smaller than m_size in case of shortened context).
        act_size = ex_e - ex_s + 1
        # Create 2d numpy array of size m_size, filled with zeros (floats).
        bppm = np.zeros((m_size, m_size))
        # Add base pair information.
        if not seq_id in bpp_dic:
            print ("ERROR: seq_id \"%s\" not in bpp_dic" % (seq_id))
            sys.exit()
        for entry in bpp_dic[seq_id]:
            m = re.search("(\d+)-(\d+),(.+)", entry)
            p1 = int(m.group(1))
            p2 = int(m.group(2))
            bpp_value = float(m.group(3))
            m_p1 = p1 - ex_s # 0-based base pair pos1 in matrix.
            m_p2 = p2 - ex_s # 0-based base pair pos2 in matrix.
            if m_p1 < 0: continue # if position outside matrix.
            if m_p2 >= m_size: continue # if index outside matrix.
            # Add bpp value to matrix if >= threshold.
            if bpp_value >= plfold_bpp_cutoff:
                # access an index: bppm.item(i,j)
                # set an index: bppm[i,j] = x
                bppm[m_p1,m_p2] = bpp_value
                bppm[m_p2,m_p1] = bpp_value
        bppms_list.append(bppm)
    return bppms_list


################################################################################

def convert_seqs_to_one_hot(seqs_dic, vp_s_dic, vp_e_dic,
                            seqs_list_1h=False,
                            fix_vp_len=False,
                            up_dic=False,
                            region_labels_dic=False,
                            use_str_elem_1h=False,
                            str_elem_up_dic=False,
                            con_dic=False):
    """
    Convert sequence dictionary in list of one-hot encoded sequences.
    Each dictionary element (sequence id = key) contains 2d list of one-hot 
    encoded nucleotides. In addition, if up_dic given, add unpaired 
    probability vector.

    >>> seqs_dic = {"CLIP_01" : "guAUCGgu"}
    >>> up_dic = {"CLIP_01" : [0.5]*8}
    >>> con_dic = {"CLIP_01" : [[0.4]*8,[0.6]*8]}
    >>> str_elem_up_dic = {"CLIP_01" : [[0.1]*8,[0.2]*8,[0.3]*8,[0.2]*8,[0.2]*8]}
    >>> vp_s = {"CLIP_01": 3}
    >>> vp_e = {"CLIP_01": 6}
    >>> seqs_list_1h = convert_seqs_to_one_hot(seqs_dic, vp_s, vp_e)
    >>> print(seqs_list_1h[0])
    [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]
    >>> seqs_list_1h = convert_seqs_to_one_hot(seqs_dic, vp_s, vp_e, up_dic=up_dic)
    >>> print(seqs_list_1h[0])
    [[1, 0, 0, 0, 0.5], [0, 0, 0, 1, 0.5], [0, 1, 0, 0, 0.5], [0, 0, 1, 0, 0.5]]
    >>> seqs_list_1h = convert_seqs_to_one_hot(seqs_dic, vp_s, vp_e, con_dic=con_dic)
    >>> print(seqs_list_1h[0])
    [[1, 0, 0, 0, 0.4, 0.6], [0, 0, 0, 1, 0.4, 0.6], [0, 1, 0, 0, 0.4, 0.6], [0, 0, 1, 0, 0.4, 0.6]]
    >>> seqs_list_1h = convert_seqs_to_one_hot(seqs_dic, vp_s, vp_e, str_elem_up_dic=str_elem_up_dic)
    >>> print(seqs_list_1h[0])
    [[1, 0, 0, 0, 0.1, 0.2, 0.3, 0.2, 0.2], [0, 0, 0, 1, 0.1, 0.2, 0.3, 0.2, 0.2], [0, 1, 0, 0, 0.1, 0.2, 0.3, 0.2, 0.2], [0, 0, 1, 0, 0.1, 0.2, 0.3, 0.2, 0.2]]
    >>> seqs_dic = {"CLIP_01" : "CG"}
    >>> vp_s = {"CLIP_01": 1}
    >>> vp_e = {"CLIP_01": 2}
    >>> region_labels_dic = {"CLIP_01" : ['E', 'I']}
    >>> str_elem_up_dic = {'CLIP_01': [[0.1, 0.2], [0.2, 0.3], [0.4, 0.2], [0.2, 0.1], [0.1, 0.2]]}
    >>> seqs_list_1h = convert_seqs_to_one_hot(seqs_dic, vp_s, vp_e, str_elem_up_dic=str_elem_up_dic, region_labels_dic=region_labels_dic)
    >>> print(seqs_list_1h[0])
    [[0, 1, 0, 0, 0.1, 0.2, 0.4, 0.2, 0.1, 1, 0], [0, 0, 1, 0, 0.2, 0.3, 0.2, 0.1, 0.2, 0, 1]]
    >>> seqs_list_1h = convert_seqs_to_one_hot(seqs_dic, vp_s, vp_e, str_elem_up_dic=str_elem_up_dic, use_str_elem_1h=True)
    >>> print(seqs_list_1h[0])
    [[0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 0]]

    """
    if not seqs_list_1h:
        seqs_list_1h = []
    for seq_id, seq in sorted(seqs_dic.items()):
        # Get viewpoint start+end of sequence.
        vp_s = vp_s_dic[seq_id]
        vp_e = vp_e_dic[seq_id]
        l_vp = vp_e - vp_s + 1
        # If fixed vp length given, only store sites with this vp length.
        if fix_vp_len:
            if not l_vp == fix_vp_len:
                continue
        # Convert sequence to one-hot-encoding.
        seq_1h = string_vectorizer(seq, vp_s, vp_e)
        # Additional feature checkings.
        if con_dic:
            if not seq_id in con_dic:
                print ("ERROR: seq_id \"%s\" not in con_dic" % (seq_id))
                sys.exit()
        if up_dic:
            if not seq_id in up_dic:
                print ("ERROR: seq_id \"%s\" not in up_dic" % (seq_id))
                sys.exit()
            if len(seq) != len(up_dic[seq_id]):
                print ("ERROR: length of unpaired probability list != sequence length (\"%s\" != \"%s\")" % (l_seq, l_up_list))
                sys.exit()
        if str_elem_up_dic:
            if not seq_id in str_elem_up_dic:
                print ("ERROR: seq_id \"%s\" not in str_elem_up_dic" % (seq_id))
                sys.exit()
        if region_labels_dic:
            if not seq_id in region_labels_dic:
                print ("ERROR: seq_id \"%s\" not in region_labels_dic" % (seq_id))
                sys.exit()
        # Add additional features to one-hot matrix.
        if con_dic or up_dic or str_elem_up_dic or region_labels_dic:
            # Start index of viewpoint region.
            i= vp_s - 1
            for row in seq_1h:
                # Unpaired probabilities of structural elements.
                if str_elem_up_dic:
                    p_e = str_elem_up_dic[seq_id][0][i]
                    p_h = str_elem_up_dic[seq_id][1][i]
                    p_i = str_elem_up_dic[seq_id][2][i]
                    p_m = str_elem_up_dic[seq_id][3][i]
                    p_s = str_elem_up_dic[seq_id][4][i]
                    # Unpaired probabilities as one-hot (max_prob_elem=1, else 0).
                    if use_str_elem_1h:
                        str_c = "E"
                        p_max = p_e
                        if p_h > p_max:
                            str_c = "H"
                            p_max = p_h
                        if p_i > p_max:
                            str_c = "I"
                            p_max = p_i
                        if p_m > p_max:
                            str_c = "M"
                            p_max = p_m
                        if p_s > p_max:
                            str_c = "S"
                        str_c_1h = char_vectorizer(str_c, 
                                       custom_alphabet = ["E", "H", "I", "M", "S"])
                        for v in str_c_1h:
                            row.append(v)
                    else:
                        row.append(p_e) # E
                        row.append(p_h) # H
                        row.append(p_i) # I
                        row.append(p_m) # M
                        row.append(p_s) # S
                    # No need to add prob for "S", since its 1-unpaired_prob
                # Unpaired probabilities (total unpaired probabilities).
                if up_dic:
                    if not str_elem_up_dic:
                        row.append(up_dic[seq_id][i])
                # Conservation scores.
                if con_dic:
                    row.append(con_dic[seq_id][0][i])
                    row.append(con_dic[seq_id][1][i])
                # Region labels (exon intron).
                if region_labels_dic:
                    label = region_labels_dic[seq_id][i]
                    label_1h = char_vectorizer(label, 
                                               custom_alphabet = ["E", "I"])
                    for v in label_1h:
                        row.append(v)
                i += 1
        # Append one-hot encoded input to inputs list.
        seqs_list_1h.append(seq_1h)
    return seqs_list_1h


################################################################################




