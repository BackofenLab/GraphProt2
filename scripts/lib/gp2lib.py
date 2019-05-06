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


fix_vp_len potentially leads to discarding some sequences

convert_seqs_to_one_hot
convert_seqs_to_graphs


"""

def load_data(data_folder, 
              use_up=False,
              use_con=False,
              use_entr=False,
              use_str_elem_up=False,
              use_sf=False,
              disable_bpp=False,
              bpp_cutoff=0.2,
              bpp_mode=1,
              vp_ext = 100,
              fix_vp_len=True):
    """
    Load data from data_folder.
    Return list of structure graphs, one-hot encoded sequences np array, 
    and label vector.
    
    Function parameters:
        use_up : if true add unpaired probabilities to graph and one-hot
        use_con : if true add conservation scores to one-hot
        use_str_elem_up: add str elements unpaired probs to one-hot
        use_sf: add site features, store in additional vector for each sequence
        use_entr: use RBP occupancy / entropy features for each sequence
        disable_bpp : disables adding of base pair information
        bpp_cutoff : bp probability threshold when adding bp probs.
        bpp_mode : see ext_mode in convert_seqs_to_graphs for details
        vp_ext : Define upstream + downstream viewpoint extension
                 Usually set equal to used plfold_L (default: 100)
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
        pos_sf_dic = read_sf_into_dic(pos_sf_file)
        neg_sf_dic = read_sf_into_dic(neg_sf_file)
    if use_entr:
        pos_sf_dic = read_entr_into_dic(pos_entr_file)
        neg_sf_dic = read_entr_into_dic(neg_entr_file)

    # Convert input sequences to one-hot encoding (optionally with unpaired probabilities vector).
    pos_seq_1h = convert_seqs_to_one_hot(pos_seqs_dic, pos_vp_s, pos_vp_e, 
                                         up_dic=pos_up_dic,
                                         con_dic=pos_con_dic,
                                         str_elem_up_dic=pos_str_elem_up_dic)
    neg_seq_1h = convert_seqs_to_one_hot(neg_seqs_dic, neg_vp_s, neg_vp_e, 
                                         up_dic=neg_up_dic, 
                                         con_dic=neg_con_dic,
                                         str_elem_up_dic=neg_str_elem_up_dic)
    # Convert input sequences to sequence or structure graphs.
    pos_graphs = convert_seqs_to_graphs(pos_seqs_dic, pos_vp_s, pos_vp_e, 
                                        up_dic=pos_up_dic, 
                                        con_dic=pos_con_dic, 
                                        str_elem_up_dic=pos_str_elem_up_dic, 
                                        bpp_dic=pos_bpp_dic, 
                                        vp_lr_ext=vp_ext, 
                                        ext_mode=bpp_mode,
                                        plfold_bpp_cutoff=bpp_cutoff)
    neg_graphs = convert_seqs_to_graphs(neg_seqs_dic, neg_vp_s, neg_vp_e, 
                                        up_dic=neg_up_dic, 
                                        con_dic=neg_con_dic, 
                                        str_elem_up_dic=neg_str_elem_up_dic, 
                                        bpp_dic=neg_bpp_dic, 
                                        vp_lr_ext=vp_ext,  
                                        ext_mode=bpp_mode,
                                        plfold_bpp_cutoff=bpp_cutoff)
    # Create labels.
    labels = [1]*len(pos_seq_1h) + [0]*len(neg_seq_1h)
    # Concatenate pos+neg one-hot lists.
    seq_1h = pos_seq_1h + neg_seq_1h
    # Convert 1h list to np array, transpose matrices and make each entry 3d (1,number_of_features,vp_length).
    new_seq_1h = []
    for idx in range(len(seq_1h)):
        M = np.array(seq_1h[idx]).transpose()
        M = np.reshape(M, (1, M.shape[0], M.shape[1]))
        new_seq_1h.append(M)
    # Concatenate pos+neg graph lists.
    graphs = pos_graphs + neg_graphs
    # From site feature dictionaries to list of site feature vectors.
    site_feat_v = []
    for site_id, site_v in sorted(pos_sf_dic.items()):
        if site_id in pos_seqs_dic:
            site_feat_v.append(site_v)
    for site_id, site_v in sorted(neg_sf_dic.items()):
        if site_id in neg_seqs_dic:
            site_feat_v.append(site_v)
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

def string_vectorizer(seq, 
                      s=None,
                      e=None):
    """
    Take string sequence, look at each letter and convert to one-hot-encoded
    vector. Optionally define start and end index (1-based) for extracting 
    sub-sequences.
    Return array of one-hot encoded vectors.

    >>> string_vectorizer("ACGU")
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    >>> string_vectorizer("")
    []
    >>> string_vectorizer("XX")
    [[0, 0, 0, 0], [0, 0, 0, 0]]

    """
    alphabet=['A','C','G','U']
    seq_l = len(seq)
    vector = [[1 if char == letter else 0 for char in alphabet] for letter in seq]
    if s and e:
        if len(seq) < e or s < 1:
            print ("ERROR: invalid indices passed to string_vectorizer (s:\"%s\", e:\"%s\")" % (s, e))
            sys.exit()
        vector = vector[s-1:e]
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

def extract_viewpoint_regions_from_fasta(seqs_dic):
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
    vp_s_dic = {}
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

def read_str_elem_up_into_dic(str_elem_up_file):

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
    {'CLIP_01': [[0.1, 0.2], [0.2, 0.3], [0.3, 0.2], [0.3, 0.1]]}

    """
    str_elem_up_dic = {}
    seq_id = ""
    # Go through .str_elem.up file, extract p_external, p_hairpin, p_internal, p_multiloop.
    with open(str_elem_up_file) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                str_elem_up_dic[seq_id] = [[],[],[],[]]
            else:
                m = re.search("\d+\t.+?\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t", line)
                p_external = float(m.group(1))
                p_hairpin = float(m.group(2))
                p_internal = float(m.group(3))
                p_multiloop = float(m.group(4))
                str_elem_up_dic[seq_id][0].append(p_external)
                str_elem_up_dic[seq_id][1].append(p_hairpin)
                str_elem_up_dic[seq_id][2].append(p_internal)
                str_elem_up_dic[seq_id][3].append(p_multiloop)
    f.closed
    return str_elem_up_dic



################################################################################

def read_entr_into_dic(entr_file):

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
    entr_dic = {}
    seq_id = ""
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
            for i in f_list:
                entr_dic[seq_id].append(float(i))
    f.closed
    return entr_dic



################################################################################

def read_sf_into_dic(sf_file):

    """
    Read site features into dictionary.
    (key: sequence id, value: vector of site feature values)

    >>> sf_test = "test_data/test.sf"
    >>> read_sf_into_dic(sf_test)
    {'CLIP_01': [0.2, 0.3, 0.3, 0.2], 'CLIP_02': [0.1, 0.2, 0.4, 0.3]}

    """
    sf_dic = {}
    seq_id = ""
    # Go through .up file, extract unpaired probs for each position.
    with open(sf_file) as f:
        for line in f:
            f_list = line.strip().split("\t")
            # Skip header line(s).
            if f_list[0] == "id":
                continue
            seq_id = f_list[0]
            f_list.pop(0)
            if not seq_id in sf_dic:
                sf_dic[seq_id] = []
            for i in f_list:
                sf_dic[seq_id].append(float(i))
    f.closed
    return sf_dic



################################################################################

def read_up_into_dic(up_file):

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

def read_con_into_dic(con_file):

    """
    Read in conservation scores (phastCons+phyloP) and store scores as 
    2xn matrix for sequence with length n. 
    Return dictionary with matrix for each sequence
    (key: sequence id, value: scores matrix)
    Entry format: [[1,2,3],[4,5,6]] : 2x3 format (2 rows, 3 columns)

    >>> con_test = "test_data/test.con"
    >>> read_con_into_dic(con_test)
    {'CLIP_01': [[0.1, 0.2], [0.3, -0.4]], 'CLIP_02': [[0.4, 0.5], [0.6, 0.7]]}

    """
    con_dic = {}
    seq_id = ""
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
    f.closed
    return con_dic


################################################################################

def read_bpp_into_dic(bpp_file, vp_s, vp_e,
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

def convert_seqs_to_graphs(seqs_dic, vp_s_dic, vp_e_dic, 
                           up_dic=False,
                           bpp_dic=False,
                           con_dic=False,
                           str_elem_up_dic=False,
                           plfold_bpp_cutoff=0.2,
                           vp_lr_ext=100, 
                           fix_vp_len=False,
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
    >>> seqs_dic2 = {"CLIP_01" : "aCGt"}
    >>> up_dic2 = {"CLIP_01" : [0.8]*4}
    >>> vp_s2 = {"CLIP_01" : 2}
    >>> vp_e2 = {"CLIP_01" : 3}
    >>> con_dic2 = {"CLIP_01" : [[0.3, 0.5, 0.7, 0.2], [0.2, 0.8, 0.9, 0.1]]}
    >>> str_elem_up_dic2 = {"CLIP_01": [[0.1]*4, [0.3]*4, [0.4]*4, [0.2]*4]}
    >>> g_list2 = convert_seqs_to_graphs(seqs_dic2, vp_s2, vp_e2, up_dic=up_dic2, con_dic=con_dic2, str_elem_up_dic=str_elem_up_dic2)
    >>> g_list2[0].node[0]['feat_vector']
    [0.8, 0.1, 0.3, 0.4, 0.2, 0.5, 0.8]
    >>> g_list2[0].node[1]['feat_vector']
    [0.8, 0.1, 0.3, 0.4, 0.2, 0.7, 0.9]

    """
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
        # Add feature values per position.
        g_i = 0
        for i,c in enumerate(seq): # i from 0.. l-1
            # Skip if outside region of interest.
            if i < (ex_s-1) or i > (ex_e-1):
                continue
            # Add nucleotide node.
            g.add_node(g_i, label=c) # zero-based graph node index.
            # Make feature vector for each graph node.
            if up_dic or str_elem_up_dic or con_dic:
                feat_vector = []
                if up_dic:
                    feat_vector.append(up_dic[seq_id][i])
                if str_elem_up_dic:
                    feat_vector.append(str_elem_up_dic[seq_id][0][i])
                    feat_vector.append(str_elem_up_dic[seq_id][1][i])
                    feat_vector.append(str_elem_up_dic[seq_id][2][i])
                    feat_vector.append(str_elem_up_dic[seq_id][3][i])
                if con_dic:
                    feat_vector.append(con_dic[seq_id][0][i])
                    feat_vector.append(con_dic[seq_id][1][i])
                g.node[g_i]['feat_vector'] = feat_vector
            # Add backbone edge.
            if g_i > 0:
                g.add_edge(g_i-1, g_i, label = '-',type='backbone')
            # Increment graph node index.
            g_i += 1
        # Add base pair edges to graph.
        if bpp_dic:
            for entry in bpp_dic[seq_id]:
                m = re.search("(\d+)-(\d+),(.+)", entry)
                p1 = int(m.group(1))
                p2 = int(m.group(2))
                bpp_value = float(m.group(3))
                g_p1 = p1 - ex_s # 0-based base pair pos1.
                g_p2 = p2 - ex_s # 0-based base pair pos2.
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
                            fix_vp_len=False,
                            up_dic=False,
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
    >>> str_elem_up_dic = {"CLIP_01" : [[0.1]*8,[0.2]*8,[0.3]*8,[0.2]*8]}
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
    [[1, 0, 0, 0, 0.1, 0.2, 0.3, 0.2], [0, 0, 0, 1, 0.1, 0.2, 0.3, 0.2], [0, 1, 0, 0, 0.1, 0.2, 0.3, 0.2], [0, 0, 1, 0, 0.1, 0.2, 0.3, 0.2]]

    """
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
        # If unpaired probabilities given, add to matrix.
        if up_dic:
            if not seq_id in up_dic:
                print ("ERROR: seq_id \"%s\" not in up_dic" % (seq_id))
                sys.exit()
            l_seq = len(seq)
            l_up_list = len(up_dic[seq_id])
            if l_seq != l_up_list:
                print ("ERROR: length of unpaired probability list != sequence length (\"%s\" != \"%s\")" % (l_seq, l_up_list))
                sys.exit()
            # Start index of up list.
            i= vp_s - 1
            # Add unpaired probabilities to one-hot matrix (add row).
            for row in seq_1h:
                row.append(up_dic[seq_id][i])
                i += 1
        # If conservation scores (phastCons, phyloP) given, add to matrix.
        if con_dic:
            if not seq_id in con_dic:
                print ("ERROR: seq_id \"%s\" not in con_dic" % (seq_id))
                sys.exit()
            # Start index of up list.
            i= vp_s - 1
            # Add unpaired probabilities to one-hot matrix (add row).
            for row in seq_1h:
                row.append(con_dic[seq_id][0][i])
                row.append(con_dic[seq_id][1][i])
                i += 1
        # If str elements unpaired probs given, add to matrix.
        if str_elem_up_dic:
            if not seq_id in str_elem_up_dic:
                print ("ERROR: seq_id \"%s\" not in str_elem_up_dic" % (seq_id))
                sys.exit()
            # Start index of up list.
            i= vp_s - 1
            # Add unpaired probabilities to one-hot matrix (add row).
            for row in seq_1h:
                row.append(str_elem_up_dic[seq_id][0][i])
                row.append(str_elem_up_dic[seq_id][1][i])
                row.append(str_elem_up_dic[seq_id][2][i])
                row.append(str_elem_up_dic[seq_id][3][i])
                i += 1
        # Append one-hot encoded input to inputs list.
        seqs_list_1h.append(seq_1h)
    return seqs_list_1h


################################################################################




