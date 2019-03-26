#!/usr/bin/env python3

import sys
import re
import networkx as nx
import numpy as np

"""
GraphProt2 Python3 function library

Run doctests from base directory:
python3 -m doctest -v lib/gp2lib.py

"""

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
    # Go through .up file, extract sequences.
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
                           up_dic=None,
                           bpp_dic=None,
                           plfold_bpp_cutoff=0.2,
                           vp_lr_ext=100, 
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

    """
    g_list = []
    for seq_id, seq in sorted(seqs_dic.items()):
        # Construct the sequence graph.
        g = nx.Graph()
        g.graph["id"] = seq_id
        l_seq = len(seq)
        vp_s = vp_s_dic[seq_id]
        vp_e = vp_e_dic[seq_id]
        # Subsequence extraction start + end (1-based).
        ex_s = vp_s
        ex_e = vp_e
        # If base pair probabilities given, adjust extraction s+e.
        if bpp_dic:
            ex_s = ex_s-vp_lr_ext
            if ex_s < 1:
                ex_s = 1
            ex_e = ex_e+vp_lr_ext
            if ex_e > l_seq:
                ex_e = l_seq
        g_i = 0
        for i,c in enumerate(seq): # i from 0.. l-1
            # Skip if outside region of interest.
            if i < (ex_s-1) or i > (ex_e-1):
                continue
            # Add nucleotide node.
            g.add_node(g_i, label=c) # zero-based graph node index.
            # Add unpaired probability attribute.
            if up_dic:
                if not seq_id in up_dic:
                    print ("ERROR: seq_id \"%s\" not in up_dic" % (seq_id))
                    sys.exit()
                if len(up_dic[seq_id]) != l_seq:
                    print ("ERROR: up_dic[seq_id] length != sequence length for seq_id \"%s\"" % (seq_id))
                    sys.exit()
                g.node[g_i]['up'] = up_dic[seq_id][i]
            # Add backbone edge.
            if g_i > 0:
                g.add_edge(g_i-1, g_i, label = '-',type='backbone')
            # Increment graph node index.
            g_i += 1
        # Add base pair edges to graph.
        if bpp_dic:
            if not seq_id in bpp_dic:
                print ("ERROR: seq_id \"%s\" not in bpp_dic" % (seq_id))
                sys.exit()
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

def convert_seqs_to_one_hot(seqs_dic, vp_s, vp_e,
                            up_dic=False):
    """
    Convert sequence dictionary in list of one-hot encoded sequences.
    Each dictionary element (sequence id = key) contains 2d list of one-hot 
    encoded nucleotides. In addition, if up_dic given, add unpaired 
    probability vector.

    >>> seqs_dic = {"CLIP_01" : "guAUCGgu"}
    >>> up_dic = {"CLIP_01" : [0.5]*8}
    >>> vp_s = {"CLIP_01": 3}
    >>> vp_e = {"CLIP_01": 6}
    >>> seqs_list_1h = convert_seqs_to_one_hot(seqs_dic, vp_s, vp_e)
    >>> print(seqs_list_1h[0])
    [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]
    >>> seqs_list_1h = convert_seqs_to_one_hot(seqs_dic, vp_s, vp_e, up_dic=up_dic)
    >>> print(seqs_list_1h[0])
    [[1, 0, 0, 0, 0.5], [0, 0, 0, 1, 0.5], [0, 1, 0, 0, 0.5], [0, 0, 1, 0, 0.5]]

    """
    seqs_list_1h = []
    for seq_id, seq in sorted(seqs_dic.items()):
        # Convert sequence to one-hot-encoding.
        seq_1h = string_vectorizer(seq, vp_s[seq_id], vp_e[seq_id])
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
            i= vp_s[seq_id] - 1
            # Add unpaired probabilities to one-hot matrix (add row).
            for row in seq_1h:
                row.append(up_dic[seq_id][i])
                i += 1
        # Append one-hot encoded input to inputs list.
        seqs_list_1h.append(seq_1h)
    return seqs_list_1h


################################################################################




