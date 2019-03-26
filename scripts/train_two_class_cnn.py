#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
from lib import gp2lib


def main():
    # Setup argparse and define available command line arguments.
    parser = setup_argument_parser()
    # Read the command line arguments.
    args = parser.parse_args()
    # Check command line input.
    check_input(args)
    # Input files.
    pos_fasta_file = "%s/positives.fa" % (args.data_in)
    neg_fasta_file = "%s/negatives.fa" % (args.data_in)
    pos_up_file = "%s/positives.up" % (args.data_in)
    neg_up_file = "%s/negatives.up" % (args.data_in)
    pos_bpp_file = "%s/positives.bpp" % (args.data_in)
    neg_bpp_file = "%s/negatives.bpp" % (args.data_in)
    pos_con_file = "%s/positives.con" % (args.data_in)
    neg_con_file = "%s/negatives.con" % (args.data_in)
    pos_entropy_file = "%s/positives.ent" % (args.data_in)
    neg_entropy_file = "%s/negatives.ent" % (args.data_in)

    # Read in FASTA sequences.
    pos_seqs_dic = gp2lib.read_fasta_into_dic(pos_fasta_file)
    neg_seqs_dic = gp2lib.read_fasta_into_dic(neg_fasta_file)
    # Get viewpoint regions.
    pos_vp_s, pos_vp_e = gp2lib.extract_viewpoint_regions_from_fasta(pos_seqs_dic)
    neg_vp_s, neg_vp_e = gp2lib.extract_viewpoint_regions_from_fasta(neg_seqs_dic)
    # upstream downstream viewpoint extension.
    # normally set equal to used plfold_L (to capture all base pairs rooting in viewpoint).
    vp_ext = 100
    # Extract most prominent viewpoint length from data.
    max_vp_l = 0
    for seq_id in pos_seqs_dic:
        vp_l = pos_vp_e[seq_id] - pos_vp_s[seq_id] + 1  # +1 since 1-based.
        if vp_l > max_vp_l:
            max_vp_l = vp_l
    if not max_vp_l:
        print("ERROR: viewpoint length extraction failed")
        sys.exit()

    # Init dictionaries.
    pos_up_dic = False
    neg_up_dic = False
    pos_bpp_dic = False
    neg_bpp_dic = False
    pos_con_dic = False
    neg_con_dic = False
    pos_entropy_dic = False
    neg_entropy_dic = False

    # Extract additional annotations.
    if args.use_up:
        print("Read in .up files ... ") 
        pos_up_dic = gp2lib.read_up_into_dic(pos_up_file)
        neg_up_dic = gp2lib.read_up_into_dic(neg_up_file)
    if args.use_bpp:
        print("Read in .bpp files ... ")
        pos_bpp_dic = gp2lib.read_bpp_into_dic(pos_bpp_file, pos_vp_s, pos_vp_e, vp_lr_ext=vp_ext)
        neg_bpp_dic = gp2lib.read_bpp_into_dic(neg_bpp_file, neg_vp_s, neg_vp_e, vp_lr_ext=vp_ext)
    if args.use_con:
        pos_con_dic = gp2lib.read_con_into_dic(pos_con_file, pos_vp_s, pos_vp_e)
        neg_con_dic = gp2lib.read_con_into_dic(neg_con_file, neg_vp_s, neg_vp_e)
    if args.use_entropy:
        pos_entropy_dic = gp2lib.read_entropy_into_dic(pos_entropy_file, pos_vp_s, pos_vp_e)
        neg_entropy_dic = gp2lib.read_entropy_into_dic(neg_entropy_file, neg_vp_s, neg_vp_e)

    # Convert input sequences to one-hot encoding (optionally with unpaired probabilities vector).
    print("Convert sequences to one-hot ... ")
    pos_seq_1h = gp2lib.convert_seqs_to_one_hot(pos_seqs_dic, pos_vp_s, pos_vp_e, up_dic=pos_up_dic)
    neg_seq_1h = gp2lib.convert_seqs_to_one_hot(neg_seqs_dic, neg_vp_s, neg_vp_e, up_dic=neg_up_dic)
    # Convert input sequences to sequence or structure graphs.
    print("Convert sequences to graphs ... ")
    pos_graphs = gp2lib.convert_seqs_to_graphs(pos_seqs_dic, pos_vp_s, pos_vp_e, up_dic=pos_up_dic, bpp_dic=pos_bpp_dic, vp_lr_ext=vp_ext)
    neg_graphs = gp2lib.convert_seqs_to_graphs(neg_seqs_dic, neg_vp_s, neg_vp_e, up_dic=neg_up_dic, bpp_dic=neg_bpp_dic, vp_lr_ext=vp_ext)
    # Convert input sequences to base pair probability matrices.
    pos_bppms = False
    neg_bppms = False
    print("Convert sequences to base pair probability matrices ... ")
    if pos_bpp_dic:
        pos_bppms = gp2lib.convert_seqs_to_bppms(pos_seqs_dic, pos_vp_s, pos_vp_e, pos_bpp_dic, vp_size=max_vp_l, vp_lr_ext=vp_ext, plfold_bpp_cutoff=args.bpp_cutoff)
    if neg_bpp_dic:
        neg_bppms = gp2lib.convert_seqs_to_bppms(neg_seqs_dic, neg_vp_s, neg_vp_e, neg_bpp_dic, vp_size=max_vp_l, vp_lr_ext=vp_ext, plfold_bpp_cutoff=args.bpp_cutoff)
    # Convert to numpy arrays.
    # Create labels.
    # Submit to training.
    print (np.sum(pos_bppms[0]))
    

################################################################################

def setup_argument_parser():
    """
    Setup argparse and define available command line arguments.
    """

    # Help screen description.
    help_description = """
    Read in data from -in data folder, containing positive+negative binding
    sites in FASTA format, as well as additional files storing unpaired
    and base pair probability information or conservation scores.
    Implement different site encodings (one-hot, graph, sequence+annotations ..)
    and submit to CNN training (binary classification).
    """

    # Create argument parser.
    parser = argparse.ArgumentParser(add_help=False,
        prog="train_two_class_cnn.py",
        usage="%(prog)s [arguments] [options]",
        description=help_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Help.
    parser.add_argument("-h", "--help",
        action = "help",
        help = "Print this help message and exit")
    # Required.
    parser.add_argument("-in", "--data-in",
        dest="data_in",
        type=str,
        help="Data folder containing training and annotation files",
        required=True)
    # Optional.
    parser.add_argument("--use-up",
        dest="use_up",
        help="Use unpaired probability information for training sites from .up files",
        default=False,
        action="store_true")
    parser.add_argument("--use-bpp",
        dest="use_bpp",
        help="Use base pair probability information for training sites from .bpp files",
        default=False,
        action="store_true")
    parser.add_argument("--use-con",
        dest="use_con",
        help="Use conservation information for training sites",
        default=False,
        action="store_true")
    parser.add_argument("--use-entropy",
        dest="use_entropy",
        help="Use entropy information for training sites",
        default=False,
        action="store_true")
    parser.add_argument("--bpp-cutoff",
        dest="bpp_cutoff",
        help="Base pair probability cutoff for filtering base pairs from .bpp files",
        type=float,
        default=0.2)
    return parser


################################################################################

def check_input(args):
    """
    Check command line input data.
    """
    # Check input.
    if not os.path.isdir(args.data_in):
        print("INPUT_ERROR: Input data folder \"%s\" not found" % (args.data_in))
        sys.exit()
    pos_fasta_file = "%s/positives.fa" % (args.data_in)
    neg_fasta_file = "%s/negatives.fa" % (args.data_in)
    pos_up_file = "%s/positives.up" % (args.data_in)
    neg_up_file = "%s/negatives.up" % (args.data_in)
    pos_bpp_file = "%s/positives.bpp" % (args.data_in)
    neg_bpp_file = "%s/negatives.bpp" % (args.data_in)
    pos_con_file = "%s/positives.con" % (args.data_in)
    neg_con_file = "%s/negatives.con" % (args.data_in)
    pos_entropy_file = "%s/positives.ent" % (args.data_in)
    neg_entropy_file = "%s/negatives.ent" % (args.data_in)
    if not os.path.isfile(pos_fasta_file):
        print("INPUT_ERROR: missing \"%s\"" % (pos_fasta_file))
        sys.exit()
    if not os.path.isfile(neg_fasta_file):
        print("INPUT_ERROR: missing \"%s\"" % (neg_fasta_file))
        sys.exit()
    if (args.use_up):
        if not os.path.isfile(pos_up_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_up_file))
            sys.exit()
        if not os.path.isfile(neg_up_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_up_file))
            sys.exit()
    if (args.use_bpp):
        if not os.path.isfile(pos_bpp_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_bpp_file))
            sys.exit()
        if not os.path.isfile(neg_bpp_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_bpp_file))
            sys.exit()
    if (args.use_con):
        if not os.path.isfile(pos_con_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_con_file))
            sys.exit()
        if not os.path.isfile(neg_con_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_con_file))
            sys.exit()
    if (args.use_entropy):
        if not os.path.isfile(pos_entropy_file):
            print("INPUT_ERROR: missing \"%s\"" % (pos_entropy_file))
            sys.exit()
        if not os.path.isfile(neg_entropy_file):
            print("INPUT_ERROR: missing \"%s\"" % (neg_entropy_file))
            sys.exit()

################################################################################


if __name__ == "__main__":
    main()
