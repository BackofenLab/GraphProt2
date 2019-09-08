#!/usr/bin/env python3

from lib import gp2lib
import numpy as np
import argparse
import os


def main(args):
    out_dataset = args.out_dataset

    in_folder = args.in_folder + "/" + args.in_dataset
    os.makedirs(args.out_folder + "/" + out_dataset)
    os.makedirs(args.out_folder + "/" + out_dataset + "/raw")
    os.makedirs(args.out_folder + "/" + out_dataset + "/processed")

    out_folder = args.out_folder + "/" + out_dataset + "/raw"
    # Use_sf and use_entr are parameters corresponding to site features, be careful when using use_entr since they
    # are biased
    # use_up: contains 1 feature (position based feature)
    # use_con: 2 features (position based features)
    # use_str_elem_up: 4 features (position based features)
    # bpp: 1 feature, only for forming graphs

    print('Dataset Out: ', out_dataset)
    # all_nodes_labels: anl
    # all_graph_indicators: agi
    # all_edges: ae
    # all_nodes_attributes: ana
    # all_graph_labels: agl
    anl, agi, ae, ana, agl, _ = gp2lib.load_geometric_data(in_folder,
                                                           gm_data=False,
                                                           use_str_elem_up=True,
                                                           use_str_elem_1h=False,
                                                           use_us_ds_labels=False,
                                                           use_region_labels=True,
                                                           center_vp=True,
                                                           vp_ext=30,
                                                           con_ext=50,
                                                           use_con=True,
                                                           use_sf=False,
                                                           use_entr=False,
                                                           add_1h_to_g=False,
                                                           all_nt_uc=True,
                                                           bpp_mode=2,
                                                           bpp_cutoff=0.5)
    gp2lib.normalize_geometric_all_nodes_attributes(ana, norm_mode=0)
    """Write files"""
    f = open(out_folder + "/" + out_dataset + "_graph_indicator.txt", 'w')
    f.writelines([str(e) + "\n" for e in agi])
    f.close()

    f = open(out_folder + "/" + out_dataset + "_graph_labels.txt", 'w')
    f.writelines([str(e) + "\n" for e in agl])
    f.close()

    f = open(out_folder + "/" + out_dataset + "_node_labels.txt", 'w')
    f.writelines([str(e) + "\n" for e in anl])
    f.close()

    f = open(out_folder + "/" + out_dataset + "_node_attributes.txt", 'w')
    f.writelines([s + "\n" for s in ana])
    f.close()

    f = open(out_folder + "/" + out_dataset + "_A.txt", 'w')
    f.writelines([str(e[0]) + ", " + str(e[1]) + "\n" for e in ae])
    f.close()

    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create TU datasets')
    parser.add_argument("--in_dataset", type=str, default="NCI1", help="Name of the in dataset")
    parser.add_argument("--out_dataset", type=str, default="NCI1", help="Name of the out dataset")
    parser.add_argument("--in_folder", type=str, help="Folder storing raw data")
    parser.add_argument("--out_folder", type=str, help="Folder storing geometric format data")

    args = parser.parse_args()
    main(args)