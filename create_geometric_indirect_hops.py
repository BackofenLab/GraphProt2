#!/usr/bin/env python3

from lib import gp2lib
import numpy as np
import argparse
import os


def get_middle_graph(g=None):
    upper_labels = ['A', 'C', 'G', 'U']
    seed_nodes = []
    for n in g.nodes():
        if g.node[n]['label'] in upper_labels:
            seed_nodes.append(n)

    seed_nodes_neighbors = []
    for n in seed_nodes:
        seed_nodes_neighbors.extend(list(g.neighbors(n)))

    added_nodes = list(set(seed_nodes_neighbors) - set(seed_nodes))
    for n in added_nodes:
        seed_nodes_neighbors.extend(list(g.neighbors(n)))
    seed_nodes_neighbors = list(set(seed_nodes_neighbors))

    sub_g = g.subgraph(seed_nodes_neighbors)

    return sub_g


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

    graphs, seqs_1h, sfv_list, all_graph_labels = gp2lib.load_data(in_folder,
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
                                                                   all_nt_uc=False,
                                                                   bpp_mode=2,
                                                                   bpp_cutoff=0.5)

    # Min-max normalize (norm_mode=0) all feature vector values (apart from 1-hot features).
    gp2lib.normalize_graph_feat_vectors(graphs, norm_mode=0)

    """Creating graph dataset using TU format of pytorch geometric"""
    all_nodes_labels = []
    all_nodes_attributes = []
    all_graph_indicators = []
    all_edges = []
    dict_node_label_idx = {'a': 0, 'c': 1, 'g': 2, 'u': 3, 'A': 0, 'C': 1, 'G': 2, 'U': 3}

    n_idx = 1
    for g_idx, g_raw in enumerate(graphs):
        g = get_middle_graph(g=g_raw)

        sorted_nodes = sorted(g.nodes())
        dict_nodeid_nodeid = {}

        for node_idx, n in enumerate(sorted_nodes):
            dict_nodeid_nodeid[n] = node_idx

        n_nodes = g.number_of_nodes()
        all_graph_indicators.extend([g_idx+1]*n_nodes)

        for n in sorted_nodes:
            all_nodes_labels.append(dict_node_label_idx[g.node[n]['label']])
            node_attribute = [str(att) for att in g.node[n]['feat_vector']]
            all_nodes_attributes.append(",".join(node_attribute))

        for e in g.edges(data=True):
            all_edges.append((dict_nodeid_nodeid[e[0]]+n_idx, dict_nodeid_nodeid[e[1]]+n_idx))
            all_edges.append((dict_nodeid_nodeid[e[1]]+n_idx, dict_nodeid_nodeid[e[0]]+n_idx))

        n_idx += n_nodes

    """==============="""
    """Write files"""
    f = open(out_folder + "/" + out_dataset + "_graph_indicator.txt", 'w')
    f.writelines([str(e) + "\n" for e in all_graph_indicators])
    f.close()

    f = open(out_folder + "/" + out_dataset + "_graph_labels.txt", 'w')
    f.writelines([str(e) + "\n" for e in all_graph_labels])
    f.close()

    f = open(out_folder + "/" + out_dataset + "_node_labels.txt", 'w')
    f.writelines([str(e) + "\n" for e in all_nodes_labels])
    f.close()

    f = open(out_folder + "/" + out_dataset + "_node_attributes.txt", 'w')
    f.writelines([s + "\n" for s in all_nodes_attributes])
    f.close()

    f = open(out_folder + "/" + out_dataset + "_A.txt", 'w')
    f.writelines([str(e[0]) + ", " + str(e[1]) + "\n" for e in all_edges])
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
