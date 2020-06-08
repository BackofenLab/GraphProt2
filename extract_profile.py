from model_util import get_scores_profile, get_scores
import torch
from MyNets import FunnelGNN, MyDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import os
import random
import argparse
import numpy as np


def decompose_node_attr(x):
    x = x.tolist()
    dict_onehot_label = {'1000': 5, '0100': 6, '0010': 7, '0001': 8}
    dict_onehot_nucleotide = {'1000': 'A', '0100': 'C', '0010': 'G', '0001': 'U'}
    dict_reg_exon_intron = {1: 'E', 0: 'I'}
    list_nucleotides = []
    list_attrs = []
    list_node_labels = []
    list_cons_1 = []
    list_cons_2 = []
    list_exons_introns = []

    for idx in range(len(x)):
        attrs = [str(att) for att in x[idx][:4]]
        onehot = "".join([str(int(i)) for i in x[idx][4:]])
        list_attrs.append(",".join(attrs))
        list_node_labels.append(dict_onehot_label[onehot])
        list_nucleotides.append(dict_onehot_nucleotide[onehot])
        list_cons_1.append(str(x[idx][0]))
        list_cons_2.append(str(x[idx][1]))
        list_exons_introns.append(dict_reg_exon_intron[int(x[idx][2])])

    return list_attrs, list_node_labels, list_nucleotides, list_exons_introns, list_cons_1, list_cons_2


def generate_weblogo_format(dataset_name, logos_input_folder, list_all_scores, list_all_nucleotides,
                            list_all_exons_introns, list_all_cons_1, list_all_cons_2, list_logo_size,
                            top_scores_logo, list_w_size):
    for w1_idx, w1 in enumerate(list_w_size):
        for w2 in list_logo_size:
            list_max_vals = []
            list_log_seqs = []
            list_log_exons_introns = []
            list_log_cons_1 = []
            list_log_cons_2 = []
            for g_idx in range(len(list_all_nucleotides)):
                scores = list_all_scores[g_idx][w1_idx]
                nucleotides = list_all_nucleotides[g_idx]
                exons_introns = list_all_exons_introns[g_idx]
                cons_1 = list_all_cons_1[g_idx]
                cons_2 = list_all_cons_2[g_idx]
                max_val = max(scores)
                max_idx = scores.index(max_val)
                log_seq = "".join(nucleotides[max_idx-int((w2-1)/2):max_idx+int((w2-1)/2) + 1])
                log_exons_introns = "".join(exons_introns[max_idx-int((w2-1)/2):max_idx+int((w2-1)/2) + 1])
                log_cons_1 = ",".join(cons_1[max_idx-int((w2-1)/2):max_idx+int((w2-1)/2) + 1])
                log_cons_2 = ",".join(cons_2[max_idx - int((w2 - 1) / 2):max_idx + int((w2 - 1) / 2) + 1])
                if len(log_seq) == w2:
                    list_max_vals.append(max_val)
                    list_log_seqs.append(log_seq)
                    list_log_exons_introns.append(log_exons_introns)
                    list_log_cons_1.append(log_cons_1)
                    list_log_cons_2.append(log_cons_2)
            _, list_log_seqs, list_log_exons_introns, list_log_cons_1, list_log_cons_2 = zip(*sorted(zip(list_max_vals,
                                                                                                         list_log_seqs,
                                                                                                         list_log_exons_introns,
                                                                                                         list_log_cons_1,
                                                                                                         list_log_cons_2),
                                                                                                     reverse=True))
            for top_k in top_scores_logo:
                f = open(logos_input_folder + dataset_name + "_nucleotide_" + str(w1) + "_" + str(w2) + "_" + str(top_k), 'w')
                f.writelines([s + "\n" for s in list_log_seqs[:top_k]])
                f.close()
                """=========================="""
                f = open(logos_input_folder + dataset_name + "_exons_introns_" + str(w1) + "_" + str(w2) + "_" + str(top_k), 'w')
                f.writelines([s + "\n" for s in list_log_exons_introns[:top_k]])
                f.close()
                """=========================="""
                f = open(logos_input_folder + dataset_name + "_cons_1_" + str(w1) + "_" + str(w2) + "_" + str(top_k), 'w')
                f.writelines([s + "\n" for s in list_log_cons_1[:top_k]])
                f.close()
                """=========================="""
                f = open(logos_input_folder + dataset_name + "_cons_2_" + str(w1) + "_" + str(w2) + "_" + str(top_k), 'w')
                f.writelines([s + "\n" for s in list_log_cons_2[:top_k]])
                f.close()


def save_sorted_predicted_seqs(pos_site_scores, list_all_nucleotides, list_all_exons_introns, dataset_name,
                               list_all_cons_1, list_all_cons_2, logos_input_folder, list_w_size, list_all_scores):
    new_seqs = ["".join(s) for s in list_all_nucleotides]
    sorted_scores, sorted_seqs, list_all_exons_introns, list_all_cons_1, list_all_cons_2, list_all_scores = \
        zip(*sorted(zip(pos_site_scores, new_seqs, list_all_exons_introns, list_all_cons_1, list_all_cons_2,
                        list_all_scores), reverse=True))
    top_sites = 200
    """=========================="""
    f = open(logos_input_folder + dataset_name + "_site_scores", 'w')
    f.writelines([str(e) + "\n" for e in sorted_scores[:top_sites]])
    f.close()
    """=========================="""
    f = open(logos_input_folder + dataset_name + "_site_seqs", 'w')
    f.writelines([s + "\n" for s in sorted_seqs[:top_sites]])
    f.close()
    """=========================="""
    f = open(logos_input_folder + dataset_name + "_site_ex_in", 'w')
    f.writelines(["".join(s) + "\n" for s in list_all_exons_introns[:top_sites]])
    f.close()
    """=========================="""
    f = open(logos_input_folder + dataset_name + "_site_cons_1", 'w')
    f.writelines([",".join(s) + "\n" for s in list_all_cons_1[:top_sites]])
    f.close()
    """=========================="""
    f = open(logos_input_folder + dataset_name + "_site_cons_2", 'w')
    f.writelines([",".join(s) + "\n" for s in list_all_cons_2[:top_sites]])
    f.close()
    """=========================="""
    for w1_idx, w in enumerate(list_w_size):
        list_nucleotide_scores = [list_all_scores[g_idx][w1_idx] for g_idx in range(len(list_all_nucleotides))]
        nucleotide_scores_str = []
        for nucleotide_scores in list_nucleotide_scores:
            s = ",".join([str(e) for e in nucleotide_scores])
            nucleotide_scores_str.append(s)

        f = open(logos_input_folder + dataset_name + "_nucleotide_site_scores_" + str(w), 'w')
        f.writelines([e + "\n" for e in nucleotide_scores_str[:top_sites]])
        f.close()


def main(args):
    random.seed(1)
    """=================================================="""
    current_path = os.getcwd()
    geometric_data_path = current_path + "/geometric_data/"
    logos_input_folder = current_path + "/logos_input/"
    model_path = current_path + "/models/profile_models/" + args.dataset_name + "/128_0.0001_0.0001"
    dataset = TUDataset(geometric_data_path, name=args.dataset_name, use_node_attr=True)
    print(dataset.num_features)
    print(dataset)
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    graphs = [dataset[idx] for idx in range(dataset.__len__())]
    labels = [g.y for g in graphs]
    list_tr_idx, _ = list(skf.split(np.zeros(len(labels)), labels))[0]
    positive_graphs = []
    for idx in list_tr_idx:
        if graphs[idx].y == 1:
            positive_graphs.append(graphs[idx])

    positive_dataset = MyDataset(positive_graphs)
    positive_loader = DataLoader(positive_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FunnelGNN(input_dim=dataset.num_features, node_hidden_dim=128, fc_hidden_dim=128, out_dim=2).to(device)
    model.load_state_dict(torch.load(model_path))
    pos_site_scores = get_scores(positive_loader, device, model)

    list_all_scores = []
    list_all_nucleotides = []
    list_all_exons_introns = []
    list_all_cons_1 = []
    list_all_cons_2 = []
    for g_idx, g in enumerate(positive_graphs):
        list_attrs, list_node_labels, list_nucleotides, list_exons_introns, list_cons_1, list_cons_2 = decompose_node_attr(g.x)
        if len(set(list_node_labels)) == 4:
            all_scores = get_scores_profile(dataset_name=args.dataset_name, x=list_attrs, list_node_labels=list_node_labels,
                                        list_w_size=args.list_w_size, model=model, device=device, batch_size=50,
                                        geometric_folder=geometric_data_path)
            list_all_scores.append(all_scores)
            list_all_nucleotides.append(list_nucleotides)
            list_all_exons_introns.append(list_exons_introns)
            list_all_cons_1.append(list_cons_1)
            list_all_cons_2.append(list_cons_2)
    print('Generating logo format')
    save_sorted_predicted_seqs(pos_site_scores, list_all_nucleotides, list_all_exons_introns, args.dataset_name,
                               list_all_cons_1, list_all_cons_2, logos_input_folder, args.list_w_size + [0], list_all_scores)
    generate_weblogo_format(args.dataset_name, logos_input_folder, list_all_scores, list_all_nucleotides,
                            list_all_exons_introns, list_all_cons_1, list_all_cons_2, args.list_logo_size,
                            args.top_scores_logo, args.list_w_size + [0])

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset_name", type=str, default="FMR1")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--node_hidden_dim", type=int, default=128, help="Node hidden dim")
    parser.add_argument("--fc_hidden_dim", type=int, default=128, help="Number of units in hidden fully connected layer")
    parser.add_argument("--list_w_size", type=int, nargs='+', default=[9, 13, 17], help="List of window sizes")
    parser.add_argument("--list_logo_size", type=int, nargs='+', default=[7, 9, 11, 13], help="List of window sizes")
    parser.add_argument("--top_scores_logo", type=int, nargs='+', default=[300, 500, 1000], help="Top scores considered for logo generation")
    args = parser.parse_args()
    print(args)
    main(args)
