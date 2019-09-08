import random
import argparse
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from MyNets import FunnelGNN, MyDataset
from sklearn.model_selection import StratifiedKFold
import util
import os
from model_util import test, select_model
import copy


def main(args):
    random.seed(1)
    current_path = os.getcwd()

    geometric_cv_data_path = current_path + "/data/geometric_cv/" + args.dataset
    processed_file = geometric_cv_data_path + '/processed/data.pt'
    if os.path.exists(processed_file):
        os.remove(processed_file)
    dataset = TUDataset(geometric_cv_data_path, name=args.dataset, use_node_attr=args.use_node_attr)

    models_folder = current_path + "/models/cv_models/" + args.dataset
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    results_path = current_path + "/results/cv_results/" + args.dataset + "_generic"
    print(dataset)
    print('Loading data done')

    all_positive_graphs = []
    all_negative_graphs = []
    all_positive_graphs_labels = []
    for idx in range(dataset.__len__()):
        if dataset[idx].y.item() == 0:
            all_negative_graphs.append(dataset[idx])
        else:
            all_positive_graphs.append(dataset[idx])
            all_positive_graphs_labels.append(dataset[idx].y)

    random.shuffle(all_negative_graphs)
    list_positive_graphs_labels = [l.item() for l in all_positive_graphs_labels]

    list_unique_positive_graphs_labels = list(set(list_positive_graphs_labels))

    for positive_graph_label in list_unique_positive_graphs_labels:
        all_positive_graphs_temp = copy.deepcopy(all_positive_graphs)

        train_graphs_positive = []
        test_graphs_positive = []
        for g in all_positive_graphs_temp:
            if g.y.item() == positive_graph_label:
                test_graphs_positive.append(g)
            else:
                train_graphs_positive.append(g)

        n_test_graphs_positive = len(test_graphs_positive)
        for g in train_graphs_positive:
            g.y = torch.tensor([1])
        for g in test_graphs_positive:
            g.y = torch.tensor([1])

        train_graphs_negative = all_negative_graphs[:-n_test_graphs_positive]
        test_graphs_negative = all_negative_graphs[-n_test_graphs_positive:]

        all_train_graphs = train_graphs_positive + train_graphs_negative
        test_graphs = test_graphs_positive + test_graphs_negative

        random.shuffle(all_train_graphs)
        random.shuffle(test_graphs)

        n_test_graphs = len(test_graphs)
        train_graphs = all_train_graphs[:-n_test_graphs]
        val_graphs = all_train_graphs[-n_test_graphs:]

        test_dataset = MyDataset(test_graphs)
        train_dataset = MyDataset(train_graphs)
        val_dataset = MyDataset(val_graphs)

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        opt_node_hidden_dim, opt_weight_decay, opt_lr = select_model(args, dataset, train_loader, val_loader, models_folder, device)

        model = FunnelGNN(input_dim=dataset.num_features, node_hidden_dim=opt_node_hidden_dim,
                          fc_hidden_dim=args.fc_hidden_dim, out_dim=2).to(device)

        model.load_state_dict(torch.load(models_folder + "/" + str(opt_node_hidden_dim) + "_" + str(opt_weight_decay) + "_" + str(opt_lr)))
        loss, acc = test(test_loader, device, model)

        with open(results_path, 'a+') as f:
            f.write(str(acc) + '\n')
        print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="NCI1", help="Name of the dataset to consider")
    parser.add_argument("--patience", type=int, default=30, help="patience")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--epochs", type=int, default=300, help="number of training epochs")
    parser.add_argument("--fc_hidden_dim", type=int, default=128, help="number of units in hidden fully connected layer")
    parser.add_argument("--list_lr", type=float, nargs='+', help="List of learning rates")
    parser.add_argument("--list_node_hidden_dim", type=int, nargs='+', help="List of node hidden dim")
    parser.add_argument("--list_weight_decay", type=float, nargs='+', help="List of l2 regularization")
    parser.add_argument('--use_node_attr', dest='use_node_attr', action='store_true', default=False, help="Using node attributes")
    args = parser.parse_args()

    print(args)
    main(args)
