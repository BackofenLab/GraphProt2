import random
import argparse
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from MyNets import FunnelGNN, FunnelGNN_EdgeAttr, MyDataset
from sklearn.model_selection import StratifiedKFold
import util
import os
from model_util import test, select_model


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
    results_path = current_path + "/results/cv_results/" + args.dataset
    print(dataset)
    print('Loading data done')

    skf = StratifiedKFold(n_splits=args.kfsplits, shuffle=False)

    for run in range(1):
        graphs = [dataset[idx] for idx in range(dataset.__len__())]
        random.shuffle(graphs)
        labels = [g.y for g in graphs]

        for list_tr_idx, list_te_idx in skf.split(np.zeros(len(labels)), labels):
            test_dataset = MyDataset([graphs[idx] for idx in list_te_idx])
            train_dataset = MyDataset([graphs[idx] for idx in list_tr_idx[:-len(list_te_idx)]])
            val_dataset = MyDataset([graphs[idx] for idx in list_tr_idx[-len(list_te_idx):]])

            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            opt_node_hidden_dim, opt_weight_decay, opt_lr = select_model(args, dataset, train_loader, val_loader, models_folder, device)

            model = FunnelGNN(input_dim=dataset.num_features, node_hidden_dim=opt_node_hidden_dim,
                              fc_hidden_dim=args.fc_hidden_dim, out_dim=dataset.num_classes).to(device)

            model.load_state_dict(torch.load(models_folder + "/" + str(opt_node_hidden_dim) + "_" + str(opt_weight_decay) + "_" + str(opt_lr)))
            loss, acc = test(test_loader, device, model)

            with open(results_path, 'a+') as f:
                f.write(str(acc) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="NCI1", help="Name of the dataset to consider")
    parser.add_argument("--patience", type=int, default=30, help="patience")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--epochs", type=int, default=300, help="number of training epochs")
    parser.add_argument("--fc_hidden_dim", type=int, default=128, help="number of units in hidden fully connected layer")
    parser.add_argument("--kfsplits", type=int, default=10, help="Number of splits for kfold")
    parser.add_argument("--list_lr", type=float, nargs='+', help="List of learning rates")
    parser.add_argument("--list_node_hidden_dim", type=int, nargs='+', help="List of node hidden dim")
    parser.add_argument("--list_weight_decay", type=float, nargs='+', help="List of l2 regularization")
    parser.add_argument('--use_node_attr', dest='use_node_attr', action='store_true', default=False, help="Using node attributes")
    args = parser.parse_args()

    print(args)
    main(args)
