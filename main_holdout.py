import random
import argparse
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from MyNets import FunnelGNN, FunnelGNN_EdgeAttr, MyDataset
import util
import os
from model_util import test, select_model


def main(args):
    random.seed(1)
    current_path = os.getcwd()

    geometric_holdout_test_data_path = current_path + "/data/geometric_holdout_test/" + args.dataset
    geometric_holdout_train_data_path = current_path + "/data/geometric_holdout_train/" + args.dataset

    processed_file_train = geometric_holdout_train_data_path + '/processed/data.pt'
    processed_file_test = geometric_holdout_test_data_path + '/processed/data.pt'

    if os.path.exists(processed_file_train):
        os.remove(processed_file_train)
    if os.path.exists(processed_file_test):
        os.remove(processed_file_test)

    train_dataset = TUDataset(geometric_holdout_train_data_path, name=args.dataset, use_node_attr=args.use_node_attr)
    test_dataset = TUDataset(geometric_holdout_test_data_path, name=args.dataset, use_node_attr=args.use_node_attr)

    models_folder = current_path + "/models/holdout_models/" + args.dataset
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    results_path = current_path + "/results/holdout_results/" + args.dataset
    nb_train_graphs = int(0.9*train_dataset.__len__())
    train_graphs = [train_dataset[idx] for idx in range(nb_train_graphs)]
    val_graphs = [train_dataset[idx] for idx in range(nb_train_graphs, train_dataset.__len__())]
    test_graphs = [test_dataset[idx] for idx in range(test_dataset.__len__())]
    random.shuffle(train_graphs)
    random.shuffle(test_graphs)

    test_dataset = MyDataset(test_graphs)
    train_dataset = MyDataset(train_graphs)
    val_dataset = MyDataset(val_graphs)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt_node_hidden_dim, opt_weight_decay, opt_lr = select_model(args, train_dataset, train_loader, val_loader, models_folder, device)

    model = FunnelGNN(input_dim=train_dataset.num_features, node_hidden_dim=opt_node_hidden_dim,
                      fc_hidden_dim=args.fc_hidden_dim, out_dim=train_dataset.num_classes).to(device)

    model.load_state_dict(torch.load(models_folder + str(opt_node_hidden_dim) + "_" + str(opt_weight_decay) + "_" + str(opt_lr)))
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
    parser.add_argument("--list_lr", type=float, nargs='+', help="List of learning rates")
    parser.add_argument("--list_node_hidden_dim", type=int, nargs='+', help="List of node hidden dim")
    parser.add_argument("--list_weight_decay", type=float, nargs='+', help="List of l2 regularization")
    parser.add_argument('--use_node_attr', dest='use_node_attr', action='store_true', default=False, help="Use node attributes")
    args = parser.parse_args()

    print(args)
    main(args)
