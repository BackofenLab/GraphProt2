import torch
import torch.nn.functional as F
from MyNets import FunnelGNN
import numpy as np
import os
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader


def train(epoch, device, model, optimizer, train_loader):
    model.train()

    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y, reduction="mean")
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader, device, model):
    model.eval()
    loss_all = 0

    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y, reduction="mean")
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss_all += loss.item() * data.num_graphs

    return loss_all / len(loader.dataset), correct / len(loader.dataset)


def get_scores(loader, device, model):
    model.eval()
    score_all = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            output = torch.exp(output)
            output = output.cpu().detach().numpy()[:, 1]
            score_all.extend(output)
    return score_all


def select_model(args, dataset, train_loader, val_loader, models_folder, device):
    opt_node_hidden_dim = args.list_node_hidden_dim[0]
    opt_weight_decay = args.list_weight_decay[0]
    opt_lr = args.list_lr[0]
    opt_val_loss = 1000000000.0
    for node_hidden_dim in args.list_node_hidden_dim:
        for weight_decay in args.list_weight_decay:
            for lr in args.list_lr:
                model = FunnelGNN(input_dim=dataset.num_features, node_hidden_dim=node_hidden_dim,
                                  fc_hidden_dim=args.fc_hidden_dim, out_dim=2).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                best_val_loss = 1000000000.0

                elapsed_patience = 0
                for epoch in range(0, args.epochs):
                    if elapsed_patience > args.patience:
                        break
                    train_loss = train(epoch, device, model, optimizer, train_loader)
                    val_loss, val_acc = test(val_loader, device, model)

                    if val_loss < best_val_loss:
                        elapsed_patience = 0
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), models_folder + "/" + str(node_hidden_dim) + "_" + str(weight_decay) + "_" + str(lr))
                    else:
                        elapsed_patience += 1

                if best_val_loss < opt_val_loss:
                    opt_val_loss = best_val_loss
                    opt_node_hidden_dim = node_hidden_dim
                    opt_weight_decay = weight_decay
                    opt_lr = lr
    return opt_node_hidden_dim, opt_weight_decay, opt_lr

def create_profile_dataset(dataset_name=None, rna_seq=None, w_size=50, dataset_folder=None):
    raw_folder = dataset_folder + "/raw"
    processed_folder = dataset_folder + "/processed"

    dict_nodelabel_id = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        os.makedirs(raw_folder)
        os.makedirs(processed_folder)

    list_graph_indicators = []
    list_edges = []
    list_node_labels = []
    n_idx = 1

    for g_idx in range(len(rna_seq)-w_size + 1):
        # graph indicators
        seq = rna_seq[g_idx:g_idx+w_size]
        n_seq = len(seq)
        list_graph_indicators.extend([g_idx + 1] * n_seq)
        # edges
        for n_idx_temp in range(n_seq):
            if n_idx_temp != (n_seq - 1):
                list_edges.append((n_idx_temp + n_idx, n_idx_temp + 1 + n_idx))
                list_edges.append((n_idx_temp + 1 + n_idx, n_idx_temp + n_idx))
            list_node_labels.append(dict_nodelabel_id[seq[n_idx_temp]])
        n_idx += n_seq

    f = open(raw_folder + "/" + dataset_name + "_graph_indicator.txt", 'w')
    f.writelines([str(e) + "\n" for e in list_graph_indicators])
    f.close()

    f = open(raw_folder + "/" + dataset_name + "_A.txt", 'w')
    f.writelines([str(e[0]) + ", " + str(e[1]) + "\n" for e in list_edges])
    f.close()

    f = open(raw_folder + "/" + dataset_name + "_node_labels.txt", 'w')
    f.writelines([str(e) + "\n" for e in list_node_labels])
    f.close()

    print("Done")


def get_scores_profile(rna_name=None, rna_seq=None, list_w_size=[3, 5, 7], model=None, device=None, batch_size=None,
                       geometric_folder=None):

    all_scores = []
    for w_size in list_w_size:
        dataset_name = rna_name + "_" + str(w_size)
        dataset_folder = geometric_folder + "/" + dataset_name
        create_profile_dataset(dataset_name=dataset_name, rna_seq=rna_seq, w_size=w_size, dataset_folder=dataset_folder)
        processed_file = dataset_folder + '/processed/data.pt'
        if os.path.exists(processed_file):
            os.remove(processed_file)
        dataset = TUDataset(dataset_folder, name=dataset_name, use_node_attr=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        scores = get_scores(loader, device, model)
        scores = [scores[0]]*int(w_size/2) + scores + [scores[-1]]*int(w_size/2)
        all_scores.append(scores)
    return list(np.mean(all_scores, axis=0))
