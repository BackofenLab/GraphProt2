import torch
import torch.nn.functional as F
from MyNets import FunnelGNN
import numpy as np
import os
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import shutil


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
    opt_paras_string = str(opt_node_hidden_dim) + "_" + str(opt_weight_decay) + "_" + str(opt_lr)
    f = open(models_folder + "/opt_paras", 'w')
    f.write(opt_paras_string)
    f.close()
    return opt_node_hidden_dim, opt_weight_decay, opt_lr


def create_profile_dataset(save_dataset_name=None, x=None, list_node_labels=None, w_size=7, dataset_folder=None):
    raw_folder = dataset_folder + "/raw"
    processed_folder = dataset_folder + "/processed"
    if os.path.exists(dataset_folder):
        shutil.rmtree(dataset_folder)
        os.makedirs(dataset_folder)
        os.makedirs(raw_folder)
        os.makedirs(processed_folder)
    else:
        os.makedirs(dataset_folder)
        os.makedirs(raw_folder)
        os.makedirs(processed_folder)

    list_graph_indicators = []
    list_all_edges = []
    list_all_node_labels = []
    list_all_node_attributes = []
    n_idx = 1

    for g_idx in range(len(x)-w_size + 1):
        # graph indicators
        list_all_node_attributes.extend(x[g_idx:g_idx+w_size])
        list_all_node_labels.extend(list_node_labels[g_idx:g_idx+w_size])
        list_graph_indicators.extend([g_idx + 1]*w_size)
        # edges
        for n_idx_temp in range(w_size):
            if n_idx_temp != (w_size - 1):
                list_all_edges.append((n_idx_temp + n_idx, n_idx_temp + 1 + n_idx))
                list_all_edges.append((n_idx_temp + 1 + n_idx, n_idx_temp + n_idx))
        n_idx += w_size

    f = open(raw_folder + "/" + save_dataset_name + "_graph_indicator.txt", 'w')
    f.writelines([str(e) + "\n" for e in list_graph_indicators])
    f.close()

    f = open(raw_folder + "/" + save_dataset_name + "_A.txt", 'w')
    f.writelines([str(e[0]) + ", " + str(e[1]) + "\n" for e in list_all_edges])
    f.close()

    f = open(raw_folder + "/" + save_dataset_name + "_node_labels.txt", 'w')
    f.writelines([str(e) + "\n" for e in list_all_node_labels])
    f.close()

    f = open(raw_folder + "/" + save_dataset_name + "_node_attributes.txt", 'w')
    f.writelines([s + "\n" for s in list_all_node_attributes])
    f.close()


# def create_profile_dataset(save_dataset_name=None, list_node_labels=None, w_size=7, dataset_folder=None):
#     raw_folder = dataset_folder + "/raw"
#     processed_folder = dataset_folder + "/processed"
#     if os.path.exists(dataset_folder):
#         shutil.rmtree(dataset_folder)
#         os.makedirs(dataset_folder)
#         os.makedirs(raw_folder)
#         os.makedirs(processed_folder)
#     else:
#         os.makedirs(dataset_folder)
#         os.makedirs(raw_folder)
#         os.makedirs(processed_folder)
#
#     list_graph_indicators = []
#     list_all_edges = []
#     list_all_node_labels = []
#     n_idx = 1
#
#     for g_idx in range(len(list_node_labels)-w_size + 1):
#         # graph indicators
#         list_all_node_labels.extend(list_node_labels[g_idx:g_idx+w_size])
#         list_graph_indicators.extend([g_idx + 1]*w_size)
#         # edges
#         for n_idx_temp in range(w_size):
#             if n_idx_temp != (w_size - 1):
#                 list_all_edges.append((n_idx_temp + n_idx, n_idx_temp + 1 + n_idx))
#                 list_all_edges.append((n_idx_temp + 1 + n_idx, n_idx_temp + n_idx))
#         n_idx += w_size
#
#     f = open(raw_folder + "/" + save_dataset_name + "_graph_indicator.txt", 'w')
#     f.writelines([str(e) + "\n" for e in list_graph_indicators])
#     f.close()
#
#     f = open(raw_folder + "/" + save_dataset_name + "_A.txt", 'w')
#     f.writelines([str(e[0]) + ", " + str(e[1]) + "\n" for e in list_all_edges])
#     f.close()
#
#     f = open(raw_folder + "/" + save_dataset_name + "_node_labels.txt", 'w')
#     f.writelines([str(e) + "\n" for e in list_all_node_labels])
#     f.close()


def get_scores_profile(dataset_name=None, x=None, list_node_labels=None, list_w_size=[3, 5, 7], model=None, device=None,
                       batch_size=None, geometric_folder=None):

    all_scores = []
    for w_size in list_w_size:
        save_dataset_name = dataset_name + "_" + str(w_size)
        dataset_folder = geometric_folder + "/" + save_dataset_name
        create_profile_dataset(save_dataset_name=save_dataset_name, x=x, list_node_labels=list_node_labels,
                               w_size=w_size, dataset_folder=dataset_folder)
        dataset_w = TUDataset(geometric_folder, name=save_dataset_name, use_node_attr=True)
        #loader = DataLoader(dataset_w, batch_size=batch_size, shuffle=False)
        loader = DataLoader(dataset_w, batch_size=batch_size, shuffle=False)
        scores = get_scores(loader, device, model)
        scores = [scores[0]]*int(w_size/2) + scores + [scores[-1]]*int(w_size/2)
        all_scores.append(scores)
    scores_mean = list(np.mean(all_scores, axis=0))
    all_scores.append(scores_mean)
    return all_scores


# def get_scores_profile(dataset_name=None, list_node_labels=None, list_w_size=[3, 5, 7], model=None, device=None,
#                        batch_size=None, geometric_folder=None):
#
#     all_scores = []
#     for w_size in list_w_size:
#         save_dataset_name = dataset_name + "_" + str(w_size)
#         dataset_folder = geometric_folder + "/" + save_dataset_name
#         create_profile_dataset(save_dataset_name=save_dataset_name, list_node_labels=list_node_labels,
#                                w_size=w_size, dataset_folder=dataset_folder)
#         dataset_w = TUDataset(geometric_folder, name=save_dataset_name, use_node_attr=False)
#         loader = DataLoader(dataset_w, batch_size=batch_size, shuffle=False)
#         scores = get_scores(loader, device, model)
#         scores = [scores[0]]*int(w_size/2) + scores + [scores[-1]]*int(w_size/2)
#         all_scores.append(scores)
#     return list(np.mean(all_scores, axis=0))
