import torch
import torch.nn.functional as F
from MyNets import FunnelGNN, FunnelGNN_EdgeAttr
import numpy as np

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
    return np.vstack(score_all)


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
