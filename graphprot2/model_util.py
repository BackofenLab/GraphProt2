import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from graphprot2.MyNets import FunnelGNN
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import numpy as np
import shutil
import sys
import os


"""

python3 -m doctest model_util.py

"""

################################################################################

def train(device, model, optimizer, train_loader):
    model.train()

    loss_all = 0
    # Loop over dataset.
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y, reduction="mean")
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


################################################################################

def test(loader, device, model):
    """
    Test data, return loss and accuracy.

    """
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


################################################################################

def get_scores(loader, device, model,
               min_max_norm=False):
    """
    Get whole site scores.

        print("data.batch", data.batch)
        l_x = len(data.x)
        print("data.x:", data.x)
        print("data.y:", data.y)
        idx_list = []
        pr_list = []
        sm = torch.nn.Softmax()
        output = model(data.x, data.edge_index, data.batch)
        probs = sm(output)
        class_1_prob = probs[0][1].cpu().detach().numpy()
        #print("output:", output)
        print("site class_1_prob:", class_1_prob)

    """

    model.eval()
    site_scores = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            l_x = len(data.x)
            #print("len.x:", l_x)
            #print("data.x:", data.x)
            #print("data.y:", data.y)
            output = model(data.x, data.edge_index, data.batch)
            output = torch.exp(output)
            output = output.cpu().detach().numpy()[:, 1]
            if min_max_norm:
                for o in output:
                    o_norm = min_max_normalize_probs(o, 1, 0, borders=[-1, 1])
                    site_scores.append(o_norm)
            else:
                site_scores.extend(output)
    return site_scores


################################################################################

def get_site_probs(loader, device, model):
    """
    Get class 1 site probabilities.

    sm = torch.nn.Softmax()
    probs = sm(output)
    class_0_prob = probs[0][0].cpu().detach().numpy()
    class_1_prob = probs[0][1].cpu().detach().numpy()
    class_1_probs.append(class_1_prob)

    print("output exp:", output)
    for i,o in enumerate(output):
        print(i, float(o))

    """

    model.eval()
    class_1_probs = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            output = torch.exp(output)
            output = output.cpu().detach().numpy()[:, 1]
            class_1_probs.extend(output)
    return class_1_probs


################################################################################

def get_subgraph_scores(loader, device, model, list_win_sizes,
                        softmax=False):
    """
    Get position-wise scores by scoring subgraphs, and normalize the
    class 1 softmax probabilities from -1 .. 1. Only works with
    batch size == 1.

    """
    # List of lists to store probability list for each site.
    pr_ll = []

    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            l_x = len(data.x)
            # Make graph index list.
            idx_list = []
            for i in range(l_x):
                idx_list.append(i)
            sm = torch.nn.Softmax()
            pr_win_list = []
            for win_size in list_win_sizes:
                win_extlr = int(win_size / 2)
                pr_list = []
                for i in range(l_x):
                    s = i - win_extlr
                    e = i + win_extlr + 1
                    if s < 0:
                        s = 0
                    if e > l_x:
                        e = l_x
                    subset = idx_list[s:e]
                    sub_edge_index = subgraph(subset, data.edge_index)
                    output = model(data.x, sub_edge_index[0], data.batch)
                    if softmax:
                        probs = sm(output)
                        #class_0_prob = float(probs[0][0].cpu().detach().numpy())
                        class_1_prob = float(probs[0][1].cpu().detach().numpy())
                        pr_list.append(class_1_prob)
                    else:
                        output = torch.exp(output)
                        output = output.cpu().detach().numpy()[:, 1]
                        class_1_prob = float(output[0])
                        pr_list.append(class_1_prob)

                for i, pr in enumerate(pr_list):
                    pr_list[i] = min_max_normalize_probs(pr, 1, 0, borders=[-1, 1])

                # Deal with scores at ends.
                start_idx = idx_list[:win_extlr]
                end_idx = idx_list[-win_extlr:]
                for i in start_idx:
                    pr_list[i] = pr_list[win_extlr]
                for i in end_idx:
                    pr_list[i] = pr_list[l_x-win_extlr-1]
                pr_win_list.append(pr_list)

            # Calculate mean list scores.
            mean_pr_list = list(np.mean(pr_win_list, axis=0))
            # Add mean scores list to existing list of lists.
            pr_ll.append(mean_pr_list)

    assert pr_ll, "pr_ll empty"
    return pr_ll


################################################################################

def min_max_normalize_probs(x, max_x, min_x,
                            borders=False):
    """
    Min-max normalization of input x, given dataset max and min.

    >>> min_max_normalize_probs(20, 30, 10)
    0.5
    >>> min_max_normalize_probs(30, 30, 10)
    1.0
    >>> min_max_normalize_probs(10, 30, 10)
    0.0
    >>> min_max_normalize_probs(0.5, 1, 0, borders=[-1, 1])
    0.0

    Formula from:
    https://en.wikipedia.org/wiki/Feature_scaling

    """
    # If min=max, all values the same, so return x.
    if (max_x - min_x) == 0:
        return x
    else:
        if borders:
            assert len(borders) == 2, "list of 2 values expected"
            a = borders[0]
            b = borders[1]
            assert a < b, "a should be < b"
            return a + (x-min_x)*(b-a) / (max_x - min_x)
        else:
            return (x-min_x) / (max_x - min_x)


################################################################################

def train_final_model_old(args, dataset, train_loader, test_loader,
                          out_folder, device,
                          data_id="data_id",
                          out_model_file = "final.model",
                          out_model_params_file = "final.params",
                          opt_lr=0.0001,
                          opt_node_hidden_dim=128,
                          opt_weight_decay=0.00001):
    """
    Train final model.

    """
    # Output files.
    model_file = out_folder + "/" + out_model_file
    params_file = out_folder + "/" + out_model_params_file

    # Define network.
    model = FunnelGNN(input_dim=dataset.num_features, node_hidden_dim=opt_node_hidden_dim,
                      fc_hidden_dim=args.fc_hidden_dim, out_dim=2).to(device)
    # Define optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_lr, weight_decay=opt_weight_decay)

    best_loss = 1000000000.0
    best_acc = 0
    elapsed_patience = 0

    # Train for epoch epochs, with patience break.
    c_epochs = 0
    for epoch in range(0, args.epochs):
        if elapsed_patience > args.patience:
            break
        c_epochs += 1
        # Train network, going over whole dataset.
        train_loss = train(device, model, optimizer, train_loader)
        # Get loss on test set.
        test_loss, test_acc = test(test_loader, device, model)

        # If loss decreases, store current model parameters and HP setting.
        if test_loss < best_loss:
            elapsed_patience = 0
            best_loss = test_loss
            torch.save(model.state_dict(), model_file)
        else:
            elapsed_patience += 1
        if test_acc > best_acc:
            best_acc = test_acc

    print("# epochs elapsed:  ", c_epochs)
    print("Model accuracy:    ", best_acc)

    # Store HPs in params file.
    PAROUT = open(params_file, "w")
    PAROUT.write("fc_hidden_dim\t%s\n" %(str(args.fc_hidden_dim)))
    PAROUT.write("epochs\t%s\n" %(str(args.epochs)))
    PAROUT.write("patience\t%s\n" %(str(args.patience)))
    PAROUT.write("batch_size\t%s\n" %(str(args.batch_size)))
    PAROUT.write("lr\t%s\n" %(str(opt_lr)))
    PAROUT.write("node_hidden_dim\t%s\n" %(str(opt_node_hidden_dim)))
    PAROUT.write("weight_decay\t%s\n" %(str(opt_weight_decay)))
    PAROUT.write("data_id\t%s" %(data_id))
    PAROUT.close()


################################################################################

def train_final_model(args, n_features, train_dataset, train_epochs, device,
                      final_model_path = "final.model",
                      batch_size=50,
                      lr=0.0001,
                      node_hidden_dim=128,
                      weight_decay=0.00001):
    """
    Train final model.

    Just train for given number of epochs, no patience or loss calculation.

    """
    # Output files.
    model_file = final_model_path

    # Train loader.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define network.
    model = FunnelGNN(input_dim=n_features, node_hidden_dim=node_hidden_dim,
                      fc_hidden_dim=args.fc_hidden_dim, out_dim=2).to(device)
    # Define optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(0, train_epochs):
        # Train network, going over whole dataset.
        train_loss = train(device, model, optimizer, train_loader)

    torch.save(model.state_dict(), model_file)


################################################################################

def select_model(args, n_features, train_dataset, val_dataset,
                 model_folder, device,
                 hps2acc_dic=None,
                 hps2epo_dic=None):
    """
    Select best hyperparameter combination for given train_dataset and val_dataset.
    Return optimal hyperparameters. Use this function for cross validation to
    estimate generalization performance.

    """
    opt_batch_size = args.list_batch_size[0]
    opt_node_hidden_dim = args.list_node_hidden_dim[0]
    opt_weight_decay = args.list_weight_decay[0]
    opt_lr = args.list_lr[0]
    opt_val_loss = 1000000000.0
    opt_acc = 0
    opt_epochs = 0
    opt_dic = {}

    for batch_size in args.list_batch_size:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        for node_hidden_dim in args.list_node_hidden_dim:
            for weight_decay in args.list_weight_decay:
                for lr in args.list_lr:
                    # Hyperparameter string.
                    hp_str = str(batch_size) + "_" + str(node_hidden_dim) + "_" + str(weight_decay) + "_" + str(lr)
                    model_path = model_folder + "/" + hp_str
                    # print('Processing with ', model_name + "_" + str(batch_size) + "_" + str(node_hidden_dim) + "_" + str(weight_decay) + "_" + str(lr))
                    model = FunnelGNN(input_dim=n_features, node_hidden_dim=node_hidden_dim,
                                     fc_hidden_dim=args.fc_hidden_dim, out_dim=2).to(device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                    best_val_loss = 1000000000.0
                    best_val_acc = 0
                    elapsed_patience = 0
                    c_epochs = 0
                    for epoch in range(0, args.epochs):
                        c_epochs += 1
                        if elapsed_patience > args.patience:
                            break
                        train_loss = train(device, model, optimizer, train_loader)
                        val_loss, val_acc = test(val_loader, device, model)

                        if val_loss < best_val_loss:
                            #print('save model...')
                            elapsed_patience = 0
                            best_val_loss = val_loss
                            best_val_acc = val_acc
                            torch.save(model.state_dict(), model_path)
                        else:
                            elapsed_patience += 1

                    # Store used epochs and best accuarcy for this HP combination.
                    if hps2acc_dic is not None:
                        if hp_str not in hps2acc_dic:
                            hps2acc_dic[hp_str] = []
                            hps2acc_dic[hp_str].append(best_val_acc)
                        else:
                            hps2acc_dic[hp_str].append(best_val_acc)
                    if hps2epo_dic is not None:
                        if hp_str not in hps2epo_dic:
                            hps2epo_dic[hp_str] = []
                            hps2epo_dic[hp_str].append(c_epochs)
                        else:
                            hps2epo_dic[hp_str].append(c_epochs)

                    if best_val_loss < opt_val_loss:
                        opt_val_loss = best_val_loss
                        opt_node_hidden_dim = node_hidden_dim
                        opt_weight_decay = weight_decay
                        opt_lr = lr
                        opt_acc = best_val_acc
                        opt_epochs = c_epochs
                        opt_batch_size = batch_size

    opt_dic["opt_batch_size"] = opt_batch_size
    opt_dic["opt_node_hidden_dim"] = opt_node_hidden_dim
    opt_dic["opt_weight_decay"] = opt_weight_decay
    opt_dic["opt_lr"] = opt_lr
    opt_dic["opt_acc"] = opt_acc
    opt_dic["opt_epochs"] = opt_epochs
    return opt_dic


################################################################################

def select_model_old(args, dataset, train_loader, val_loader, models_folder, device):
    """
    Do the hyperparameter (HP) optimization (inner CV).

    For every HP combination train a model and evaluate validation loss.
    Each time loss < opt_val_loss, store HP setting as optimal one.
    To just train one model, give only one combination. Stored model name
    is made up of the optimal hyperparameters for #nodes of hidden
    dimensions, weight decay, and learning rate:
    <node_hidden_dim>_<weight_decay>_<lr>

    Return optimal HPs.

    """

    # Select first HP values in list.
    opt_node_hidden_dim = args.list_node_hidden_dim[0]
    opt_weight_decay = args.list_weight_decay[0]
    opt_lr = args.list_lr[0]
    opt_val_loss = 1000000000.0

    # For every HP combination (inner CV).
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
                    # Train network, going over whole dataset.
                    train_loss = train(device, model, optimizer, train_loader)
                    # Get loss on validation set.
                    val_loss, val_acc = test(val_loader, device, model)

                    # If loss decreases, store current model parameters and HP setting.
                    if val_loss < best_val_loss:
                        elapsed_patience = 0
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), models_folder + "/" + str(node_hidden_dim) + "_" + str(weight_decay) + "_" + str(lr))
                    else:
                        elapsed_patience += 1

                # Store optimal HP parameter setting.
                if best_val_loss < opt_val_loss:
                    opt_val_loss = best_val_loss
                    opt_node_hidden_dim = node_hidden_dim
                    opt_weight_decay = weight_decay
                    opt_lr = lr
    return opt_node_hidden_dim, opt_weight_decay, opt_lr


################################################################################

def get_window_scores(list_w_sizes,
                      dataset_name=None,
                      list_node_attr=None,
                      list_node_labels=None,
                      use_node_attr=False,
                      con_ext=False,
                      lc_context=False,
                      model=None,
                      device=None,
                      batch_size=None,
                      geometric_folder=None):
    """
    Predict whole site scores in a sliding window fashion, starting from
    first position with pos_ext on one side only, and ending at last position
    with pos_ext on upstream side. Several window sizes can also be given,
    resulting in position-wise scores averaged over all window scores at
    a certain position.

    """
    # Checks.
    assert list_w_sizes, "given list_w_sizes empty"
    dataset_folder = geometric_folder + "/" + dataset_name
    save_dataset_name = dataset_name
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
    g_idx = 0
    g_len = len(list_node_labels)
    c_w_sizes = len(list_w_sizes)

    # Set a maximum lowercase context extension for window prediction.
    lc_ext = con_ext
    if con_ext:
        if con_ext > 50:
            lc_ext = 50

    for w_size in list_w_sizes:
        # Extension left / right.
        extlr = int(w_size/2)
        # For each position in graph.
        for i in range(g_len):
            reg_s = i - extlr
            reg_e = i + extlr + 1
            # If lowercase context extension specified.
            if con_ext:
                reg_s = reg_s - con_ext
                reg_e = reg_e + con_ext
            if reg_e > g_len:
                reg_e = g_len
            if reg_s < 0:
                reg_s = 0

            if con_ext:
                center_pos = i + 1
                sl_node_labels = get_uc_lc_list_segment(list_node_labels,
                                                        center_pos,
                                                        vp_ext=extlr,
                                                        con_ext=lc_ext)
            else:
                sl_node_labels = list_node_labels[reg_s:reg_e]
            # Sublist of node attributes.
            sl_node_attr = []
            if use_node_attr:
                sl_node_attr = list_node_attr[reg_s:reg_e]
                assert len(sl_node_attr) == len(sl_node_labels), "sl_node_attr length != sl_node_labels length (%i != %i)" %(len(sl_node_attr), len(sl_node_labels))
            g_idx += 1
            sl_len = len(sl_node_labels)
            # Graph indicators.
            list_graph_indicators.extend([g_idx]*sl_len)
            # Nodel labels.
            list_all_node_labels.extend(sl_node_labels)
            # Node attributes.
            if use_node_attr:
                list_all_node_attributes.extend(sl_node_attr)
            # Edges.
            for n_idx_temp in range(sl_len):
                if n_idx_temp != (sl_len - 1):
                    list_all_edges.append((n_idx_temp + n_idx, n_idx_temp + 1 + n_idx))
                    list_all_edges.append((n_idx_temp + 1 + n_idx, n_idx_temp + n_idx))
            n_idx += sl_len
    assert list_graph_indicators, "list_graph_indicators empty"

    # Add dummy graph that contains all nucleotides.
    g_idx += 1
    if lc_context:
        list_graph_indicators.extend([g_idx]*8)
        list_all_node_labels.extend([1,2,3,4,5,6,7,8])
        if use_node_attr:
            list_all_node_attributes.extend([list_node_attr[0]]*8)
        for n_idx_temp in range(8):
            if n_idx_temp != 7:
                list_all_edges.append((n_idx_temp + n_idx, n_idx_temp + 1 + n_idx))
                list_all_edges.append((n_idx_temp + 1 + n_idx, n_idx_temp + n_idx))
    else:
        list_graph_indicators.extend([g_idx]*4)
        list_all_node_labels.extend([1,2,3,4])
        if use_node_attr:
            list_all_node_attributes.extend([list_node_attr[0]]*4)
        for n_idx_temp in range(4):
            if n_idx_temp != 3:
                list_all_edges.append((n_idx_temp + n_idx, n_idx_temp + 1 + n_idx))
                list_all_edges.append((n_idx_temp + 1 + n_idx, n_idx_temp + n_idx))

    f = open(raw_folder + "/" + save_dataset_name + "_graph_indicator.txt", 'w')
    f.writelines([str(e) + "\n" for e in list_graph_indicators])
    f.close()

    f = open(raw_folder + "/" + save_dataset_name + "_A.txt", 'w')
    f.writelines([str(e[0]) + ", " + str(e[1]) + "\n" for e in list_all_edges])
    f.close()

    f = open(raw_folder + "/" + save_dataset_name + "_node_labels.txt", 'w')
    f.writelines([str(e) + "\n" for e in list_all_node_labels])
    f.close()
    if use_node_attr:
        f = open(raw_folder + "/" + save_dataset_name + "_node_attributes.txt", 'w')
        f.writelines([s + "\n" for s in list_all_node_attributes])
        f.close()

    # Predict top sites.
    # Read in top sites dataset.
    dataset = TUDataset(dataset_folder, name=save_dataset_name, use_node_attr=use_node_attr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # Score top sites.
    scores = get_scores(loader, device, model,
                        min_max_norm=True)
    assert scores, "scores list empty"
    # Remove dummy prediction.
    scores = scores[:-1]
    assert g_len*c_w_sizes == len(scores), "length scores != length of list of node labels"
    all_scores = [scores[i:i + g_len] for i in range(0, len(scores), g_len)]
    scores_mean = list(np.mean(all_scores, axis=0))
    return scores_mean


################################################################################

def get_uc_lc_list_segment(seq, cp,
                           vp_ext=20,
                           uc2lc_label_dic=False,
                           con_ext=0):
    """
    Given a list (seq), a center position inside the list, a viewpoint
    extension value, and a context extension value: get the lowercase-
    uppercase-lowercase sequence segment.

    cp:
        1-based list position that marks the center of the segment.
    uc2lc_label_dic:
        Uppercase to lowercase labels mapping dictionary.

    >>> seq = [1,2,3,4,1,2,3,4]
    >>> cp = 4
    >>> vp_ext = 1
    >>> con_ext = 2
    >>> get_uc_lc_list_segment(seq, cp, vp_ext=vp_ext, con_ext=con_ext)
    [5, 6, 3, 4, 1, 6, 7]
    >>> cp = 1
    >>> vp_ext = 2
    >>> con_ext = 2
    >>> get_uc_lc_list_segment(seq, cp, vp_ext=vp_ext, con_ext=con_ext)
    [1, 2, 3, 8, 5]
    >>> cp = 8
    >>> vp_ext = 0
    >>> con_ext = 2
    >>> get_uc_lc_list_segment(seq, cp, vp_ext=vp_ext, con_ext=con_ext)
    [6, 7, 4]
    >>> cp = 4
    >>> vp_ext = 1
    >>> con_ext = 4
    >>> get_uc_lc_list_segment(seq, cp, vp_ext=vp_ext, con_ext=con_ext)
    [5, 6, 3, 4, 1, 6, 7, 8]
    >>> cp = 1
    >>> vp_ext = 0
    >>> con_ext = 0
    >>> get_uc_lc_list_segment(seq, cp, vp_ext=vp_ext, con_ext=con_ext)
    [1]

    """
    # Checks.
    assert seq, "given seq empty"
    assert cp, "invalid cp given"
    lseq = len(seq)
    assert cp <= lseq, "given cp > lseq"
    assert cp > 0, "given cp < 1"

    if not uc2lc_label_dic:
        uc2lc_label_dic = {1 : 5,
                           2 : 6,
                           3 : 7,
                           4 : 8}

    # Upstream extensions.
    usucs = cp - vp_ext - 1
    usuce = cp - 1
    uslcs = cp - vp_ext - con_ext - 1
    uslce = usucs
    if usucs < 0:
        usucs = 0
    if usuce < 1:
        usuce = 0
    if uslcs < 0:
        uslcs = 0
    if uslce < 1:
        uslce = 0

    # Downstream extensions.
    dsucs = cp
    dsuce = cp + vp_ext
    dslcs = dsuce
    dslce = cp + vp_ext + con_ext
    if dsucs >= lseq:
        dsucs = lseq
    if dsuce > lseq:
        dsuce = lseq
    if dslcs >= lseq:
        dslcs = lseq
    if dslce > lseq:
        dslce = lseq

    # Center position.
    cps = cp-1
    cpe = cp

    # Extract segments.
    usucseg = seq[usucs:usuce]
    uslcseg = seq[uslcs:uslce]
    dsucseg = seq[dsucs:dsuce]
    dslcseg = seq[dslcs:dslce]
    cpseg = seq[cps:cpe]

    # Change context labels to lowercase labels.
    for i,l in enumerate(uslcseg):
        new_l = uc2lc_label_dic[l]
        uslcseg[i] = new_l
    for i,l in enumerate(dslcseg):
        new_l = uc2lc_label_dic[l]
        dslcseg[i] = new_l

    # Give it to me.
    final_seg = uslcseg + usucseg + cpseg + dsucseg + dslcseg

    #print("USLC: %i %i" %(uslcs,uslce))
    #print("uslcseg:", uslcseg)
    #print("USUC: %i %i" %(usucs,usuce))
    #print("usucseg:", usucseg)
    #print("CP:   %i %i" %(cps, cpe))
    #print("cpseg:", cpseg)
    #print("DSUC: %i %i" %(dsucs,dsuce))
    #print("dsucseg:", dsucseg)
    #print("DSLC: %i %i" %(dslcs,dslce))
    #print("dslcseg:", dslcseg)
    return final_seg


################################################################################

def get_whole_site_scores(top_pos_list, args,
                          dataset_name=None,
                          list_node_attr=None,
                          list_node_labels=None,
                          use_node_attr=False,
                          model=None,
                          device=None,
                          batch_size=None,
                          geometric_folder=None):
    """
    Extract sites from a list of top positions, and predict whole site
    scores for sites centered on these positions. Use args.peak_ext and
    args.con_ext to define sites.

    """
    # Checks.
    assert top_pos_list, "given top_pos_list empty"
    dataset_folder = geometric_folder + "/" + dataset_name
    save_dataset_name = dataset_name
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
    g_idx = 0

    # Generate PyG data for top sites.
    for top_pos in top_pos_list:
        reg_s = top_pos - args.peak_ext - 1
        reg_e = top_pos + args.peak_ext
        if reg_e > len(list_node_labels):
            reg_e = len(list_node_labels)
        if reg_s < 0:
            reg_s = 0
        sl_node_labels = list_node_labels[reg_s:reg_e]
        sl_node_attr = []
        if use_node_attr:
            sl_node_attr = list_node_attr[reg_s:reg_e]
        g_idx += 1
        g_len = len(sl_node_labels)
        # Graph indicators.
        list_graph_indicators.extend([g_idx]*g_len)
        # Nodel labels.
        list_all_node_labels.extend(sl_node_labels)
        # Node attributes.
        if use_node_attr:
            list_all_node_attributes.extend(sl_node_attr)
        # Edges.
        for n_idx_temp in range(g_len):
            if n_idx_temp != (g_len - 1):
                list_all_edges.append((n_idx_temp + n_idx, n_idx_temp + 1 + n_idx))
                list_all_edges.append((n_idx_temp + 1 + n_idx, n_idx_temp + n_idx))
        n_idx += g_len

    f = open(raw_folder + "/" + save_dataset_name + "_graph_indicator.txt", 'w')
    f.writelines([str(e) + "\n" for e in list_graph_indicators])
    f.close()

    f = open(raw_folder + "/" + save_dataset_name + "_A.txt", 'w')
    f.writelines([str(e[0]) + ", " + str(e[1]) + "\n" for e in list_all_edges])
    f.close()

    f = open(raw_folder + "/" + save_dataset_name + "_node_labels.txt", 'w')
    f.writelines([str(e) + "\n" for e in list_all_node_labels])
    f.close()
    if use_node_attr:
        f = open(raw_folder + "/" + save_dataset_name + "_node_attributes.txt", 'w')
        f.writelines([s + "\n" for s in list_all_node_attributes])
        f.close()

    # Predict top sites.
    # Read in top sites dataset.
    dataset = TUDataset(dataset_folder, name=save_dataset_name, use_node_attr=use_node_attr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # Score top sites.
    scores = get_scores(loader, device, model,
                        min_max_norm=True)
    assert scores, "scores list empty"
    assert len(top_pos_list) == len(scores), "length scores != length top_pos_list"
    return scores


################################################################################

def get_scores_profile(dataset_name=None,
                       x=None, list_node_labels=None,
                       list_w_sizes=[3, 5, 7],
                       use_node_attr=False,
                       zero_sc_ends=False,
                       model=None, device=None, batch_size=None,
                       geometric_folder=None):

    """
    Calculate profile scores by outputting subgraphs in PyG format,
    read in and score them. Return mean profile scores in case of several
    window sizes given, otherwise single window size scores.

    """

    all_scores = []
    dataset_folder = geometric_folder + "/" + dataset_name
    save_dataset_name = dataset_name
    # Calculate profile scores for each window size.
    for w_size in list_w_sizes:
        # Create raw set containing window subgraphs.
        create_profile_dataset(save_dataset_name=save_dataset_name, x=x, list_node_labels=list_node_labels,
                               w_size=w_size, dataset_folder=dataset_folder)
        # Read in dataset, containing one site split into windows.
        dataset_w = TUDataset(dataset_folder, name=save_dataset_name, use_node_attr=use_node_attr)
        # Load the set.
        loader = DataLoader(dataset_w, batch_size=batch_size, shuffle=False)
        # Score each subgraph.
        scores = get_scores(loader, device, model,
                            min_max_norm=True)
        # Add scores at end to get full site length scores list.
        if zero_sc_ends:
            scores = [0]*int(w_size/2) + scores + [0]*int(w_size/2)
        else:
            scores = [scores[0]]*int(w_size/2) + scores + [scores[-1]]*int(w_size/2)
        all_scores.append(scores)
    scores_mean = list(np.mean(all_scores, axis=0))
    # Return mean scores (if only one win_size equals to win_size scores).
    return scores_mean


################################################################################

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

    for g_idx in range(len(list_node_labels)-w_size + 1):
        # graph indicators
        if x:
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
    if x:
        f = open(raw_folder + "/" + save_dataset_name + "_node_attributes.txt", 'w')
        f.writelines([s + "\n" for s in list_all_node_attributes])
        f.close()


################################################################################
