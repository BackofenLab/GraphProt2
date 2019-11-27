from model_util import get_scores_profile
import torch
from MyNets import FunnelGNN
import random
import argparse


def main(args):
    random.seed(1)
    rna_seq = 'ATCAAACACTCCCCTCCTCTTTAGTCTTTGCGGCCACTGCAGATGGCACCTTCTCTGTGAAGCCAGCTTTACCTTTTCCCCAT'
    pretrain_model_path = "/home/dinh/data/projects/code/pycharm/graphprot2/models/cv_models/pretrain_model"
    geometric_folder = "/home/dinh/data/projects/code/pycharm/graphprot2/data/geometric_cv"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FunnelGNN(input_dim=args.input_dim, node_hidden_dim=args.node_hidden_dim, fc_hidden_dim=args.fc_hidden_dim,
                      out_dim=2).to(device)
    model.load_state_dict(torch.load(pretrain_model_path))

    scores = get_scores_profile(rna_name=args.rna_name, rna_seq=rna_seq, list_w_size=args.list_w_size, model=model,
                                device=device, batch_size=args.batch_size, geometric_folder=geometric_folder)
    print(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--rna_name", type=str, default="PUS", help="Name of RNA")
    parser.add_argument("--input_dim", type=int, default=4, help="Input dim")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--node_hidden_dim", type=int, default=128, help="Node hidden dim")
    parser.add_argument("--fc_hidden_dim", type=int, default=128, help="Number of units in hidden fully connected layer")
    parser.add_argument("--list_w_size", type=int, nargs='+', default=[3, 5, 7], help="List of window sizes")
    parser.add_argument("--out_dim", type=int, default=2, help="Output dim")
    args = parser.parse_args()

    print(args)
    main(args)