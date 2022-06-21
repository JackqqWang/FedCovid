import argparse
import torch
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="Transformer", type=str, help= "CNN, LSTM, Transformer, biLSTM")
    parser.add_argument("--noise", default = 1, type=int)
    parser.add_argument("--algorithm", default="FedAvg", type=str) # for debug
    # FedAvg FedProx, per_FedAvg
    parser.add_argument("--strategy", default="central_to_all", type=str) 
    # strategy: only_central, central_to_all, only_small, all_state
    parser.add_argument("--aggregation", default= "weighted_mean", type=str)
    parser.add_argument("--lr_stra", default= 1, type=int)
    parser.add_argument("--define_loss", default= "ours", type=str) # for debug
    parser.add_argument("--noise_weight", default = 0.75, type = float)
    parser.add_argument("--weight_list", nargs= '+', default=[1, 1, 1], type=int)
    parser.add_argument("--gamma", default = 2, type = float)
    parser.add_argument("--alpha", default = 0.25, type = float)
    parser.add_argument("--stddev", default = 0.2, type = float)
    parser.add_argument("--pos_margin", default = 0.004, type = float)
    parser.add_argument("--neg_margin", default = 0.001, type = float)
    parser.add_argument("--warmup_round", default = 200, type = int)
    parser.add_argument("--best_f1", default=False, type=bool)
    parser.add_argument("--communication_round", default = 400, type = int)
    parser.add_argument("--normal_local_epoch", default = 20, type = int)
    parser.add_argument("--big_state_epoch", default = 20, type = int)
    parser.add_argument("--small_state_epoch", default = 10, type = int)
    parser.add_argument("--active_rate", default = 0.4, type = float)
    parser.add_argument("--schedule", default="FL", type=str)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=None, type=int)
    parser.add_argument("--fenpitrain", default=True, type=bool)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    return args