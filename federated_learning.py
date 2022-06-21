import json
import time
import copy
# import argparse
import random
import torch
import os
from options import args_parser
from neural_nets import LSTM_iqvia_paper, Transformer_iqvia_paper, CNN_iqvia_paper, biLSTM_iqvia_paper
import numpy as np
from distributed_training_utils import Client, Server
import experiment_manager as xpm
import default_hyperparameters as dhp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter
from data_utils.data_slover import get_data_loaders
import pandas as pd
random.seed(1023)
np.random.seed(1023)
torch.manual_seed(1023)
torch.cuda.manual_seed(1023)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


print("Torch Version: ", torch.__version__)
args = args_parser()

# with open(os.path.join('config', 'federated_learning.json')) as data_file:
#     experiments_raw = json.load(data_file)[args.schedule]

# hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
# experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

# def run_experiments(experiments):
    # print("warm up round: {}".format(args.warmup_round))
    # print("communication round: {}".format(args.communication_round))
    # print("big state training epoch: {}".format(args.big_state_epoch))
    # print("small state training epoch: {}".format(args.small_state_epoch))
    # print("active client rate: {}".format(args.active_rate))
    # print("aggregation approach: {}".format(args.aggregation))
    # print("cuda device: {}".format(args.device))
    # print("loss: {}".format(args.define_loss))
    # print("algorithm: {}".format(args.algorithm))

    # for xp_count, xp in enumerate(experiments):
    #     hp = dhp.get_hp(xp.hyperparameters)
    #     xp.prepare(hp)
        
        # print("xp:")
        # print(xp)

        # Load the Data and split it among the Clients
        # client_loaders, train_loader, test_loader, val_loader, stats = get_data_loaders(hp)
print("warm up round: {}".format(args.warmup_round))
print("communication round: {}".format(args.communication_round))
print("big state training epoch: {}".format(args.big_state_epoch))
print("small state training epoch: {}".format(args.small_state_epoch))
print("active client rate: {}".format(args.active_rate))
print("aggregation approach: {}".format(args.aggregation))
print("cuda device: {}".format(args.device))
print("loss: {}".format(args.define_loss))
print("algorithm: {}".format(args.algorithm))
print("strategy: {}".format(args.strategy))
print("backbone: {}".format(args.backbone))
print("lr strategy enable: {}".format(args.lr_stra))
client_loaders, train_loader, test_loader, val_loader, stats = get_data_loaders()


# Instantiate Clients and Server with Neural Net
# categorical_embedding_sizes = [(2, 1), (3, 2), (8801, 50)]
categorical_embedding_sizes = [(2, 1), (3, 2)]
n_lstm = 1
if args.backbone == "LSTM":
    net_model = LSTM_iqvia_paper(categorical_embedding_sizes, 1, 2, [100, 80], p=0.1, n_lstm=n_lstm, device=args.device, noise=args.noise).to(args.device)
elif args.backbone == "Transformer":
    net_model = Transformer_iqvia_paper(categorical_embedding_sizes, 1, 2, [100, 80], p=0.1, device=args.device, noise=args.noise).to(args.device)
elif args.backbone == "biLSTM":
    net_model = biLSTM_iqvia_paper(categorical_embedding_sizes, 1, 2, [100, 80], p=0.1, device=args.device, noise = args.noise).to(args.device)
elif args.backbone == "CNN":
    net_model = CNN_iqvia_paper(categorical_embedding_sizes, 1, 2, [100, 80], p=0.1, device=args.device, noise = args.noise).to(args.device)
clients = [Client(client_loader, val_loader, net_model.to(args.device), i, algorithm=args.algorithm, define_loss= args.define_loss, lr_stra = args.lr_stra, pos_margin= args.pos_margin, neg_margin = args.neg_margin, gamma = args.gamma, alpha = args.alpha)
            for i, client_loader in enumerate(client_loaders)]
server = Server(None, test_loader, net_model.to(args.device), stats)




central_clients  = [clients[2], clients[17], clients[4],
                    clients[18], clients[24], clients[9],  clients[12], clients[28], clients[1], clients[5]]
small_clients = [clients[i] for i in range(len(clients)) if clients[i] not in central_clients]
# Start Distributed Training Process
print("Start Distributed Training..")
t1 = time.time()
server_test_round_f1 = []
server_test_round_accuracy = []
server_test_round_kappa = []
server_test_round_precision = []
server_test_round_pr_auc = []
server_test_round_roc_auc = []
server_test_round_recall = []


best_f1 = 0
best_round = 1
for c_round in range(1, args.communication_round + 1):
    print("Starting local update Round {}".format(c_round))
    
    if args.strategy == "only_central":
        participating_clients = central_clients
        print("strategy is {}".format(args.strategy))
        participating_clients = central_clients
        for client in participating_clients:
            client.synchronize_with_server(server)
            client.compute_weight_update(args.big_state_epoch)
    elif args.strategy == 'only_small':
        participating_clients = small_clients
        for client in participating_clients:
            client.synchronize_with_server(server)
            client.compute_weight_update(args.small_state_epoch)

    elif args.strategy == 'central_to_all':
        print("strategy is {}".format(args.strategy))
        if c_round <= args.warmup_round:
            participating_clients = central_clients
            for client in participating_clients:
                client.synchronize_with_server(server)
                client.compute_weight_update(args.big_state_epoch)

        else:
            participating_clients = random.sample(clients, int(len(clients) * args.active_rate))
            for client in participating_clients:
                if client in central_clients:
                    client.synchronize_with_server(server)
                    client.compute_weight_update(args.big_state_epoch)

                else:
                    client.synchronize_with_server(server)

    elif args.strategy == 'all_state':
        participating_clients = random.sample(clients, int(len(clients) * args.active_rate))
        for client in participating_clients:
            client.synchronize_with_server(server)
            client.compute_weight_update(args.normal_local_epoch)


    server.aggregate_weight_updates(participating_clients, aggregation= args.aggregation)
    print("Communication Round {} Finished".format(c_round))
    print("Server Evaluate...")
    accuracy, preci, recal, f1, kappa, roc_auc, pr_auc, matrix = server.evaluate(iter=c_round)
    print('Accuracy: %f' % accuracy)
    print('Precision: %f' % preci)
    print('Recall: %f' % recal)
    print('F1 score: %f' % f1)
    print('Cohens kappa: %f' % kappa)
    print("ROC AUC: ", roc_auc)
    print("PR AUC: ", pr_auc)
    print(matrix)

    if f1>=best_f1:
        best_f1 = f1
        best_model_state  = copy.deepcopy(server.model.state_dict())
        best_round = c_round

    server_test_round_f1.append(f1)
    server_test_round_accuracy.append(accuracy)
    server_test_round_kappa.append(kappa)
    server_test_round_precision.append(preci)
    server_test_round_pr_auc.append(pr_auc)
    server_test_round_roc_auc.append(roc_auc)
    server_test_round_recall.append(recal)
    # Timing
    total_time = time.time() - t1
    avrg_time_per_c_round = total_time / c_round
    e = int(avrg_time_per_c_round * (args.communication_round - c_round))
    print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60),
        "[{:.2f}%]\n".format(c_round / args.communication_round * 100))

save_path = "save/lr_{}_backbone_{}_active_{}_weight_list_{}_alg_{}_loss_{}_agg_{}_stra_{}_warm_up_{}_big_local_{}_small_local_{}_comm_{}/"\
                                 .format(args.lr_stra, args.backbone, str(args.active_rate), str(args.weight_list), args.algorithm, args.define_loss, args.aggregation, args.strategy, args.warmup_round, args.big_state_epoch, args.small_state_epoch, args.communication_round)
isExist = os.path.exists(save_path)
if not isExist:
    os.makedirs(save_path)
for j in range(len(clients)):
    # print("client {}:".format(j))
    plt.figure(j)
    plt.plot([i for i in range(len(clients[j].val_f1_list))], clients[j].val_f1_list)
    plt.title("client datasize {}".format(len(clients[j].train_loader.dataset)))
    plt.savefig(save_path + 'f1_client_{}.png'.format(j))
    
print("******loss: {} stra: {} agg: {} last 10 round average*********".format(args.define_loss, args.strategy, args.aggregation))
print('Last 10 Average test Accuracy in server: %f' % np.mean(server_test_round_accuracy[-10:]))
print('Last 10 Average test Precision in server: %f' % np.mean(server_test_round_precision[-10:]))
print('Last 10 Average test Recall in server: %f' % np.mean(server_test_round_recall[-10:]))
print('Last 10 Average test F1 score in server: %f' % np.mean(server_test_round_f1[-10:]))
print('Last 10 Average test Cohens kappa in server: %f' % np.mean(server_test_round_kappa[-10:]))
print("Last 10 Average test ROC AUC in server:%f ", np.mean(server_test_round_roc_auc[-10:]))
print("Last 10 Average test PR AUC in server:%f ", np.mean(server_test_round_pr_auc[-10:]))
print("******best server test f1 score*********")
print("best f1 is : {}".format(best_f1))
print("best round {}".format(best_round))


torch.save(best_model_state, save_path+"server_optimal_round_{}_str_{}.pt".format(best_round, args.strategy))

middle_result = pd.DataFrame(
{'F1': server_test_round_f1,
'Kappa': server_test_round_kappa,
'PRAUC': server_test_round_pr_auc,
'Accuracy':server_test_round_accuracy,
'Precision':server_test_round_precision,
'ROCAUC':server_test_round_roc_auc,
})
middle_result.to_csv(save_path + 'Fedavg_round_{}_str_{}.csv'.format(args.communication_round, args.strategy))
plt.figure()
plt.plot([i for i in range(len(server_test_round_f1))], server_test_round_f1, label = 'test f1')
plt.xlabel("communication round")
plt.ylabel("f1")
plt.title("server test f1")
plt.savefig(save_path+'server_round_{}_str_{}.png'.format(args.communication_round, args.strategy))

del server
clients.clear()
torch.cuda.empty_cache()

if __name__ == "__main__":
    print("warm up round: {}".format(args.warmup_round))
    print("communication round: {}".format(args.communication_round))
    print("big state training epoch: {}".format(args.big_state_epoch))
    print("small state training epoch: {}".format(args.small_state_epoch))
    print("active client rate: {}".format(args.active_rate))
    print("aggregation approach: {}".format(args.aggregation))
    print("cuda device: {}".format(args.device))
    print("loss: {}".format(args.define_loss))
    print("algorithm: {}".format(args.algorithm))
    print("strategy: {}".format(args.strategy))
    print("backbone: {}".format(args.backbone))
    print("lr strategy enable: {}".format(args.lr_stra))