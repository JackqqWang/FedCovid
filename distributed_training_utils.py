import torch
import torch.optim as optim
import compression_utils as comp
import torch.nn.functional as F
from optimizer_utils.MySGD import MySGD
from metric_utils.evaluation import evaluate
import matplotlib.pyplot as plt
from pytorch_metric_learning.reducers import MeanReducer
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from options import args_parser
from pytorch_metric_learning import losses
from pytorch_metric_learning import distances, losses, miners, reducers

args = args_parser()
device = args.device


def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def copy_decoder(target, source):
    for name in target:
        if "down_convs" not in name:
            target[name].data = source[name].data.clone()


def add(target, source):
    for name in target:
        target[name].data += source[name].data.clone()


def scale(target, scaling):
    for name in target:
        target[name].data = scaling * target[name].data.clone()


def subtract(target, source):
    for name in target:
        target[name].data -= source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def add_subtract(target, minuend, subtrahend):
    for name in target:
        target[name].data = target[name].data + minuend[name].data.clone() - subtrahend[name].data.clone()


def average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()
        
def weighted_average(target, sources, weights):
    for name in target:
        summ = torch.sum(weights)
        n = len(sources)
        modify = [weight / summ * n for weight in weights]
        target[name].data = torch.mean(torch.stack([m * source[name].data for source, m in zip(sources, modify)]),
                                       dim=0).clone()

# def average_decoder(target, sources):
#     for name in target:
#         if "down_convs" not in name:
#             target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()

# def momentum_update(target, sources, weights):


def weighted_average_decoder(target, sources, weights):
    for name in target:
        if "down_convs" not in name:
            summ = torch.sum(weights)
            n = len(sources)
            modify = [weight / summ * n for weight in weights]
            target[name].data = torch.mean(torch.stack([m * source[name].data for source, m in zip(sources, modify)]),
                                           dim=0).clone()


def weighted_weights(target, source, alpha=0.25):
    for name in target:
        target[name].data = alpha * target[name].data.clone() + (1 - alpha) * source[name].data.clone()


def majority_vote(target, sources, lr):
    for name in target:
        mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
        target[name].data = (lr * mask).clone()


def compress(target, source, compress_fun):
    """compress_fun : a function f : tensor (shape) -> tensor (shape)"""
    for name in target:
        target[name].data = compress_fun(source[name].data.clone())


def cal_proximal_term(minuend, subtrahend):
    proximal_term = 0.0
    for name in minuend:
        proximal_term += (minuend[name].data.clone() - subtrahend[name].data.clone()).norm(2)
    return proximal_term


class DistributedTrainingDevice(object):
    def __init__(self, train_loader, test_loader, model):
        # self.xp = experiment
        self.train_loader = train_loader
        self.test_loader = test_loader
        # self.val_loader = val_loader
        self.model = model


class Client(DistributedTrainingDevice):

    def __init__(self, train_loader, val_loader, model, num_id, algorithm="FedAvg", define_loss = 'noise', lr_stra = True, pos_margin = 0.5, neg_margin = 1, gamma = 0, alpha = 1):
        super().__init__(train_loader, val_loader, model)

        self.id = num_id
        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW_compressed = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.A = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.n_params = sum([T.numel() for T in self.W.values()])
        self.bits_sent = []
        self.val_f1_list = []
        self.val_loader = val_loader
        self.define_loss = define_loss
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.gamma = gamma
        self.alpha = alpha
        self.lr_stra = lr_stra


        self.algorithm = algorithm
        if self.algorithm == 'per_FedAvg':
            self.optimizer = MySGD(self.model.parameters(), 0.001)
        else:
            if lr_stra:
                if len(self.train_loader.dataset) > 200:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), 0.005)
                else:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), 0.001)
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), 0.001)

        # State
        self.epoch = 0
        self.train_loss = 0.0


    def synchronize_with_server(self, server):

        copy(target=self.W, source=server.W)
    
    def train_cnn_with_early_stop(self, iterations, mu=0.5, algorithm='FedAvg', noise = False):
        running_loss = 0.0
        w = torch.Tensor([1.0, 1.0]).to(device)
        best_f1 = 0
        for i in range(iterations):
            self.model.train()
            self.epoch += 1
            for j, (x, y) in enumerate(self.train_loader):
                if x.shape[0] <= 1:
                    continue
                x, y = x.to(device), y.to(device).long()
                self.optimizer.zero_grad()
                if noise:
                    # y_, y_prime = self.model(x)
                    y_, y_prime_list = self.model(x)
                else:
                    y_ = self.model(x)

                if self.define_loss == 'ours':
                    distance = distances.CosineSimilarity()
                    reducer = reducers.ThresholdReducer(low=0)
                    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
                    mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")

                    prediction_final = []
                    ground_final = []
                    pos_prediction = []
                    posi_ground = []
                    for i in range(y.size(0)):
                        if int(y[i].data.cpu().numpy()) == 1:
                            pos_prediction.append(y_[i])
                            posi_ground.append(y[i])
                            posi_ground.append(y[i])
                            posi_ground.append(y[i])
                            posi_ground.append(y[i])
                            posi_ground.append(y[i])
                            prediction_final.append(y_[i])
                            for item in y_prime_list:
                                prediction_final.append(item[i])
                                pos_prediction.append(item[i])
                            ground_final.append(y[i]) 
                            ground_final.append(y[i])
                            ground_final.append(y[i])
                            ground_final.append(y[i])
                            ground_final.append(y[i])
                        else:
                            prediction_final.append(y_[i])
                            ground_final.append(y[i]) 

                    prediction_final = torch.stack(prediction_final, dim=0)
                    ground_final = torch.stack(ground_final, dim=0)
                    if len(pos_prediction) == 0:
                        loss = args.weight_list[0] * F.cross_entropy(prediction_final, ground_final, w) + \
                         args.weight_list[1]*0 + \
                         args.weight_list[2] * loss_func(prediction_final, ground_final, mining_func(prediction_final, ground_final))
                    if len(pos_prediction) > 0:
                        pos_prediction = torch.stack(pos_prediction, dim=0)
                        posi_ground = torch.stack(posi_ground, dim=0)
                        loss = args.weight_list[0] * F.cross_entropy(prediction_final, ground_final, w) + \
                         args.weight_list[1]*F.cross_entropy(pos_prediction, posi_ground, w) + \
                         args.weight_list[2] * loss_func(prediction_final, ground_final, mining_func(prediction_final, ground_final))

                    loss = loss / sum(args.weight_list)
                    proximal_term = cal_proximal_term(self.W, self.W_old)
                    loss += mu * proximal_term / 2
                elif self.define_loss == "normal":
                    loss = F.cross_entropy(y_, y, w)
                else:
                    print("undefined loss - STOP")
                if algorithm == 'FedProx':
                    proximal_term = cal_proximal_term(self.W, self.W_old)
                    loss += mu * proximal_term / 2
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            _, _, _, f1, _, _, _, _ = self.validation(0)
            if args.best_f1 == True:
                if f1 >= best_f1:
                    best_f1 = f1
                    best_model = self.model
                    

            self.val_f1_list.append(f1)
        if args.best_f1 == True:
            self.model = best_model

        running_loss /= len(self.train_loader)
        return running_loss / iterations

    def validation(self, iter):
        self.model.eval()
        test_loss = 0
        pred_list = []
        label_list = []
        prob_list = []
        for idx, (data, target) in enumerate(self.val_loader):
            data, target = data.to(device), target.to(device)
            if args.noise:
                log_probs, _ = self.model(data)
            else:
                log_probs = self.model(data)
            test_loss += F.cross_entropy(log_probs, target).item()
            y_prob, y_pred = log_probs.data.max(1, keepdim=True)
            pred_list.append(y_pred.cpu())
            prob_list.append(y_prob.cpu())
            label_list.append(target.cpu())

        pred_all = pred_list[0]
        prob_all = prob_list[0]
        label_all = label_list[0]
        for i in range(len(pred_list) - 1):
            pred_all = torch.cat((pred_all, pred_list[i + 1]), dim=0)
            prob_all = torch.cat((prob_all, prob_list[i + 1]), dim=0)
            label_all = torch.cat((label_all, label_list[i + 1]), dim=0)
        accuracy, preci, recal, f1, kappa, roc_auc, pr_auc, matrix = evaluate(label_all, pred_all, prob_all, iter)
        return accuracy, preci, recal, f1, kappa, roc_auc, pr_auc, matrix 


    def per_FedAvg(self, iterations, beta):
        running_loss = 0.0
        # w = torch.Tensor([1.0, 16.0]).to(device)
        temp_W = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        for i in range(iterations):
            train_loader = iter(self.train_loader)
            self.epoch += 1
            copy(temp_W, self.W)

            x, y = next(train_loader)
            x, y = x.to(device), y.to(device).long()
            self.optimizer.zero_grad()
            y_ = self.model(x)
            loss = F.cross_entropy(y_, y)
            loss.backward()
            self.optimizer.step()
            copy(self.W, temp_W)
            self.optimizer.step(beta=beta)

            running_loss += loss

        return running_loss / iterations

    def compute_weight_update(self, iterations=1):


        self.model.train()

        copy(target=self.W_old, source=self.W)
        if self.algorithm == 'FedAvg' or self.algorithm == 'FedProx':
            self.train_loss = self.train_cnn_with_early_stop (iterations, mu=0.5, algorithm = self.algorithm, noise = args.noise)

        elif self.algorithm == 'per_FedAvg':
            self.train_loss = self.per_FedAvg(iterations, beta=0.05)

        print("Training loss at epoch {} of Client {} is {:3f}".format(self.epoch, self.id, self.train_loss))

        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
    

    def compress_weight_update_up(self, compression=None, accumulate=False, count_bits=False):

        if accumulate and compression[0] != "none":

            add(target=self.A, source=self.dW)
            compress(target=self.dW_compressed, source=self.A, compress_fun=comp.compression_function(*compression))
            subtract(target=self.A, source=self.dW_compressed)

        else:
            compress(target=self.dW_compressed, source=self.dW, compress_fun=comp.compression_function(*compression))

        if count_bits:
            self.bits_sent += [comp.get_update_size(self.dW_compressed, compression)]





class Server(DistributedTrainingDevice):

    def __init__(self, train_loader, val_loader, model, stats):
        super().__init__(train_loader, val_loader, model)

        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.dW_compressed = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        self.A = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        self.n_params = sum([T.numel() for T in self.W.values()])
        self.bits_sent = []

        self.client_sizes = torch.Tensor(stats["split"]).to(device)


    def aggregate_weight_updates(self, clients, aggregation="mean"):

        if aggregation == "mean":

            average(target=self.W, sources=[client.W for client in clients])
        elif aggregation == "weighted_mean":
            weighted_average(target=self.W, sources=[client.W for client in clients],
                             weights=torch.stack([torch.log(self.client_sizes[client.id]) for client in clients]))
    

    def compress_weight_update_down(self, compression=None, accumulate=False, count_bits=False):
        if accumulate and compression[0] != "none":
            add(target=self.A, source=self.dW)
            compress(target=self.dW_compressed, source=self.A, compress_fun=comp.compression_function(*compression))
            subtract(target=self.A, source=self.dW_compressed)

        else:
            compress(target=self.dW_compressed, source=self.dW, compress_fun=comp.compression_function(*compression))

        add(target=self.W, source=self.dW_compressed)
        if count_bits:
            self.bits_sent += [comp.get_update_size(self.dW_compressed, compression)]

    def evaluate(self, iter):
        self.model.eval()
        test_loss = 0
        pred_list = []
        label_list = []
        prob_list = []
        for idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(device), target.to(device)
            if args.noise:
                log_probs, _ = self.model(data)
            else:
                log_probs = self.model(data)
            test_loss += F.cross_entropy(log_probs, target).item()
            y_prob, y_pred = log_probs.data.max(1, keepdim=True)
            pred_list.append(y_pred.cpu())
            prob_list.append(y_prob.cpu())
            label_list.append(target.cpu())

        pred_all = pred_list[0]
        prob_all = prob_list[0]
        label_all = label_list[0]
        for i in range(len(pred_list) - 1):
            pred_all = torch.cat((pred_all, pred_list[i + 1]), dim=0)
            prob_all = torch.cat((prob_all, prob_list[i + 1]), dim=0)
            label_all = torch.cat((label_all, label_list[i + 1]), dim=0)
        accuracy, preci, recal, f1, kappa, roc_auc, pr_auc, matrix = evaluate(label_all, pred_all, prob_all, iter)
        return accuracy, preci, recal, f1, kappa, roc_auc, pr_auc, matrix 


