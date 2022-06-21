import torch
import numpy as np
from tensorboardX import SummaryWriter
from neural_nets import LSTM_iqvia, LSTM_iqvia_paper
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_utils.data_slover import get_iqvia, data_augmentation
from data_utils.data_module import CustomImageDataset
from metric_utils.evaluation import evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(net_g, data_loader, iter=0, writer=None):
    net_g.eval()
    test_loss = 0
    pred_list=[]
    label_list=[]
    prob_list=[]
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_prob, y_pred = log_probs.data.max(1, keepdim=True)
        pred_list.append(y_pred.cpu())
        prob_list.append(y_prob.cpu())
        label_list.append(target.cpu())

    pred_all=pred_list[0]
    prob_all=prob_list[0]
    label_all=label_list[0]
    for i in range(len(pred_list)-1):
        pred_all=torch.cat((pred_all,pred_list[i+1]),dim=0)
        prob_all = torch.cat((prob_all, prob_list[i + 1]), dim=0)
        label_all=torch.cat((label_all,label_list[i+1]),dim=0)
    accuracy, preci, recal, mcc, f1, kappa, roc_auc, pr_auc, matrix = evaluate(label_all, pred_all, prob_all)
    if writer is not None:
        writer.add_scalar('Val/Accuracy', accuracy, iter)
        writer.add_scalar('Val/Precision', preci, iter)
        writer.add_scalar('Val/Recall', recal, iter)
        writer.add_scalar('Val/MCC', mcc, iter)
        writer.add_scalar('Val/f1', f1, iter)
        writer.add_scalar('Val/kappa', kappa, iter)
        writer.add_scalar('Val/roc_auc', roc_auc, iter)
        writer.add_scalar('Val/pr_auc', pr_auc, iter)


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = get_iqvia()
    x_train, y_train = data_augmentation(x_train, y_train, method='up_sampling')
    train_loader = torch.utils.data.DataLoader(CustomImageDataset(x_train, y_train), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test), batch_size=64, shuffle=False)

    categorical_embedding_sizes = [(2, 1), (3, 2), (8801, 50)]
    net_model = LSTM_iqvia_paper(categorical_embedding_sizes, 1, 2, [100, 50], p=0.4, n_lstm=2).to(device)

    optimizer = torch.optim.Adam(net_model.parameters(), lr=1e-3)
    writer = SummaryWriter(comment=' Centralized Training with up_sampling')
    w = torch.Tensor([1.0, 1.0]).to(device)
    print("Start training..")

    for iter in range(200):
        batch_loss = []
        net_model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device).long()
            optimizer.zero_grad()
            pred = net_model(x)
            loss = F.cross_entropy(pred, y, w)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        avg_loss = sum(batch_loss)/len(batch_loss)
        print("Loss at epoch {} is {:.3f}".format(iter, avg_loss))
        writer.add_scalar('Loss/Train', avg_loss, iter)
        if iter % 5 == 0:
            test(net_model, test_loader, iter, writer)
    writer.close()
