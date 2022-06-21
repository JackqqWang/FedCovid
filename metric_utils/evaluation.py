from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, cohen_kappa_score, \
roc_curve,  precision_recall_curve, auc, confusion_matrix, roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
import os
save_path_check = "save/see/"
isExist = os.path.exists(save_path_check)
if not isExist:
    os.makedirs(save_path_check)
def evaluate(y_classes, yhat_classes, yhat_probs, iter):
    # y_classes = np.argmax(y_test, axis=1)
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_classes, yhat_classes)
    # print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    preci = precision_score(y_classes, yhat_classes)
    # print("precision: {}".format(preci))
    # print('Precision: %f' % preci)
    # recall: tp / (tp + fn)
    recal = recall_score(y_classes, yhat_classes)
    # print("recall: {}".format(recal))
    # print('Recall: %f' % recal)
    # f1: 2 tp / (2 tp + fp + fn)
    mcc = matthews_corrcoef(y_classes,yhat_classes)
    # print('mcc: %f' % mcc)
    f1 = f1_score(y_classes, yhat_classes)
    # print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_classes, yhat_classes)
    # print('Cohens kappa: %f' % kappa)
    # ROC AUC
    # fpr, tpr, thresholds = roc_curve(y_classes, yhat_probs, pos_label=1)
    # roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score(y_classes, yhat_probs)
    # print(thresholds)
    # print("ROC AUC: ", roc_auc)
    # PR AUC
    precision, recall, thresholds = precision_recall_curve(y_classes, yhat_probs, pos_label=1)
    # plt.figure()
    # plt.plot(recall, precision)
    # plt.savefig(save_path_check + 'recall_precison_{}.png'.format(iter))
    # print("in pr-auc pre: {} recal: {}".format(precision, recall))
    # pr_auc = auc(recall, precision)
    pr_auc = metrics.auc(recall, precision)
    # print(thresholds)
    # print("PR AUC: ", pr_auc)
    # confusion matrix
    matrix = confusion_matrix(y_classes, yhat_classes)
    # print(matrix)

    return accuracy, preci, recal, f1, kappa, roc_auc, pr_auc, matrix

    