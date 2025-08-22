import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve,cohen_kappa_score,classification_report

def cal_mcc(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
def cross_entropy_loss(outputs, target_onehot):
    loss1 = -torch.sum(target_onehot * F.log_softmax(outputs, dim=1))
    return loss1
def sensitivity(y_true, y_pred):
    """
    y_true, y_pred: 0/1 数组
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp / (tp + fn) if (tp + fn) > 0 else 0.0
def precision(y_true, y_pred):
    """
    y_true, y_pred: 0/1 数组
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    return tp / (tp + fp) if (tp + fp) > 0 else 0.0
def evaluation(en1, en2,fuse1,fuse2,decoder, dataloader,device,_class_=None):
    batch_size = 1
    en1.eval()
    en2.eval()
    fuse1.eval()
    fuse2.eval()
    decoder.eval()
    gt_list_sp= []
    pr_list_sp = []
    with torch.no_grad():
        for img, pet, label,id in dataloader:

            img ,pet = img.to(device), pet.to(device)
            input1 = en1(img)[0]
            input2 = en2(pet)[0]
            output1=fuse1(input1,input2)[0]
            output2=fuse2(input2,input1)[0]

            outputs = decoder((output1 + output2)/2)
            #outputs = decoder((input1+input2)/2)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            out=predicted.to('cpu').detach().numpy()

            pr_list_sp.append(np.max(out))
            gt_list_sp.append(label.item())
        print(classification_report(gt_list_sp, pr_list_sp, labels=[0, 1],target_names=['class0', 'class1']))
        macro_f1 = round(f1_score(gt_list_sp, pr_list_sp, average='macro'),5)
        acc = round(accuracy_score(gt_list_sp, pr_list_sp), 5)
        precis = round(precision(gt_list_sp, pr_list_sp), 5)
        kappa = round(cohen_kappa_score(gt_list_sp, pr_list_sp),5)
        sen = round(sensitivity(gt_list_sp, pr_list_sp),5)
        mcc = round(cal_mcc(gt_list_sp, pr_list_sp),5)

        # thresh = return_best_thr(gt_list_sp, pr_list_sp)
        # acc = round(accuracy_score(gt_list_sp, pr_list_sp >= thresh), 4)
        # f1 = round(f1_score(gt_list_sp, pr_list_sp >= thresh), 4)
        #auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    return acc,macro_f1,kappa,sen,precis,mcc



def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr