import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def cosine_sim(features, proto):
    # feat has shape (B,C,H,W), proto has shape (PHW,C)
    feat = torch.nn.functional.normalize(features, p=2, dim=1)
    proto = torch.nn.functional.normalize(proto, p=2, dim=1)
    if features.ndim == 2: # case where another proto is givent as features
        sim = torch.einsum('Bc,Pc->BP', feat, proto)
    elif features.ndim == 3:
        sim = torch.einsum('bcn, Pc->bnP', feat, proto)
    else:
        sim = torch.einsum('bcHW,Pc->bHWP', feat, proto) # B,H,W,PHW
    return sim

def compute_and_save_auroc(gt, pred, path=None, title=''):
    fpr, tpr, _ = roc_curve(gt, pred)
    auroc = auc(fpr, tpr)

    if path is not None:
        plt.figure()
        plt.style.use('ggplot')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('Sensivity')
        plt.title(f'{title}. AUROC: {np.round(auroc, 3)}')
        plt.savefig(path)
        plt.clf()
        plt.cla()
        plt.close()

    return auroc

def compute_and_save_auprc(gt, pred, path, title=''):
    assert path.endswith('.png') or path.endswith('.jpg')

    prec, recall, _ = precision_recall_curve(gt, pred)
    auprc = auc(recall, prec)

    plt.figure()
    plt.style.use('ggplot')
    plt.plot(recall, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title}. AUPRC: {np.round(auprc, 3)}')
    plt.savefig(path)
    plt.clf()
    plt.cla()
    plt.close()

    return auprc
