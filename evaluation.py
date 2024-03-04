import csv
import pickle
import obonet
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import average_precision_score as aupr
from sklearn.metrics import roc_curve, auc

import seaborn as sns
from matplotlib import pyplot as plt

plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

# go.obo pan 0629
go_graph = obonet.read_obo(open("./Datasets/go-basic.obo", 'r'))


def bootstrap(Y_true, Y_pred):
    n = Y_true.shape[0]
    idx = np.random.choice(n, n)

    return Y_true[idx], Y_pred[idx]


def load_test_prots(fn):
    proteins = []
    seqid_mtrx = []
    with open(fn, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            inds = row[1:]
            inds = np.asarray([int(i) for i in inds]).reshape(1, len(inds))
            proteins.append(row[0])
            seqid_mtrx.append(inds)

    return np.asarray(proteins), np.concatenate(seqid_mtrx, axis=0)


def load_go2ic_mapping(fn):
    goterm2ic = {}
    fRead = open(fn, 'r')
    for line in fRead:
        goterm, ic = line.strip().split()
        goterm2ic[goterm] = float(ic)
    fRead.close()

    return goterm2ic


def propagate_go_preds(Y_hat, goterms):
    go2id = {go: ii for ii, go in enumerate(goterms)}
    for goterm in goterms:
        if goterm in go_graph:
            parents = set(goterms).intersection(nx.descendants(go_graph,
                                                               goterm))
            for parent in parents:
                Y_hat[:, go2id[parent]] = np.maximum(Y_hat[:, go2id[goterm]],
                                                     Y_hat[:, go2id[parent]])

    return Y_hat


class Method(object):
    def __init__(self, method_name, pckl_fn):
        annot = pickle.load(open(pckl_fn, 'rb'))
        self.Y_true = annot['Y_true']
        self.Y_pred = annot['Y_pred']
        self.goterms = annot['goterms']
        # self.gonames = annot['gonames']
        self.proteins = annot['proteins']
        # self.ont = annot['ontology']
        self.ont = 'cc'
        self.method_name = method_name
        self._propagate_preds()
        if self.ont == 'ec':
            goidx = [i for i, goterm in enumerate(self.goterms) if goterm.find('-') == -1]
            self.Y_true = self.Y_true[:, goidx]
            self.Y_pred = self.Y_pred[:, goidx]
            self.goterms = [self.goterms[idx] for idx in goidx]
            self.gonames = [self.gonames[idx] for idx in goidx]

    def _propagate_preds(self):
        self.Y_pred = propagate_go_preds(self.Y_pred, self.goterms)

    def _cafa_go_aupr(self, labels, preds):
        # propagate goterms (take into account goterm specificity)
        # number of test proteins
        n = labels.shape[0]

        goterms = np.asarray(self.goterms)
        ont2root = {'bp': 'GO:0008150', 'mf': 'GO:0003674', 'cc': 'GO:0005575'}

        prot2goterms = {}
        for i in range(0, n):
            all_gos = set()
            for goterm in goterms[np.where(labels[i] == 1)[0]]:
                all_gos = all_gos.union(nx.descendants(go_graph, goterm))
                all_gos.add(goterm)
            all_gos.discard(ont2root[self.ont])
            prot2goterms[i] = all_gos

        # CAFA-like F-max predictions
        F_list = []
        AvgPr_list = []
        AvgRc_list = []
        thresh_list = []

        for t in range(1, 100):
            threshold = t / 100.0
            predictions = (preds > threshold).astype(np.int)

            m = 0
            precision = 0.0
            recall = 0.0
            for i in range(0, n):
                pred_gos = set()
                for goterm in goterms[np.where(predictions[i] == 1)[0]]:
                    pred_gos = pred_gos.union(nx.descendants(go_graph,
                                                             goterm))
                    pred_gos.add(goterm)
                pred_gos.discard(ont2root[self.ont])

                num_pred = len(pred_gos)
                num_true = len(prot2goterms[i])
                num_overlap = len(prot2goterms[i].intersection(pred_gos))
                if num_pred > 0 and num_true > 0:
                    m += 1
                    precision += float(num_overlap) / num_pred
                    recall += float(num_overlap) / num_true

            if m > 0:
                AvgPr = precision / m
                AvgRc = recall / n

                if AvgPr + AvgRc > 0:
                    F_score = 2 * (AvgPr * AvgRc) / (AvgPr + AvgRc)
                    # record in list
                    F_list.append(F_score)
                    AvgPr_list.append(AvgPr)
                    AvgRc_list.append(AvgRc)
                    thresh_list.append(threshold)

        F_list = np.asarray(F_list)
        AvgPr_list = np.asarray(AvgPr_list)
        AvgRc_list = np.asarray(AvgRc_list)
        thresh_list = np.asarray(thresh_list)

        return AvgRc_list, AvgPr_list, F_list, thresh_list

    def _function_centric_aupr(self, keep_pidx=None, keep_goidx=None):
        """ Compute functon-centric AUPR """
        if keep_pidx is not None:
            Y_true = self.Y_true[keep_pidx]
            Y_pred = self.Y_pred[keep_pidx]
        else:
            Y_true = self.Y_true
            Y_pred = self.Y_pred

        if keep_goidx is not None:
            tmp = []
            for goidx in keep_goidx:
                if Y_true[:, goidx].sum() > 0:
                    tmp.append(goidx)
            keep_goidx = tmp
        else:
            keep_goidx = np.where(Y_true.sum(axis=0) > 0)[0]

        print("### Number of functions =%d" % (len(keep_goidx)))

        Y_true = Y_true[:, keep_goidx]
        Y_pred = Y_pred[:, keep_goidx]

        # micro average
        micro_aupr = aupr(Y_true, Y_pred, average='micro')
        # macro average
        macro_aupr = aupr(Y_true, Y_pred, average='macro')

        # each function
        aupr_goterms = aupr(Y_true, Y_pred, average=None)

        return micro_aupr, macro_aupr, aupr_goterms

    def AUC(self, keep_pidx=None):
        if keep_pidx is not None:
            Y_true = self.Y_true[keep_pidx]
            Y_pred = self.Y_pred[keep_pidx]
        else:
            Y_true = self.Y_true
            Y_pred = self.Y_pred

        FPR, TPR, threshold = roc_curve(Y_true, Y_pred, pos_label=1)
        AUC = auc(FPR, TPR)
        return AUC

    def _protein_centric_fmax(self, keep_pidx=None):
        """ Compute protein-centric fmax """
        if keep_pidx is not None:
            Y_true = self.Y_true[keep_pidx]
            Y_pred = self.Y_pred[keep_pidx]
        else:
            Y_true = self.Y_true
            Y_pred = self.Y_pred

        # compute recall/precision
        Recall, Precision, Fscore, thresholds = self._cafa_go_aupr(Y_true,Y_pred)
        return Fscore, Recall, Precision, thresholds

    def fmax(self, keep_pidx):
        fscore, _, _, _ = self._protein_centric_fmax(keep_pidx=keep_pidx)

        return max(fscore)

    def macro_aupr(self, keep_pidx=None):
        _, macro_aupr, _ = self._function_centric_aupr(keep_pidx=keep_pidx)
        return macro_aupr

    def micro_aupr(self, keep_pidx=None):
        micro_aupr, _, _ = self._function_centric_aupr(keep_pidx=keep_pidx)
        return micro_aupr


def box_plot(methods, seqid_mtrx, palette, title, n_folds=10):
    # initialize perf dictionary
    names = [method.method_name for method in methods]
    seqids = [30, 40, 50, 70, 95]
    perfs = {name: np.zeros((len(seqids), n_folds)) for name in names}

    for method in methods:
        print("### Method=%s" % (method.method_name))
        for i in range(len(seqids)):
            print("### Seqid=%d" % (seqids[i]))
            idx = np.where(seqid_mtrx[:, i] == 1)[0]
            for j in range(n_folds):
                print("Fold=%d" % (j))
                bootstrap = np.random.choice(idx, len(idx))
                perfs[method.method_name][i, j] = method.fmax(bootstrap)
        print('\n\n')

    data = []
    for name in names:
        for i, seqid in enumerate(seqids):
            data.append([perfs[name][i], n_folds * [seqid], n_folds * [name]])
    data = np.concatenate(data, axis=1)

    # Call the function to create plot
    df = pd.DataFrame(columns=['perf', 'sequence_similarity', 'model'],data=data.T)
    df['perf'] = df['perf'].astype(float)
    df['sequence_similarity'] = df['sequence_similarity'].astype(int)

    plt.figure(figsize=(7.2, 7.2))
    boxprops = {'linewidth': 1}
    lineprops = {'color': 'k', 'linewidth': 1}

    boxplot_kwargs = {'boxprops': boxprops, 'medianprops': lineprops,'whiskerprops': lineprops, 'capprops': lineprops,'width': 0.8, 'palette': palette}
    sns.boxplot(x='sequence_similarity', y='perf', hue='model', data=df, **boxplot_kwargs)

    plt.style.use("seaborn-bright")
    plt.ylim([0.3, 0.70])
    plt.legend(fontsize=12, loc='upper left')
    plt.ylabel(r"$F_{max}$", fontsize=18)
    plt.xlabel("Maximum % sequence identity to training set", fontsize=18)
    plt.title(title, fontsize=20, weight='bold', pad=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(title + '_box.png', bbox_inches='tight')



def protein_centric_aupr_curves(methods, title, colors, prot_idx):
    plt.figure(figsize=(7.2, 7.2))
    f_scores = np.linspace(0.1, 0.9, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1.0)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('F1={0:0.1f}'.format(f_score), xy=(0.85, y[45] + 0.02),fontsize=10, color='gray', alpha=0.7)

    for i, method in enumerate(methods):
        name = method.method_name
        fscore, rec, pre, _ = method._protein_centric_fmax(prot_idx)
        idx = np.argmax(fscore)

        if i == 0:
            plt.plot(rec, pre, '-', label=name, color=colors[i])
        else:
            plt.plot(rec, pre, '-', label=name, color=colors[i])
        print(name, "Fmax=", fscore[idx])

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.legend(fontsize=12, loc='upper right')
    plt.savefig(title + '.png', bbox_inches='tight')




if __name__ == "__main__":
    methods = []
    methods.append(Method('PFresgo', 'MF_PFresGO_results.pckl'))
    #methods.append(Method('PFresgo', 'BP_PFresGO_results.pckl'))
    #methods.append(Method('PFresgo', 'CC_PFresGO_results.pckl'))

    test_prots, seqid_mtrx = load_test_prots('./Datasets/nrPDB-GO_2019.06.18_test.csv')
    prot_idx = np.where(seqid_mtrx[:, 4] == 1)[0]
    vals = {}
    for method in methods:
        micro_aupr, macro_aupr, _ = method._function_centric_aupr(prot_idx)
        auc = method.AUC
        fmax = method.fmax(prot_idx)








