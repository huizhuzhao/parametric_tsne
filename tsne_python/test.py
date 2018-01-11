# encoding: utf-8
# author: huizhu
# created time: 2018年01月10日 星期三 11时48分11秒

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from foundation.datasets import dataset_from_mols
import tsne


def egfr_test():
    sdf_dir = '/home/xtalpi/datasets/mol_data/drugbank/sdf3D/'
    active_smi = '/home/xtalpi/datasets/mol_data/dude/egfr/actives_final.ism'
    decoys_smi = '/home/xtalpi/datasets/mol_data/dude/egfr/decoys_final.ism'
    sdf_files = [x for x in os.listdir(sdf_dir) if x.endswith('sdf')]
    sdf_files = [os.path.join(sdf_dir, f) for f in sdf_files]

    df_active = pd.read_csv(active_smi, sep=' ', names=['smiles', 'id', 'chemblid'])
    df_decoys = pd.read_csv(decoys_smi, sep=' ', names=['smiles', 'id', 'chemblid'])

    n2 = 10000
    smiles_list = df_active['smiles'].tolist() + df_decoys['smiles'].tolist()[:n2]
    labels = [1] * df_active.shape[0] + [0] * n2
    print(len(smiles_list), len(labels))

    dataset = dataset_from_mols(smiles_list, featurizer_type='morgan', transformer_type='morgan', batch_size=32)

    data = np.asarray(dataset.data[0], dtype=np.float32)

    Y = tsne.tsne(data, 2, 50, 20.)
    print(data.shape, Y.shape)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.show()


def pca(X, no_dims=50):
    """
    (n, d) = X.shape
    """
    X = X - np.mean(X, axis=0, keepdims=True)
    (eig_value, eig_vector) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, eig_vector[:, 0:no_dims])

    return Y


def main():
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")

    no_dims = 2
    initial_dims = 50
    perplexity = 30.0

    X = pca(X)
    print(X.shape)








if __name__ == '__main__':
    main()

