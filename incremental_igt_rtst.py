import pandas as pd
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
import pickle
from datetime import datetime
from sklearn.datasets import load_iris

from src.Node import Node
from src.NCMTree import NCMTree
from src.NCMForest import NCMForest
from src.NCMGridSearch import grid_search_rapport_njobs, grid_search_rapport
from headers.NCMClassifier import NCMClassifier
from headers.utils import *
from sklearn.datasets import make_classification
import time
from sklearn.metrics import classification_report
from src.NCMGridSearch import incremental_grid_search_rapport_njobs

if __name__ == '__main__':
    path_manu_X_train = "../../cls_datasets-master/manu_dataset_label_encoding/data_train_cnn_pca.bin"
    path_manu_y_train = "../../cls_datasets-master/manu_dataset_label_encoding/label_train.bin"
    path_manu_X_test = "../../cls_datasets-master/manu_dataset_label_encoding/data_test_cnn_pca.bin"
    path_manu_y_test = "../../cls_datasets-master/manu_dataset_label_encoding/label_test.bin"
    path_model = "../../models/2020-01-29_best_eq_sample.pkl"
    path_typo_X_test = "../../cls_datasets-master/typo_dataset_label_encoding/data_test_cnn_pca.bin"
    path_typo_y_test = "../../cls_datasets-master/typo_dataset_label_encoding/label_test.bin"

    # Load data manu for incremental
    with open(path_manu_X_train, "rb") as bin_f_data:
        with open(path_manu_y_train, "rb") as bin_f_label:
            X_train_inc = pickle.load(bin_f_data)
            y_train_inc = pickle.load(bin_f_label)

    with open(path_manu_X_test, "rb") as bin_f_data:
        with open(path_manu_y_test, "rb") as bin_f_label:
            X_test_inc = pickle.load(bin_f_data)
            y_test_inc = pickle.load(bin_f_label)
    # Load data typo
    with open(path_typo_X_test, "rb") as bin_f_data:
        with open(path_typo_y_test, "rb") as bin_f_label:
            X_test_typo = pickle.load(bin_f_data)
            y_test_typo = pickle.load(bin_f_label)

    permutation = np.random.permutation(X_train_inc.shape[0])
    shuffle_X_train_inc = X_train_inc[permutation]
    shuffle_y_train_inc = y_train_inc[permutation]
    del X_train_inc
    del y_train_inc


    params_dict_igt = {'pi': [0],
                       'jensen_threshold': [0.1, 0.2, 0.4],
                       'recreate': [True, False],
                       'batch_size': [1000, 2000]
                       }

    params_dict_rtst = {'pi': [0.2, 0.4, 0.6, 0.8],
                        'jensen_threshold': [0.1, 0.2, 0.4],
                        'recreate': [True, False],
                        'batch_size': [1000, 2000]
                        }
    incremental_grid_search_rapport_njobs(params_dict_igt, X_typo=X_test_typo, y_typo=y_test_typo,
                                          X_manu_train=shuffle_X_train_inc, y_manu_train=shuffle_y_train_inc,
                                          X_manu_test=X_test_inc, y_manu_test=y_test_inc, file='rapport_IGT.csv',
                                          verbose=2, n_jobs=3, n_random=6, path=path_model, mode='IGT')

    incremental_grid_search_rapport_njobs(params_dict_rtst, X_typo=X_test_typo, y_typo=y_test_typo,
                                          X_manu_train=shuffle_X_train_inc, y_manu_train=shuffle_y_train_inc,
                                          X_manu_test=X_test_inc, y_manu_test=y_test_inc, file='rapport_RTST.csv',
                                          verbose=2, n_jobs=3, n_random=6, path=path_model, mode='RTST')
