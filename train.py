
import pandas as pd
import copy
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


if __name__ == '__main__':

    
    path_typo_X_train = "../../cls_datasets-master/typo_dataset_label_encoding/data_train_cnn_pca.bin"
    path_typo_y_train = "../../cls_datasets-master/typo_dataset_label_encoding/label_train.bin"
    path_typo_X_test = "../../cls_datasets-master/typo_dataset_label_encoding/data_test_cnn_pca.bin"
    path_typo_y_test = "../../cls_datasets-master/typo_dataset_label_encoding/label_test.bin"
    #Load data
    with open(path_typo_X_train, "rb") as bin_f_data:
      with open(path_typo_y_train, "rb") as bin_f_label:
        X_train = pickle.load(bin_f_data)
        y_train = pickle.load(bin_f_label)

    with open(path_typo_X_test, "rb") as bin_f_data:
      with open(path_typo_y_test, "rb") as bin_f_label:
        X_test = pickle.load(bin_f_data)
        y_test = pickle.load(bin_f_label)

    print("X_train shape : ", X_train.shape)
    print("X_test shape : ", X_test.shape)
    print("y_train shape : ", y_train.shape)
    print("y_test shape : ", y_test.shape)


    # Train euclidean parameters
    params_dict = {'n_trees': [50, 60, 75],
                   'method_subclasses': [0.3, 0.6],
                   'method_max_features': ['sqrt',0.2, 0.5],
                   'distance': ['mahalanobis'],
                   'method_split': ['maj_class', 'eq_samples','farthest_max','alea'],
                   'min_samples_leaf': [20,30,50],
                   'min_samples_split': [1],
                   'max_depth': [100]
                   }
    rapport_maha = grid_search_rapport_njobs(params_dict,X_train, y_train, X_test, y_test, 'rapport_maha.csv',verbose=2, save_iteration=1,n_jobs=6, n_random=100)
