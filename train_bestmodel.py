import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

if __name__ =="__main__":

    path_typo_X_train = "../../cls_datasets-master/typo_dataset_label_encoding/data_train_cnn_pca.bin"
    path_typo_y_train = "../../cls_datasets-master/typo_dataset_label_encoding/label_train.bin"

    path_typo_X_test = "../../cls_datasets-master/typo_dataset_label_encoding/data_test_cnn_pca.bin"
    path_typo_y_test = "../../cls_datasets-master/typo_dataset_label_encoding/label_test.bin"

    #path_manu_X_train = "../../cls_datasets-master/manu_dataset_label_encoding/data_train_cnn_pca.bin"
    #path_manu_y_train = "../../cls_datasets-master/manu_dataset_label_encoding/label_train.bin"

    path_manu_X_test = "../../cls_datasets-master/manu_dataset_label_encoding/data_test_cnn_pca.bin"
    path_manu_y_test = "../../cls_datasets-master/manu_dataset_label_encoding/label_test.bin"


    #Load data
    with open(path_typo_X_train, "rb") as bin_f_data:
      with open(path_typo_y_train, "rb") as bin_f_label:
        X_train = pickle.load(bin_f_data)
        y_train = pickle.load(bin_f_label)

    with open(path_typo_X_test, "rb") as bin_f_data:
      with open(path_typo_y_test, "rb") as bin_f_label:
        X_test = pickle.load(bin_f_data)
        y_test = pickle.load(bin_f_label)

    with open(path_manu_X_test, "rb") as bin_f_data:
        with open(path_manu_y_test, "rb") as bin_f_label:
            X_test_manu = pickle.load(bin_f_data)
            y_test_manu = pickle.load(bin_f_label)

    print("X_train shape : ", X_train.shape)
    print("X_test shape : ", X_test.shape)
    print("y_train shape : ", y_train.shape)
    print("y_test shape : ", y_test.shape)
    df = pd.DataFrame(columns=['method_split','accuracy_typo','accuracy_manu'])
    #
    ncm_eqsample = NCMForest(n_trees=75,
                             method_split='eq_sample',
                             method_k_bis=0.6,
                             method_max_features=0.2,
                             max_depth=100,
                             distance='euclidean',
                             min_samples_leaf=30)
    ncm_eqsample.fit(X_train, y_train)
    d = datetime.now()
    f = d.strftime('%Y-%m-%d_') + 'best_eq_sample'
    pickle.dump(ncm_eqsample, open('models/' + f + '.pkl', "wb"))

    y_pred = ncm_eqsample.predict(X_test)
    y_pred_manu = ncm_eqsample.predict(X_test_manu)
    accuracy_typo = accuracy_score(y_test, y_pred)
    accuracy_manu = accuracy_score(y_test_manu, y_pred_manu)

    print(accuracy_typo, accuracy_manu)

    df = df.append([{'method_split':'eq_sample',
                     'accuracy_typo':accuracy_typo,
                     'accuracy_manu':accuracy_manu}])




    ncm_alea = NCMForest(n_trees=75,
                             method_split='alea',
                             method_k_bis=0.6,
                             method_max_features=0.2,
                             max_depth=100,
                             distance='euclidean',
                             min_samples_leaf=30)
    ncm_alea.fit(X_train, y_train)
    d = datetime.now()
    f = d.strftime('%Y-%m-%d_') + 'best_alea'
    pickle.dump(ncm_alea, open('models/' + f + '.pkl', "wb"))

    y_pred = ncm_alea.predict(X_test)
    y_pred_manu = ncm_alea.predict(X_test_manu)
    accuracy_typo = accuracy_score(y_test, y_pred)
    accuracy_manu = accuracy_score(y_test_manu, y_pred_manu)

    print(accuracy_typo, accuracy_manu)

    df = df.append([{'method_split':'alea',
                     'accuracy_typo':accuracy_typo,
                     'accuracy_manu':accuracy_manu}])

    ncm_farthestmax = NCMForest(n_trees=75,
                             method_split='farthest_max',
                             method_k_bis=0.3,
                             method_max_features=0.2,
                             max_depth=100,
                             distance='euclidean',
                             min_samples_leaf=30)
    ncm_farthestmax.fit(X_train, y_train)
    d = datetime.now()
    f = d.strftime('%Y-%m-%d_') + 'best_farthest_max'
    pickle.dump(ncm_farthestmax, open('models/' + f + '.pkl', "wb"))

    y_pred = ncm_farthestmax.predict(X_test)
    y_pred_manu = ncm_farthestmax.predict(X_test_manu)
    accuracy_typo = accuracy_score(y_test, y_pred)
    accuracy_manu = accuracy_score(y_test_manu, y_pred_manu)

    print(accuracy_typo, accuracy_manu)

    df = df.append([{'method_split':'farthest_max',
                     'accuracy_typo':accuracy_typo,
                     'accuracy_manu':accuracy_manu}])

    ncm_majclass = NCMForest(n_trees=50,
                             method_split='maj_class',
                             method_k_bis=0.6,
                             method_max_features=0.5,
                             max_depth=100,
                             distance='euclidean',
                             min_samples_leaf=30)
    ncm_majclass.fit(X_train, y_train)
    d = datetime.now()
    f = d.strftime('%Y-%m-%d_') + 'best_maj_class'
    pickle.dump(ncm_majclass, open('models/' + f + '.pkl', "wb"))

    y_pred = ncm_majclass.predict(X_test)
    y_pred_manu = ncm_majclass.predict(X_test_manu)
    accuracy_typo = accuracy_score(y_test, y_pred)
    accuracy_manu = accuracy_score(y_test_manu, y_pred_manu)

    print(accuracy_typo, accuracy_manu)

    df = df.append([{'method_split':'maj_class',
                     'accuracy_typo':accuracy_typo,
                     'accuracy_manu':accuracy_manu}])

    df.to_csv('./results/best_models_score.csv',sep=';')