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


    def load_model(path):
        return pickle.load(open(path, "rb"))

    def load_batch(X, y, cursor, BATCH_SIZE):
        if cursor + BATCH_SIZE < len(X):
            batch_X = X[cursor:cursor + BATCH_SIZE]
            batch_y = y[cursor:cursor + BATCH_SIZE]
            cursor = cursor + BATCH_SIZE
        else:
            batch_X = X[cursor:]
            batch_y = y[cursor:]
        return batch_X, batch_y, cursor

    BATCH_SIZE = 1000

    print("LOAD MODEL")

    model = load_model(path_model)
    
    nb_batch = int(len(shuffle_X_train_inc) / BATCH_SIZE)

    if len(shuffle_X_train_inc) % BATCH_SIZE == 0:
        nb_batch = nb_batch
    else:
        nb_batch = nb_batch + 1


    score_uls_df_manu = pd.DataFrame()
    score_uls_df_typo = pd.DataFrame()
    cursor = 0
    """
    print("NB BATCH :", nb_batch)
    for i in range(0, nb_batch):
        print("ULS BATCH", i)
        batch_X, batch_y, cursor = load_batch(shuffle_X_train_inc, shuffle_y_train_inc, cursor, BATCH_SIZE)
        start = time.time()
        model.ULS(batch_X, batch_y)
        end = time.time()
        t = end - start
        print("ULS done in", t)
        y_pred_manu = model.predict(X_test_inc)
        y_pred_typo = model.predict(X_test_typo)

        # Store result
        df_temp_manu = pd.DataFrame(classification_report(y_test_inc, y_pred_manu, output_dict=True, digits=4)).loc[["recall"]]
        df_temp_manu["batch_nb"] = i
        df_temp_manu["time"] = round(t, 3)
        score_uls_df_manu = score_uls_df_manu.append(df_temp_manu)
        print("Write df in ./results/uls_result_manu.csv")
        score_uls_df_manu.to_csv('./results/uls_result_manu.csv', sep=";")

        df_temp_typo = pd.DataFrame(classification_report(y_test_typo, y_pred_typo, output_dict=True, digits=4)).loc[["recall"]]
        df_temp_typo["batch_nb"] = i
        df_temp_typo["time"] = round(t, 3)
        score_uls_df_typo = score_uls_df_typo.append(df_temp_typo)
        print("Write df in ./results/uls_result_typo.csv")
        score_uls_df_typo.to_csv('./results/uls_result_typo.csv', sep=";")
        print("Score test manu (uls):", df_temp_manu["accuracy"])
        print("Score test typo (uls):", df_temp_typo["accuracy"])
        print()
    """
    score_igt_df_manu = pd.DataFrame()
    score_igt_df_typo = pd.DataFrame()
    cursor = 0
    print("NB BATCH :", nb_batch)
    for i in range(0, nb_batch):
        print("IGT BATCH", i)
        batch_X, batch_y, cursor = load_batch(shuffle_X_train_inc, shuffle_y_train_inc, cursor, BATCH_SIZE)
        start = time.time()
        model.IGT(batch_X, batch_y, jensen_threshold=0.1, recreate=True)
        end = time.time()
        t = end - start
        print("IGT done in", t)
        y_pred_manu = model.predict(X_test_inc)
        y_pred_typo = model.predict(X_test_typo)
        # Store result
        df_temp_manu = pd.DataFrame(classification_report(y_test_inc, y_pred_manu, output_dict=True, digits=4)).loc[["recall"]]
        df_temp_manu["batch_nb"] = i
        df_temp_manu["time"] = round(t, 3)
        score_igt_df_manu = score_igt_df_manu.append(df_temp_manu)
        print("Write df in ./results/igt_result_manu_2.csv")
        score_igt_df_manu.to_csv('./results/igt_result_manu_2.csv', sep=";")
        df_temp_typo = pd.DataFrame(classification_report(y_test_typo, y_pred_typo, output_dict=True, digits=4)).loc[["recall"]]
        df_temp_typo["batch_nb"] = i
        df_temp_typo["time"] = round(t, 3)
        score_igt_df_typo = score_igt_df_typo.append(df_temp_typo)
        print("Write df in ./results/igt_result_typo_2.csv")
        score_igt_df_typo.to_csv('./results/igt_result_typo_2.csv', sep=";")
        print("Score test manu (igt):", df_temp_manu["accuracy"])
        print("Score test typo (igt):", df_temp_typo["accuracy"])
        print()
