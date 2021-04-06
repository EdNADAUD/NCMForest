import pickle

import pandas as pd
import numpy as np
import sys
from src.Node import Node
from src.NCMTree import NCMTree
from src.NCMForest import NCMForest
import itertools
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
import itertools
import multiprocessing
from multiprocessing import Pool, Lock
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import os
from itertools import repeat
from functools import partial
import random
from sklearn.metrics import classification_report
sys.path.append("..")

from headers.utils import load_beer_dataset

def accuracy_print(X_train, y_train, X_test, y_test, clf, bool_cross=False, print_f=False):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param clf:
    :param bool_cross:
    :param print_f:
    :return:
    """
    start = time.time()
    y_train_pred = clf.predict(X_train)
    end = time.time() - start
    y_test_pred = clf.predict(X_test)
    score_train = accuracy_score(y_train, y_train_pred)
    score_test = accuracy_score(y_test, y_test_pred)
    if bool_cross:
        cross_val = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)
    else:
        cross_val = False
    if print_f:
        print("Score en train : ", round(score_train, 3))
        print("Score en test : ", round(score_test, 3))
        print("Cross-val mean : ", round(cross_val.mean(), 3), " ecart-type ", round(cross_val.std(), 3))
    return score_train, score_test, cross_val, end


def grid_search_rapport_njobs(params_dict, X_train, y_train, X_test, y_test, file='rapport.csv', verbose=1,
                              save_iteration=20, n_jobs=1, n_random=None):

    """

    :param params_dict:
    :param X:
    :param y:
    :param file:
    :param verbose:
    :param save_iteration:
    :param n_jobs:
    :return:
    """
    d = datetime.now()
    f = d.strftime('%Y-%m-%d-%H-%M_')

    manager = multiprocessing.Manager()
    ns = manager.Namespace()
    lock = manager.Lock()
    ns.rapport_df = pd.DataFrame(columns=['n_trees', 'method_subclasses', 'method_max_features', 'distance', 'method_split',
                                          'min_samples_leaf', 'min_samples_split', 'max_depth', 'score_train',
                                          'score_test', 'avg_depth', 'avg_size', 'fit_time', 'predict_time'])
    ns.it = 0
    ns.file_name = 'results/' + f + file

    ns.verbose = verbose
    ns.save_iteration = save_iteration

    print("START :")
    start_full = time.time()

    ns.X_train = X_train
    ns.X_test = X_test
    ns.y_train = y_train
    ns.y_test = y_test

    keys = params_dict.keys()
    values = (params_dict[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    if n_random:
        combinations = random.choices(combinations, k=n_random)
    ns.nb_combi = len(combinations)
    print("Number of combinations: ", str(ns.nb_combi))

    if n_jobs == -1:
        p = Pool(processes=os.cpu_count())
    else:
        p = Pool(processes=n_jobs)
    res = p.map(partial(test_combinaison, ns=ns, lock=lock), combinations)
    p.close()
    p.join()
    ns.rapport_df.to_csv(ns.file_name, sep=';')
    print("END :" + str(time.time() - start_full))
    return ns.rapport_df


def test_combinaison(comb, ns, lock):
    """

    :param comb:
    :param ns:
    :param lock:
    :return:
    """
    try:
        ncm = NCMForest(n_trees=comb['n_trees'],
                        method_k_bis=comb['method_subclasses'],
                        method_max_features=comb['method_max_features'],
                        distance=comb['distance'],
                        method_split=comb['method_split'],
                        min_samples_leaf=comb['min_samples_leaf'],
                        min_samples_split=comb['min_samples_split'],
                        max_depth=comb['max_depth']
                        )
        start_fit = time.time()
        if ns.verbose > 1:
            print("Fitting with params: \n")
            print(
                "n_trees:{}\nmethod_kbis:{}\nmethod_max_features:{}\ndistance:{}\nmethod_split:{}\nmin_samples_leaf:{}\nmin_samples_split:{}\nmax_depth:{}".format(
                    comb['n_trees'], comb['method_subclasses'], comb['method_max_features'], comb['distance'],
                    comb['method_split'], comb['min_samples_leaf'], comb['min_samples_split'], comb['max_depth']))
        ncm.fit(ns.X_train, ns.y_train)
        end_fit = time.time() - start_fit
        score_train, score_test, _, predict_time = accuracy_print(ns.X_train, ns.y_train, ns.X_test, ns.y_test, ncm)
        if ns.verbose > 1:
            print(ncm)
            print('Fit time :', end_fit)
            print('Predict time :', predict_time)
        lock.acquire()
        try:
            depths = 0
            cardinalities = 0
            for tree in ncm.trees:
                depths = depths + tree.depth
                cardinalities = cardinalities + tree.cardinality
            avg_depth = depths / len(ncm.trees)
            avg_size = cardinalities / len(ncm.trees)

            if ns.verbose > 1:
                print("[ACQUIRE] : About to write in ns.rapport_df")
            ns.rapport_df = ns.rapport_df.append([{
                'n_trees': comb['n_trees'],
                'method_subclasses': comb['method_subclasses'],
                'method_max_features': comb['method_max_features'],
                'distance': comb['distance'],
                'method_split': comb['method_split'],
                'min_samples_leaf': comb['min_samples_leaf'],
                'min_samples_split': comb['min_samples_split'],
                'max_depth': comb['max_depth'],
                'score_train': round(score_train, 3),
                'score_test': round(score_test, 3),
                'avg_depth': round(avg_depth, 3),
                'avg_size': round(avg_size, 3),
                'fit_time': round(end_fit, 1),
                'predict_time': round(predict_time, 1)}])
        finally:
            lock.release()
            if ns.verbose > 1:
                print("[RELEASE] : Release rapport_df")
        if ns.it % ns.save_iteration == 0:
            print("[ACQUIRE]")
            lock.acquire()
            try:
                ns.rapport_df.to_csv(ns.file_name, sep=';')
            finally:
                lock.release()

    except Exception as e:
        lock.acquire()
        try:
            ns.rapport_df = ns.rapport_df.append([{
                'n_trees': (comb, e),
                'method_subclasses': 0,
                'method_max_features': 0,
                'distance': 0,
                'method_split': 0,
                'min_samples_leaf': 0,
                'min_samples_split': 0,
                'max_depth': 0,
                'score_train': 0,
                'score_test': 0,
                'avg_depth': 0,
                'avg_size': 0,
                'fit_time': 0,
                'predict_time': 0}])
            print('ERROR : saving file..')
            print(e)
            ns.rapport_df.to_csv(ns.file_name, sep=';')
        finally:
            lock.release()
    ns.it += 1
    if ns.verbose > 0:
        print()
        print(
            """Progression : """ + str(round((ns.it / ns.nb_combi) * 100)) + """%  (""" + str(ns.it) + """ of """ + str(
                ns.nb_combi) + """)""")
    return ns.rapport_df


def grid_search_rapport(params_dict, X_train, y_train, X_test, y_test, file='rapport.csv', verbose=1, save_iteration=20,
                        n_random=None):
    d = datetime.now()
    string_date = d.strftime('%Y-%m-%d-%H-%M_')
    file_name = 'results/' + string_date + file

    print("START :")
    start_full = time.time()

    keys = params_dict.keys()
    values = (params_dict[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    if n_random:
        combinations = random.choices(combinations, k=n_random)
    nb_combi = len(combinations)
    print("Number of combinations: ", str(nb_combi))
    it = 1

    rapport_df = pd.DataFrame(columns=['n_trees', 'method_subclasses', 'method_max_features', 'distance', 'method_split',
                                       'min_samples_leaf', 'min_samples_split', 'max_depth', 'score_train',
                                       'score_test', 'avg_depth', 'avg_size', 'fit_time', 'predict_time'])

    for comb in combinations:
        try:
            ncm = NCMForest(n_trees=comb['n_trees'],
                            method_k_bis=comb['method_subclasses'],
                            method_max_features=comb['method_max_features'],
                            distance=comb['distance'],
                            method_split=comb['method_split'],
                            min_samples_leaf=comb['min_samples_leaf'],
                            min_samples_split=comb['min_samples_split'],
                            max_depth=comb['max_depth']
                            )
            start_fit = time.time()
            if verbose > 1:
                print("Fitting with params: \n")
                print(
                    "n_trees:{}\nmethod_kbis:{}\nmethod_max_features:{}\ndistance:{}\nmethod_split:{}\nmin_samples_leaf:{}\nmin_samples_split:{}\nmax_depth:{}".format(
                        comb['n_trees'], comb['method_subclasses'], comb['method_max_features'], comb['distance'],
                        comb['method_split'], comb['min_samples_leaf'], comb['min_samples_split'], comb['max_depth']))
            ncm.fit(X_train, y_train)
            end_fit = time.time() - start_fit
            score_train, score_test, _, predict_time = accuracy_print(X_train, y_train, X_test, y_test, ncm)
            if verbose > 1:
                print(ncm)
                print('Fit time :', end_fit)
                print('Predict time :', predict_time)
            depths = 0
            cardinalities = 0
            for tree in ncm.trees:
                depths = depths + tree.depth
                cardinalities = cardinalities + tree.cardinality
            avg_depth = depths / len(ncm.trees)
            avg_size = cardinalities / len(ncm.trees)

            rapport_df = rapport_df.append([{
                'n_trees': comb['n_trees'],
                'method_subclasses': comb['method_subclasses'],
                'method_max_features': comb['method_max_features'],
                'distance': comb['distance'],
                'method_split': comb['method_split'],
                'min_samples_leaf': comb['min_samples_leaf'],
                'min_samples_split': comb['min_samples_split'],
                'max_depth': comb['max_depth'],
                'score_train': round(score_train, 3),
                'score_test': round(score_test, 3),
                'avg_depth': round(avg_depth, 3),
                'avg_size': round(avg_size, 3),
                'fit_time': round(end_fit, 1),
                'predict_time': round(predict_time, 1)}])

            if it % save_iteration == 0:
                rapport_df.to_csv(file_name, sep=';')

            if verbose > 0:
                print(
                    """Progression : """ + str(round((it / nb_combi) * 100)) + """%   (""" + str(it) + """ of """ + str(
                        nb_combi) + """)""")
        except Exception as e:
            rapport_df = rapport_df.append([{
                'comb': (comb, e),
                'score_train': 0,
                'score_test': 0,
                'depth': 0,
                'fit_time': 0,
                'predict_time': 0}])
            print('ERROR : saving file..')
            rapport_df.to_csv(file_name, sep=';')
        it += 1
    rapport_df.to_csv(file_name, sep=';')
    print("END :" + str(time.time() - start_full))
    return rapport_df


def incremental_grid_search_rapport_njobs(params_dict, X_typo, y_typo, X_manu_train, y_manu_train, X_manu_test,
                                          y_manu_test, file='rapport.csv', verbose=1, n_jobs=-1, n_random=None, path=None, mode=None):
    """
    :param params_dict:
    :param X:
    :param y:
    :param file:
    :param verbose:
    :param save_iteration:
    :param n_jobs:
    :param n_random:
    :param model:
    :param path:
    :param batch_size:
    :return:
    """
    d = datetime.now()
    f = d.strftime('%Y-%m-%d-%H-%M_')
    manager = multiprocessing.Manager()
    ns = manager.Namespace()
    lock = manager.Lock()

    #ns.rapport_df = pd.DataFrame(columns=['BATCH_SIZE', 'score_manu', 'score_typo', 'jensen_threshold', 'pi'])
    ns.score_df_manu = pd.DataFrame()
    ns.score_df_typo = pd.DataFrame()

    ns.it = 0
    ns.file_name_typo = 'results/' + f +'_typo_'+ file
    ns.file_name_manu = 'results/' + f +'_manu_'+ file
    ns.verbose = verbose

    print('START ', f)
    start_full = time.time()

    ns.X_typo = X_typo
    ns.y_typo = y_typo
    ns.X_manu_train = X_manu_train
    ns.y_manu_train = y_manu_train
    ns.X_manu_test = X_manu_test
    ns.y_manu_test = y_manu_test
    
    keys = params_dict.keys()
    values = (params_dict[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    if n_random:
        combinations = random.choices(combinations, k=n_random)
    ns.nb_combi = len(combinations)
    print("Number of combinations: ", str(ns.nb_combi))

    if n_jobs == -1:
        p = Pool(processes=os.cpu_count())
    else:
        p = Pool(processes=n_jobs)
    p.map(partial(test_combinaison_inc, ns=ns, lock=lock, path=path, mode=mode),
                combinations)
    p.close()
    p.join()
    print("END :" + str(time.time() - start_full))


def load_batch(X, y, cursor, BATCH_SIZE):
    if cursor + BATCH_SIZE < len(X):
        batch_X = X[cursor:cursor + BATCH_SIZE]
        batch_y = y[cursor:cursor + BATCH_SIZE]
        cursor = cursor + BATCH_SIZE
    else:
        batch_X = X[cursor:]
        batch_y = y[cursor:]
    return batch_X, batch_y, cursor


def load_model(path):
    return pickle.load(open(path, "rb"))


def test_combinaison_inc(comb, ns, lock, path, mode):
    """
    :param comb:
    :param ns:
    :param lock:
    :return:
"""
    nb_batch = int(len(ns.X_manu_train) / comb['batch_size'])

    if len(ns.X_manu_train) % comb['batch_size'] == 0:
        nb_batch = nb_batch
    else:
        nb_batch = nb_batch + 1
    cursor = 0
    print("Nb batch:", nb_batch)
    if ns.verbose > 1:
        print("Fitting with params: \n")
        print(
            "BATCH_SIZE:{}\njensen_threshold:{}\nrecreate:{}\npi:{}".format(
                comb['batch_size'], comb['jensen_threshold'], comb['recreate'], comb['pi']))
    model = load_model(path)


    for i in range(0, nb_batch):
            try:
                print(mode, " BATCH ", i)
                batch_X, batch_y, cursor = load_batch(ns.X_manu_train, ns.y_manu_train, cursor, comb['batch_size'])

                start_fit = time.time()
                if mode == "IGT":
                    print("fit IGT")
                    print(batch_X.shape)
                    model.IGT(batch_X, batch_y, jensen_threshold=comb['jensen_threshold'], recreate=comb['recreate'])
                else:
                    print("fit RTST")
                    print(batch_X.shape)
                    model.RTST(batch_X, batch_y, jensen_threshold=comb['jensen_threshold'], pi=comb['pi'],
                               recreate=comb['recreate'])
                end_fit = time.time() - start_fit
                print("----------- Incremental Done -------------")

                print('----- Predict Manu-----')
                y_pred_manu = model.predict(ns.X_manu_test)
                print('----- Predict Typo-----')
                y_pred_typo = model.predict(ns.X_typo)


                # Store result
                df_temp_manu = pd.DataFrame(classification_report(ns.y_manu_test, y_pred_manu, output_dict=True, digits=4)).loc[["recall"]]
                df_temp_manu["batch_nb"] = i
                df_temp_manu["time"] = round(end_fit, 3)
                df_temp_manu["batch_size"] = comb['batch_size']
                df_temp_manu["jensen_threshold"] = comb['jensen_threshold']
                df_temp_manu["recreate"] = comb['recreate']
                df_temp_manu["pi"] = comb['pi']
                df_temp_manu["mode"] = mode



                lock.acquire()
                try:
                    ns.score_df_manu = ns.score_df_manu.append(df_temp_manu)
                    print("Write df in "+ns.file_name_manu)
                    ns.score_df_manu.to_csv(ns.file_name_manu, sep=";")
                finally:
                    lock.release()

                df_temp_typo = pd.DataFrame(classification_report(ns.y_typo, y_pred_typo, output_dict=True, digits=4)).loc[["recall"]]
                df_temp_typo["batch_nb"] = i
                df_temp_typo["time"] = round(end_fit, 3)
                df_temp_typo["batch_size"] = comb['batch_size']
                df_temp_typo["jensen_threshold"] = comb['jensen_threshold']
                df_temp_typo["recreate"] = comb['recreate']
                df_temp_typo["pi"] = comb['pi']
                df_temp_typo["mode"] = mode

                lock.acquire()
                try:
                    ns.score_df_typo = ns.score_df_typo.append(df_temp_typo)
                    print("Write df in "+ns.file_name_typo)
                    ns.score_df_typo.to_csv(ns.file_name_typo, sep=";")
                finally:
                    lock.release()

                if ns.verbose > 1:
                    print("cursor :", cursor)
                    print("Score test manu :", df_temp_manu["accuracy"].values)
                    print("Score test typo :", df_temp_typo["accuracy"].values)
            except Exception as e:
                print(e)

    lock.acquire()
    try:
        ns.it += 1
    finally:
        lock.release()
    if ns.verbose > 0:
        print()
        print(
            """Progression : """ + str(round((ns.it / ns.nb_combi) * 100)) + """%  (""" + str(ns.it) + """ of """ + str(
                ns.nb_combi) + """)""")
