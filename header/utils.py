'''
Author: Ludovic Carlu
Description: this file will store all the useful functions that can be used
	in machine learning algorithms in order to not repeat them several
	times if they are used in different algorithms
Functions:
	bootstrap_oob
'''

# Libraries
import numpy as np
import pandas as pd
import scipy.stats


def boostrap_oob(df_input):
    """
    :param df_input: Df which contains as columns features and y
    :return boostrap_df and and oob_df
    """
    bootstrap = df_input.sample(len(df_input.index), replace=True)
    oob_index = [x for x in df_input.index if x not in bootstrap.index]
    oob = df_input.iloc[oob_index]

    return bootstrap, oob


def most_frequent_classes(X):
    """

    :param X:
    :return:
    """
    print("type_most_frequent_classes:",type(X))
    (classes, counts) = np.unique(X, return_counts=True)
    index = np.argmax(counts)
    print("return most_fr_cls:",classes[index])
    return classes[index]


def load_beer_dataset(file_path="../data/beer_quality.xlsx"):

    beer = pd.read_excel(file_path)
    y = beer["quality"].values
    X = beer.drop(["quality"], axis=1).values

    return X, y


def load_Frogs_dataset(file_path="../data/Frogs_MFCCs.csv", alea=3):
    frogs = pd.read_csv(file_path)
    X = frogs.drop(["MFCCs_ 1", "RecordID", "Genus", "Family", "Species"], axis=1)

    if alea == 1:
        y = frogs["Genus"]
    elif alea == 2:
        y = frogs["Family"]
    else:
        y = frogs["Species"]
    return X, y

def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance

def load_data_pfe():
    data_path = '../data/pfe_data_features/data_features.csv'
    test_path = '../data/pfe_data_features/test_features.csv'
    label_data_path = '../data/pfe_data_features/train_label.csv'
    label_test_path = '../data/pfe_data_features/test_label.csv'

    X_train = pd.read_csv(data_path)
    y_train = pd.read_csv(label_data_path)
    X_test = pd.read_csv(test_path)
    y_test = pd.read_csv(label_test_path)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # X, y = load_beer_dataset()
    # X, y = load_Frogs_dataset()
    X, y = load_data_pfe()
