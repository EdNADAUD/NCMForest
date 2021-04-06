import sys

import numpy as np
from scipy.special import softmax

sys.path.append("..")
from headers.utils import *
from math import log2
from math import sqrt
import pandas as pd
from src.Node import Node


class NCMTree:
    def __init__(self, max_depth=10, min_samples_split=2,
                 min_samples_leaf=1, random_state=None, debug=False, distance="euclidean",
                 method_subclasses="sqrt", method_split="alea", method_max_features="sqrt"):
        """

        :param max_depth:
        :param min_samples_split:
        :param min_samples_leaf:
        :param random_state:
        :param debug:
        :param distance:
        :param method_subclasses:
        :param method_split:
        :param method_max_features:
        """
        self.method_max_features = method_max_features
        self.max_features = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.root = None
        self.depth = 1
        self.debug = debug
        self.cardinality = 0
        self.distance = distance
        self.method_subclasses = method_subclasses
        self.method_split = method_split

        if debug:
            print('=== Initialisation NCMTree ===')
            print('\t max_features: {}'.format(self.method_max_features))
            print('\t max_depth: {}'.format(self.max_depth))
            print('\t min_samples_split: {}'.format(self.min_samples_split))
            print('\t min_samples_leaf: {}'.format(self.min_samples_leaf))
            print('\t random_state: {}'.format(self.random_state))
            print('====================================')

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return:
        """

        # number of features selected
        if self.method_max_features == 'log2':
            self.max_features = int(log2(X.shape[1]))
            assert X.shape[1] != 0, 'Dimensions nulles'
            assert self.max_features != 0, 'self.max_features = 0'

        elif self.method_max_features == 'sqrt':
            self.max_features = int(sqrt(X.shape[1]))
            assert X.shape[1] != 0, 'Dimensions nulles'
            assert self.max_features != 0, 'self.max_features = 0'

        elif type(self.method_max_features) == float:
            self.max_features = int(self.method_max_features * (X.shape[1]))
            assert X.shape[1] != 0, 'Dimensions nulles'
            assert self.max_features != 0, 'self.max_features = 0'

        elif type(self.method_max_features) == int:
            self.max_features = self.method_max_features
            assert X.shape[1] != 0, 'Dimensions nulles'
            assert self.max_features != 0, 'self.max_features = 0'

        self.root = self.build_nodes(X, y, None, 0)

    def build_nodes(self, X, y, parent, localdepth=0):
        """
        recursive function for growing tree
        :param localdepth:
        :param X:
        :param y:
        :param parent:
        :return:
        """
        current_node = Node(parent, False, self.min_samples_leaf, self.max_features, self.distance, self.method_subclasses,
                            self.method_split)
        self.cardinality += 1
        current_node.fit(X, y)
        split_possible, left_index, right_index = current_node.predict_split(X)

        # ------------stopping criterion (and in Node.py, fct() predict_split )------------
        if split_possible and localdepth < self.max_depth and len(X) > self.min_samples_split:
            if self.depth < localdepth:
                self.depth = localdepth
            # LEFT NODE
            left_child = self.build_nodes(X[left_index], y[left_index], current_node, localdepth+1)
            current_node.set_left_child(left_child)
            # RIGHT NODE
            right_child = self.build_nodes(X[right_index], y[right_index], current_node, localdepth+1)
            current_node.set_right_child(right_child)
        else:
            current_node.set_leaf(True)
        return current_node

    def predict(self, X):
        """

        :param X:
        :param proba:
        :return:
        """
        return self.root.predict_all(X)

    def ULS(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        self.root.ULS(X, y)

    def IGT(self, X, y, extra_depth, jensen_threshold=0.1, recreate=True):
        """

        :param recreate:
        :param jensen_threshold:
        :param extra_depth:
        :param X: incremental data
        :param y: incremental label data
        :return:
        """
        # select IGT leaves and generated + incremental data
        splittable_leaves = self.root.IGT(X, y, jensen_threshold, recreate)
        self.max_depth = self.depth+extra_depth
        all_depths = []
        old_depth = self.depth

        for leaf_tab in splittable_leaves:
            leaf = leaf_tab[0]
            X = leaf_tab[1]
            y = leaf_tab[2]
            leaf.set_leaf(False)

            split_possible, left_index, right_index = leaf.predict_split(X)
            if split_possible and self.depth < self.max_depth and len(X) > self.min_samples_split:
                # LEFT NODE
                left_child = self.build_nodes(X[left_index], y[left_index], leaf)
                leaf.set_left_child(left_child)
                # RIGHT NODE
                right_child = self.build_nodes(X[right_index], y[right_index], leaf)
                leaf.set_right_child(right_child)
            else:
                leaf.set_leaf(True)

            all_depths.append(self.depth)
            self.depth = old_depth

        if splittable_leaves:
            self.depth = max(all_depths)

    def RTST(self, X, y, extra_depth=10, jensen_threshold=0.1, recreate=True, pi=0.1):
        """
        @param X:
        @param y:
        @param extra_depth: maximum growing allowed
        @param jensen_threshold:
        @param recreate:
        @param pi: max proportion of nodes that can be recreated
        """
        all_sizes = self.root.get_all_sizes()
        df_node = pd.DataFrame([], columns=['node', 'proba', 'size'])
        total_size = self.root.size()
        for key, value in all_sizes.items():
            df_node = df_node.append(pd.Series([key, 1/(value+1), value/total_size], index=['node', 'proba', 'size']), ignore_index=True)

        current_pi = 0.0
        while current_pi < pi:
            index_del = np.random.choice(df_node['node'].index, size=1, replace=False, p=softmax(df_node['proba']))[0]
            current_pi = current_pi + df_node.at[index_del, 'size']
            current_node = df_node.at[index_del, 'node']
            child_nodes = current_node.get_child_nodes()
            child_nodes.remove(current_node)  # todo optimize
            current_node.set_left_child(None)
            current_node.set_right_child(None)
            current_node.set_leaf(True)
            self.cardinality = self.cardinality - len(child_nodes)
            df_node.drop(df_node[df_node['node'].isin(child_nodes)].index, inplace=True)
            del child_nodes[:]
            df_node.drop(index_del, axis=0, inplace=True)

        self.IGT(X, y, extra_depth, jensen_threshold, recreate)



