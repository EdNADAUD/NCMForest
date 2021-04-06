from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
import pickle
from datetime import datetime
from sklearn.datasets import load_iris

from src.Node import Node
from src.NCMTree import NCMTree
from src.NCMForest import NCMForest
from src.NCMGridSearch import grid_search_rapport_njobs
from src.NCMGridSearch import grid_search_rapport
from headers.NCMClassifier import NCMClassifier
from headers.utils import *
import copy
from sklearn.datasets import load_iris,load_digits,load_breast_cancer

# Pour mettre le projet sur gitlab
# On fait !git add *
# Puis !git commit -m "Message"
# Et enfin !git push

# Pour prendre le code sur git faire !git pull

def test_Node():
    x =np.array([[1.0, 1.0], [2.0, 2.0], [1.5, 1.5], [10.0, 10.0], [8.0, 8.0], [10.0, 10.0], [8.0, 8.0],  [10.0, 10.0], [8.0, 8.0],[1.0, 1.0], [2.0, 2.0], [1.5, 1.5], [10.0, 10.0], [8.0, 8.0], [10.0, 10.0], [8.0, 8.0],  [10.0, 10.0], [8.0, 8.0]])
    y = np.array(['a', 'b','c' ,'d','c' ,'d','a', 'b','c' ,'e','c' ,'e'])
    node = Node(None, False, 1, 2, method_split='farthest_max', method_k_bis=1.0)
    node.fit(x, y)
    node.plot(x)

def test_NCMClassifier():
    X, y = make_classification(n_samples=100,
                               n_features=4,
                               n_informative=4,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=3,
                               random_state=0, n_clusters_per_class=1,
                               shuffle=False)

    clf = NCMClassifier(metric='euclidean', sub_features=[0, 1])
    clf.fit(X, y)
    #y_pred = splitting_clf.predict(X)
    #print(y_pred)

def test_NCMTree():
    #X = np.array([[1.0, 1.0], [2.0, 2.0], [1.5, 1.5], [10.0, 10.0], [8.0, 8.0], [10.0, 10.0], [8.0, 8.0], [10.0, 10.0],
                  #[8.0, 8.0], [1.0, 1.0], [2.0, 2.0], [1.5, 1.5], [10.0, 10.0], [8.0, 8.0], [10.0, 10.0], [8.0, 8.0],
                 # [10.0, 10.0], [8.0, 8.0]])
    #y = np.array(['a', 'b', 'c', 'd', 'c', 'd', 'a', 'b', 'c', 'e', 'c', 'e', 'c', 'e', 'c', 'e', 'c', 'e'])
    X, y = make_classification(n_samples=20,
                               n_features=2,
                               n_informative=2,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=3,
                               random_state=0, n_clusters_per_class=1,
                               shuffle=False)

    tree = NCMTree(criterion='gini', max_classes=None, max_depth=10, min_samples_split=2,
                   min_samples_leaf=5, method_max_features='sqrt', random_state=None, debug=False, distance="euclidean",
                   method_k_bis="log2", method_split="maj_class")
    tree.fit(X, y)
    print(tree.predict(X)) ## PROBLEME ??

def test_NCMForest():
    X, y = make_classification(n_samples=1000,
                               n_features=10,
                               n_informative=10,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=4,
                               random_state=0, n_clusters_per_class=1,
                               shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = NCMForest(n_trees=1, max_depth=50, min_samples_split=1, min_samples_leaf=10,
                         method_max_features=0.3, method_k_bis=0.5, method_split='eq_samples',
                         distance="euclidean")
    clf.fit(X_train, y_train)
    clf1 = copy.deepcopy(clf)
    print("Euclidian")
    print("First tree depth : " + str(clf.trees[0].depth))
    print("---")
    print("NCMF train proba: " + str(clf.score(X_train, y_train)))
    print("NCMF test proba: " + str(clf.score(X_test, y_test)))

    # splitting_clf.IGT(X_test, y_test, X_train, y_train, 10)
    # print("--IGT---")
    # print("First tree depth : " + str(splitting_clf.trees[0].depth))
    # print("NCMF train proba: " + str(splitting_clf.score(X_train, y_train)))
    # print("NCMF test proba: " + str(splitting_clf.score(X_test, y_test)))
    # print("NCM OOB: ", splitting_clf.oob_score())
    #
    # clf1.ULS(X_test, y_test)
    # print("--ULS---")
    # print("First tree depth : " + str(clf1.trees[0].depth))
    # print("NCMF train proba: " + str(clf1.score(X_train, y_train)))
    # print("NCMF test proba: " + str(clf1.score(X_test, y_test)))
    # print("NCM OOB: ", clf1.oob_score())


def test_gridsearch():
    X, y = load_beer_dataset('data/beer_quality.xlsx')
    params_dict = {'n_trees': [50],
                   'method_subclasses': [0.7],
                   'method_max_features': [0.2],
                   'distance': ['euclidean'],
                   'method_split': ['maj_class', 'eq_samples'],
                   'min_samples_leaf': [5, 10],
                   'min_samples_split': [1],
                   'max_depth': [100]
                   }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    result_df = grid_search_rapport_njobs(params_dict,X_train, y_train, X_test, y_test,'rapport.csv',verbose=2, save_iteration=2, n_jobs=-1, n_random=3)
    result_df = grid_search_rapport(params_dict,X_train, y_train, X_test, y_test,'rapport2.csv',verbose=2, save_iteration=2, n_random=3)


def save_restore():
    X, y = make_classification(n_samples=1000,
                               n_features=10,
                               n_informative=10,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=4,
                               random_state=0, n_clusters_per_class=1,
                               shuffle=False)
    ncmf = NCMForest(n_trees=100, max_depth=50, min_samples_split=1, min_samples_leaf=10,
                         method_max_features=0.3, method_k_bis=0.5, method_split='eq_samples',
                         distance="euclidean")
    ncmf.fit(X,y)
    d = datetime.now()
    f = d.strftime('%Y-%m-%d_model')
    pickle.dump(ncmf, open('models/'+f+'.pkl', "wb"))
    ncmf1 = pickle.load(open('models/'+f+'.pkl', "rb"))
    print(ncmf1)
    print(ncmf1.predict(X)[1])

def test_update_centroid_without_new_classes():

    iris = load_iris()
    X = iris.data[:, 2:]  # we only take the first two features.
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    #splitting_clf = NCMClassifier('euclidean')
    clf = NCMTree('euclidean')
    clf.fit(X_train, y_train)

    fig = plt.figure(figsize=(10, 8))
    ax1=plt.subplot(211)
    ax1 = plot_decision_regions(X, y, clf=clf, legend=2)
    # Adding axes annotations
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('Decision on training data')

    ax2=plt.subplot(212)
    clf.update_centroid(X_test, y_test, importance_old=0.1)
    ax2 = plot_decision_regions(X, y, clf=clf, legend=2)
    # Adding axes annotations
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('New Decision with incremental')
    plt.show()

def test_update_centroid_with_new_classes():
    iris = load_iris()
    X = iris.data[:, 2:]  # we only take the first two features.
    y = iris.target

    X_train = X[:100, :]
    X_test = X[100:, :]
    y_train = y[:100]
    y_test = y[100:]
    print(X_train.shape)
    print(y_train.shape)
    print(y_train.shape)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    clf = NCMTree('euclidean')
    #splitting_clf = NCMClassifier('mahalanobis')
    clf.fit(X_train, y_train)
    print('fit:ok')

    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(211)
    ax1 = plot_decision_regions(X_train, y_train, clf=clf, legend=2)
    # Adding axes annotations
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('Decision on training data')

    ax2 = plt.subplot(212)
    clf.ULS(X_test, y_test)
    ax2 = plot_decision_regions(X, y, clf=clf, legend=2)
    # Adding axes annotations
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('New Decision with incremental')
    plt.show()

def get_depth_and_size(trees):
    depths = 0
    sizes = 0
    for tree in trees:
        depths = depths + tree.depth
        sizes = sizes + tree.cardinality
    return round(depths/len(trees) ,3), round(sizes/len(trees) ,3)

def test_igt():
    X, y = make_classification(n_samples=1000,
                               n_features=10,
                               n_informative=10,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=10,
                               random_state=0, n_clusters_per_class=1,
                               shuffle=False)
    ncmf = NCMForest(n_trees=8, max_depth=50, min_samples_split=1, min_samples_leaf=5,
                     method_max_features=0.5, method_subclasses=0.5, method_split='eq_samples',
                     distance="euclidean")
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.30, random_state=42)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.05, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.90, random_state=42)
    print("------start fit ------------------------------")
    ncmf.fit(X1_train, y1_train)
    ncmf2 = copy.deepcopy(ncmf)
    ncmf3 = copy.deepcopy(ncmf)

    depths, sizes = get_depth_and_size(ncmf.trees)
    print("Avg tree depth : ", depths)
    print("Avg tree size : ", sizes)
    print("NCMF train accuracy: " + str(ncmf.score(X1_train, y1_train)))
    print("NCMF test accuracy: " + str(ncmf.score(X1_test, y1_test)))
    print("NCMF new train accuracy: " + str(ncmf.score(X2_train, y2_train)))
    print("NCMF new test accuracy: " + str(ncmf.score(X2_test, y2_test)))

    print("--uls---")
    ncmf.ULS(X2_train, y2_train)
    print("NCMF train proba: " + str(ncmf.score(X1_train, y1_train)))
    print("NCMF test proba: " + str(ncmf.score(X1_test, y1_test)))
    print("NCMF new train proba: " + str(ncmf.score(X2_train, y2_train)))
    print("NCMF new test proba: " + str(ncmf.score(X2_test, y2_test)))

    print("--igt---")
    ncmf2.IGT(X2_train, y2_train)
    depths, sizes = get_depth_and_size(ncmf2.trees)
    print("Avg tree depth : ", depths)
    print("Avg tree size : ", sizes)
    print("NCMF train proba: " + str(ncmf2.score(X1_train, y1_train)))
    print("NCMF test proba: " + str(ncmf2.score(X1_test, y1_test)))
    print("NCMF new train proba: " + str(ncmf2.score(X2_train, y2_train)))
    print("NCMF new test proba: " + str(ncmf2.score(X2_test, y2_test)))

    print("--rtst---")
    ncmf3.RTST(X2_train, y2_train)
    depths, sizes = get_depth_and_size(ncmf3.trees)
    print("Avg tree depth : ", depths)
    print("Avg tree size : ", sizes)
    print("NCMF train proba: " + str(ncmf3.score(X1_train, y1_train)))
    print("NCMF test proba: " + str(ncmf3.score(X1_test, y1_test)))
    print("NCMF new train proba: " + str(ncmf3.score(X2_train, y2_train)))
    print("NCMF new test proba: " + str(ncmf3.score(X2_test, y2_test)))

def test_iris_incremental_data():
    iris_data = load_iris()
    X=iris_data["data"]
    Y=iris_data['target']
    X1,X2,Y1,Y2 = train_test_split(X,Y,train_size=0.5,random_state=42)
    
    X_train11,X_test11,Y_train11,Y_test11 = train_test_split(X1,Y1,train_size=0.7,random_state=42)
    X_train22,X_test22,Y_train22,Y_test22 = train_test_split(X2,Y2,train_size=0.7,random_state=42)

    ncmf = NCMForest(n_trees=10, max_depth=50, min_samples_split=1, min_samples_leaf=5,
                     method_max_features=0.5, method_subclasses=0.5, method_split='eq_samples',
                     distance="euclidean")
    ncmf.fit(X_train11,Y_train11)
    ncmf1 = copy.deepcopy(ncmf)
    ncmf2 = copy.deepcopy(ncmf)
    
    print("---------------Start fit-----------------")
    depths, sizes = get_depth_and_size(ncmf.trees)
    print("Avg tree depth : ", depths)
    print("Avg tree size : ", sizes)
    print("Score train",(ncmf.score(X_train11,Y_train11)))
    print("Score new train",(ncmf.score(X_train22,Y_train22)))

    print("Score test ",(ncmf.score(X_test11,Y_test11)))
    print("Score new test ",(ncmf.score(X_test22,Y_test22)))

    print("---------uls---------")
    ncmf.ULS(X_train22,Y_train22)
    depths, sizes = get_depth_and_size(ncmf.trees)
    print("Avg tree depth : ", depths)
    print("Avg tree size : ", sizes)
    print("Score new test",(ncmf.score(X_test22,Y_test22)))
    print("Score new train",(ncmf.score(X_train22,Y_train22)))
    
    
    print("---------igt---------")
    ncmf1.IGT(X_train22,Y_train22)
    depths, sizes = get_depth_and_size(ncmf1.trees)
    print("Avg tree depth : ", depths)
    print("Avg tree size : ", sizes)
    print("Score new test",(ncmf1.score(X_test22,Y_test22)))
    print("Score new train",(ncmf1.score(X_train22,Y_train22)))
    
    
    print("---------rtst---------")
    ncmf2.RTST(X_train22,Y_train22)
    depths, sizes = get_depth_and_size(ncmf2.trees)
    print("Avg tree depth : ", depths)
    print("Avg tree size : ", sizes)
    print("Score new test",(ncmf2.score(X_test22,Y_test22)))
    print("Score new train",(ncmf2.score(X_train22,Y_train22)))
    
    for i,j in enumerate(ncmf2.trees):
        #print(i.predict(np.array([1,2,5,6]).reshape(1,4)))
        #print(i.predict(X_test))
        #print(X_train1.shape)
        #print(i.root.proportion_classes,i.root.total_effectives,i.root.is_leaf,i.root.left_child)
        #print('------------------------------')
        #print(X_train1)
       # print("Tree number {}".format(i))
        continue
        #construct_tree(j.root)
        

def test_incremental_data_batch_five(dataset,datasetName):
    
    X=dataset["data"]
    Y=dataset['target']
    
    X1,X2,Y1,Y2 = train_test_split(X,Y,train_size=0.5,random_state=42)
    
    X_traintemp,X_train1,Y_traintemp,Y_train1 = train_test_split(X1,Y1,train_size=0.8,random_state=42)
    X_train2,X_train3,Y_train2,Y_train3= train_test_split(X_traintemp,Y_traintemp,train_size=0.5,random_state=42)
    X_traintemp,X_test,Y_traintemp,Y_test = train_test_split(X2,Y2,train_size=0.8,random_state=42)
    X_train4,X_train5,Y_train4,Y_train5= train_test_split(X_traintemp,Y_traintemp,train_size=0.5,random_state=42)
    
    
    tab_X_train=[X_train2,X_train3,X_train4,X_train5]
    tab_Y_train=[Y_train2,Y_train3,Y_train4,Y_train5]
   
    ncmf = NCMForest(n_trees=10, max_depth=50, min_samples_split=1, min_samples_leaf=5,
                     method_max_features=0.5, method_subclasses=0.5, method_split='eq_samples',
                     distance="euclidean")
    
    ncmf.fit(X_train1,Y_train1)
    ncmf1 = copy.deepcopy(ncmf)
    ncmf2 = copy.deepcopy(ncmf)
    
    print("           "+datasetName+" DATA")
    print("---------------Initial fit-----------------")
    
    depths, sizes = get_depth_and_size(ncmf.trees)
    print("Avg tree depth : ", depths)
    print("Avg tree size : ", sizes)
    print("    Score own train:",(ncmf.score(X_train1,Y_train1)))
    print("    Score for batch train 2:",(ncmf.score(X_train2,Y_train2)))
    print("    Score for batch train 3:",(ncmf.score(X_train3,Y_train3)))
    print("    Score for batch train 4:",(ncmf.score(X_train4,Y_train4)))
    print("    Score for batch train 5:",(ncmf.score(X_train5,Y_train5)))
   
    print("    Score test ",(ncmf.score(X_test,Y_test)))

    print("---------uls---------")
    index=1
    for x,y in zip(tab_X_train,tab_Y_train):
        print("-------- "+str(index)+ " Batch ---------")
        ncmf.ULS(x,y)
        depths, sizes = get_depth_and_size(ncmf.trees)
        print("Avg tree depth : ", depths)
        print("Avg tree size : ", sizes)
        print("   Score new train :",(ncmf.score(x,y)))
        print("   Score test :",(ncmf.score(X_test,Y_test)))
        index=index+1
    index=1
    print("---------igt---------")
    for x,y in zip(tab_X_train,tab_Y_train):
        print("-------- "+str(index)+ " Batch ---------")
        ncmf1.IGT(x,y)
        depths, sizes = get_depth_and_size(ncmf1.trees)
        print("Avg tree depth : ", depths)
        print("Avg tree size : ", sizes)
        print("    Score new train :",(ncmf1.score(x,y)))
        print("    Score test :",(ncmf1.score(X_test,Y_test)))
        index=index+1
    index=1
    print("---------rtst---------")
    for x,y in zip(tab_X_train,tab_Y_train):
        print("-------- "+str(index)+ " Batch ---------")
        ncmf2.RTST(x,y)
        depths, sizes = get_depth_and_size(ncmf2.trees)
        print("Avg tree depth : ", depths)
        print("Avg tree size : ", sizes)
        print("    Score new train :",(ncmf2.score(x,y)))
        print("    Score test",(ncmf2.score(X_test,Y_test)))
        index=index+1
        
        
def test_incremental_data_batch_three(dataset,datasetName):
    
    X=dataset["data"]
    Y=dataset['target']
    
    X1,X2,Y1,Y2 = train_test_split(X,Y,train_size=0.5,random_state=42)
    
    X_train11,X_train22,Y_train11,Y_train22 = train_test_split(X1,Y1,train_size=0.5,random_state=42)
    X_train33,X_test22,Y_train33,Y_test22 = train_test_split(X2,Y2,train_size=0.5,random_state=42)

    tab_X_train=[X_train22,X_train33]
    tab_Y_train=[Y_train22,Y_train33]
   
    ncmf = NCMForest(n_trees=10, max_depth=50, min_samples_split=1, min_samples_leaf=5,
                     method_max_features=0.5, method_subclasses=0.5, method_split='eq_samples',
                     distance="euclidean")
    ncmf.fit(X_train11,Y_train11)
    ncmf1 = copy.deepcopy(ncmf)
    ncmf2 = copy.deepcopy(ncmf)
    #ncmf1 = copy.copy(ncmf)
    #ncmf2 = copy.copy(ncmf)
    print("           "+datasetName+" DATA")
    print("---------------Start fit-----------------")
    
    depths, sizes = get_depth_and_size(ncmf.trees)
    print("Avg tree depth : ", depths)
    print("Avg tree size : ", sizes)
    print("    Score train:",(ncmf.score(X_train11,Y_train11)))
    print("    Score new train 1:",(ncmf.score(X_train22,Y_train22)))
    print("    Score new train 2:",(ncmf.score(X_train33,Y_train33)))
   
    #print("Score test ",(ncmf.score(X_test11,Y_test11)))
    print("    Score test ",(ncmf.score(X_test22,Y_test22)))

    print("---------uls---------")
    
    for x,y in zip(tab_X_train,tab_Y_train):    
        ncmf.ULS(x,y)
        depths, sizes = get_depth_and_size(ncmf.trees)
        print("Avg tree depth : ", depths)
        print("Avg tree size : ", sizes)
        print("   Score new train :",(ncmf.score(x,y)))
        #print("Score new train 2:",(ncmf.score(X_train33,Y_train33)))
        print("   Score test :",(ncmf.score(X_test22,Y_test22)))
    
    
    print("---------igt---------")
    for x,y in zip(tab_X_train,tab_Y_train):    
        ncmf1.IGT(x,y)
        depths, sizes = get_depth_and_size(ncmf1.trees)
        print("Avg tree depth : ", depths)
        print("Avg tree size : ", sizes)
        print("    Score new train :",(ncmf1.score(x,y)))
        #print("Score new train 2:",(ncmf1.score(X_train33,Y_train33)))
        print("    Score test :",(ncmf1.score(X_test22,Y_test22)))

    
    print("---------rtst---------")
    
    for x,y in zip(tab_X_train,tab_Y_train):    
        ncmf2.RTST(x,y)
        depths, sizes = get_depth_and_size(ncmf2.trees)
        print("Avg tree depth : ", depths)
        print("Avg tree size : ", sizes)
        print("    Score new train :",(ncmf2.score(x,y)))
        #print("Score new train 2:",(ncmf2.score(X_train33,Y_train33)))
        print("    Score test",(ncmf2.score(X_test22,Y_test22)))


    
    for i,j in enumerate(ncmf2.trees):
        #print(i.predict(np.array([1,2,5,6]).reshape(1,4)))
        #print(i.predict(X_test))
        #print(X_train1.shape)
        #print(i.root.proportion_classes,i.root.total_effectives,i.root.is_leaf,i.root.left_child)
        #print('------------------------------')
        #print(X_train1)
       # print("Tree number {}".format(i))
        continue
        #construct_tree(j.root)
               
        
def test_incremental_data_simple(dataset,datasetName):
    
    X=dataset["data"]
    Y=dataset['target']
    
    X1,X2,Y1,Y2 = train_test_split(X,Y,train_size=0.5,random_state=42)
    
    X_train11,X_test11,Y_train11,Y_test11 = train_test_split(X2,Y2,train_size=0.7,random_state=42)
    #X_train22,X_test22,Y_train22,Y_test22 = train_test_split(X2,Y2,train_size=0.7,random_state=42)

    ncmf = NCMForest(n_trees=20, max_depth=60, min_samples_split=1, min_samples_leaf=5,
                     method_max_features=0.5, method_subclasses=0.5, method_split='eq_samples',
                     distance="euclidean")
    ncmf.fit(X1,Y1)
    ncmf1 = copy.deepcopy(ncmf)
    ncmf2 = copy.deepcopy(ncmf)
    print("           "+datasetName+" DATA")
    print("---------------Start fit-----------------")
    depths, sizes = get_depth_and_size(ncmf.trees)
    print("Avg tree depth : ", depths)
    print("Avg tree size : ", sizes)
    print("Score train",(ncmf.score(X1,Y1)))
    print("Score new train",(ncmf.score(X_train11,Y_train11)))

    print("Score test ",(ncmf.score(X_test11,Y_test11)))
    #print("Score new test ",(ncmf.score(X_test22,Y_test22)))

    print("--------uls---------")
    ncmf.ULS(X_train11,Y_train11)
    depths, sizes = get_depth_and_size(ncmf.trees)
    print("Avg tree depth : ", depths)
    print("Avg tree size : ", sizes)
    print("Score test",(ncmf.score(X_test11,Y_test11)))
    print("Score new train",(ncmf.score(X_train11,Y_train11)))
    
    
    print("--------igt---------")
    ncmf1.IGT(X_train11,Y_train11)
    depths, sizes = get_depth_and_size(ncmf1.trees)
    print("Avg tree depth : ", depths)
    print("Avg tree size : ", sizes)
    print("Score test",(ncmf1.score(X_test11,Y_test11)))
    print("Score new train",(ncmf1.score(X_train11,Y_train11)))
    
    
    print("---------rtst---------")
    ncmf2.RTST(X_train11,Y_train11)
    depths, sizes = get_depth_and_size(ncmf2.trees)
    print("Avg tree depth : ", depths)
    print("Avg tree size : ", sizes)
    print("Score new test",(ncmf2.score(X_test11,Y_test11)))
    print("Score new train",(ncmf2.score(X_train11,Y_train11)))
   
def construct_tree(parent):
    print(parent.proportion_classes,parent.total_effectives)
    if not parent.is_leaf: 
        left_child=parent.left_child
        right_child=parent.right_child
        print("       ")
        print(left_child.parent)
        print("          <--")
        print(left_child.proportion_classes,left_child.total_effectives)
        print("     ")
        print(right_child.parent)
        print("        -->")
        print("                    ",right_child.proportion_classes,right_child.total_effectives)
    #if not parent.is_leaf:
        construct_tree(left_child)
        construct_tree(right_child)
    else:
        print("              ")
        print("Tree Builded")
        
    
    
def test_perf():
    from pympler import asizeof
    import time
    X, y = make_classification(n_samples=30000,
                               n_features=256,
                               n_informative=256,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=10,
                               random_state=0, n_clusters_per_class=1,
                               shuffle=False)

    ncmf = NCMForest(n_trees=20, max_depth=50, min_samples_split=1, min_samples_leaf=10,
                         method_max_features=0.5, method_k_bis=0.5, method_split='eq_samples',
                         distance="euclidean")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    start = time.time()
    ncmf.fit(X_train, y_train)
    end = time.time() - start
    print('END  FIT :  ', end)

    start = time.time()
    ncmf.predict(X_train)
    end = time.time() - start
    print('END  PREDICT :  ',end)

    d = datetime.now()
    f = d.strftime('%Y-%m-%d_model')+'OPTIMAL_30k_256f_10C'
    pickle.dump(ncmf, open('models/' + f + '.pkl', "wb"))
    print(asizeof.asized(ncmf, detail=3).format())

def test_ram():
    from pympler import asizeof
    ncmf1 = pickle.load(open('models/2020-01-28_model_model_30k_256f_10C.pkl', "rb"))
    print("------------ FOREST----------------")
    print(asizeof.asized(ncmf1, detail=3).format())
    print("------------ ARBRES----------------")
    print(asizeof.asized(ncmf1.trees, detail=3).format())
    print("------------ ARBRE 1----------------")
    print(asizeof.asized(ncmf1.trees[0], detail=3).format())
    print("------------ ROOT----------------")
    print(asizeof.asized(ncmf1.trees[0].root, detail=3).format())



if __name__ =="__main__":
    # print('----------- IGT -----------')
    #test_igt()
    
    
    test_incremental_data_batch_five(load_breast_cancer(),"Breast Cancer")
    #test_incremental_data_simple(load_iris(),"IRIS")
    
    # print("---------- Next Test -----------")
    # print('-----------NODE-----------')
    # test_Node()
    # print()
    # print('------------CLASSIFIER------------')
    #test_NCMClassifier()
    # print()
    # print('------------TREEE-----------')
    # test_NCMTree()
    # print()
    # print('----------FOREST------------')
    # test_NCMForest()
    # print()
    # print('----------GRIDSEARCH------------')
    #test_gridsearch()
    # test_gridsearch()
    # print('---------------------------------')
    # print('----------SAVE AND RESTORE------------')
    #save_restore()
    # print('---------------------------------')
    # test_update_centroid_without_new_classes()
    #test_update_centroid_with_new_classes()
    #test_perf()
    #test_ram()
