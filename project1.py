__author__ = 'main'

from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sknn import mlp
import numpy as np


from sklearn import metrics

import random
import pandas as pd
import datetime
import logging
NUM_ITERATIONS = 1000

def size(tree_obj):
    return tree_obj.tree_.node_count

def neural(train_x, train_y, test_x, test_y, n_epochs, iterations):
    name = "Results/neural_"+str(n_epochs)+"_results"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".csv"
    file = open(name, "w")
    file.write("Neural Network w/ n_layers = "+str(n_epochs)+" Analysis Started on "+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    file.write("Iteration, Instances, Train Time, Test Time, Training Accuracy, Testing Accuracy")

    logging.info("Starting Neural Analysis")
    outer_time = datetime.datetime.now()
    nn = mlp.Classifier(layers=[
       # mlp.Layer("Maxout", units=100, pieces=2),
        mlp.Layer("Softmax")], n_iter=n_epochs)
    for i in range(iterations):
        sample_size = int(random.uniform(0.001, 1.0)*train_y.shape[0])
        index = random.sample(xrange(0, train_y.shape[0]), sample_size)
        start = datetime.datetime.now()
        nn.fit(train_x[index], train_y[index])
        end = datetime.datetime.now()
        train_time = end - start
        train_score = nn.score(train_x, train_y)
        start = datetime.datetime.now()
        test_score = nn.score(test_x, test_y)
        test_time = datetime.datetime.now() - start

        file.write("%4d, %4d, %s, %s, %2.6f, %2.6f \n" % (i, len(index), train_time, test_time, train_score, test_score))
    logging.info("Analysis completed in %s" % (datetime.datetime.now() - outer_time))
    file.close()

def boosting(train_x, train_y, test_x, test_y, n_estimators, iterations):
    name = "Results/adaboost_"+str(n_estimators)+"_results"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".csv"
    file = open(name, "w")
    file.write("AdaBoost w/ n_estimators = "+str(n_estimators)+" Analysis Started on "+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    file.write("Iteration, Instances, Train Time, Test Time, Training Accuracy, Testing Accuracy")

    logging.info("Starting Boosting Analysis")
    outer_time = datetime.datetime.now()
    boost = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=19), n_estimators=n_estimators)
    for i in range(iterations):
        sample_size = int(random.uniform(0.001, 1.0)*train_y.shape[0])
        index = random.sample(xrange(0, train_y.shape[0]), sample_size)
        start = datetime.datetime.now()
        boost.fit(train_x[index], train_y[index])
        end = datetime.datetime.now()
        train_time = end - start
        train_score = boost.score(train_x, train_y)
        start = datetime.datetime.now()
        test_score = boost.score(test_x, test_y)
        test_time = datetime.datetime.now() - start

        file.write("%4d, %4d, %s, %s, %2.6f, %2.6f \n" % (i, len(index), train_time, test_time, train_score, test_score))
    logging.info("Analysis completed in %s" % (datetime.datetime.now() - outer_time))
    file.close()

def decision_tree(train_x, train_y, test_x, test_y, iterations):
    file = open("Results/dt_results.csv", "w")
    file.write("Analysis Stated on "+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    file.write("Depth, Iteration, Instances, Train Time, Test Time, # Nodes, Training Accuracy, Testing Accuracy")
    logging.info("Starting Decision Tree Analysis")
    outer_time = datetime.datetime.now()
    for depth in range(10, 20, 1):
        # logging.info("Starting Depth %d" % depth)
        inner_time = datetime.datetime.now()
        dt = tree.DecisionTreeClassifier(max_depth=depth)
        for i in range(iterations):
            sample_size = int(random.uniform(0.001, 1.0)*train_y.shape[0])
            index = random.sample(xrange(0, train_y.shape[0]), sample_size)
            start = datetime.datetime.now()
            dt.fit(train_x[index], train_y[index])
            end = datetime.datetime.now()
            train_time = end - start
            train_score = dt.score(train_x, train_y)
            start = datetime.datetime.now()
            test_score = dt.score(test_x, test_y)
            test_time = datetime.datetime.now() - start

            file.write("%4d, %4d, %4d, %s, %s, %d, %2.6f, %2.6f \n" % (depth,i, len(index), train_time, test_time, size(dt), train_score, test_score))
        logging.info( "Depth %4d completed in %s" %(depth, datetime.datetime.now() - inner_time))
    logging.info("Analysis completed in %s" % (datetime.datetime.now() - outer_time))
    file.close()

def k_nn(train_x, train_y, test_x, test_y, k, iterations):
    name = "Results/knn_"+str(k)+"_results"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".csv"
    file = open(name, "w")
    file.write("KNN K = "+str(k)+" Analysis Started on "+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    file.write("Iteration, Instances, Train Time, Test Time, Training Accuracy, Testing Accuracy")

    logging.info("Starting KNN Analysis")
    outer_time = datetime.datetime.now()
    knn = KNeighborsClassifier(n_neighbors=k)
    for i in range(iterations):
        sample_size = int(random.uniform(0.001, 1.0)*train_y.shape[0])
        index = random.sample(xrange(0, train_y.shape[0]), sample_size)
        start = datetime.datetime.now()
        knn.fit(train_x[index], train_y[index])
        end = datetime.datetime.now()
        train_time = end - start
        train_score = knn.score(train_x, train_y)
        start = datetime.datetime.now()
        test_score = knn.score(test_x, test_y)
        test_time = datetime.datetime.now() - start

        file.write("%4d, %4d, %s, %s, %2.6f, %2.6f \n" % (i, len(index), train_time, test_time, train_score, test_score))
    logging.info("Analysis completed in %s" % (datetime.datetime.now() - outer_time))
    file.close()

def support_vector_machine(train_x, train_y, test_x, test_y, kern, iterations):
    name = "Results/svm_"+kern+"_results.csv"
    file = open(name, "w")
    file.write("Analysis Stated on "+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    file.write("Iteration, Instances, Train Time, Test Time, Training Accuracy, Testing Accuracy")

    logging.info("Starting Support Vector Machine Analysis")
    outer_time = datetime.datetime.now()
    sv = svm.SVC(kernel=kern)
    for i in range(iterations):
        sample_size = int(random.uniform(0.001, 1.0)*train_y.shape[0])
        index = random.sample(xrange(0, train_y.shape[0]), sample_size)
        start = datetime.datetime.now()
        sv.fit(train_x[index], train_y[index])
        end = datetime.datetime.now()
        train_time = end - start
        train_score = sv.score(train_x, train_y)
        start = datetime.datetime.now()
        test_score = sv.score(test_x, test_y)
        test_time = datetime.datetime.now() - start

        file.write("%4d, %4d, %s, %s, %2.6f, %2.6f \n" % (i, len(index), train_time, test_time, train_score, test_score))
    logging.info("Analysis completed in %s" % (datetime.datetime.now() - outer_time))
    file.close()

def main():

    logname = "log/log_subject_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +".txt"
    logging.basicConfig(filename=logname, level=logging.DEBUG)
    # create the training & test sets, skipping the header row with [2:]
    x = pd.read_csv(open('Data/train_x_subject.csv', 'r'), delimiter=',').as_matrix()[:, :]
    y = pd.read_csv(open('Data/train_y_subject.csv','r')).as_matrix()[:]

    indices = np.random.permutation(y.shape[0])

    train_idx, test_idx = indices[:9000], indices[9000:]
    train_x = x[train_idx,:]
    train_y = y[train_idx,:]
    test_x = x[test_idx, :]
    test_y = y[test_idx,:]
    #
    #
    # start_analysis = datetime.datetime.now()
    # # support_vector_machine(train_x, train_y, test_x, test_y, 'linear', 1000)
    # # support_vector_machine(train_x, train_y, test_x, test_y, 'rbf', 1000)
    # #
    # #
    # # decision_tree(train_x, train_y, test_x, test_y, 1000)
    # #
    # # k_nn(train_x, train_y, test_x, test_y, 2, 1000)
    # # k_nn(train_x, train_y, test_x, test_y, 3, 1000)
    # # k_nn(train_x, train_y, test_x, test_y, 5, 1000)
    # #
    # boosting(train_x, train_y, test_x, test_y, 50, 100)
    # boosting(train_x, train_y, test_x, test_y, 100, 100)
    # boosting(train_x, train_y, test_x, test_y, 150, 100)
    # #
    # neural(train_x, train_y, test_x, test_y, 2, 100)
    # neural(train_x, train_y, test_x, test_y, 3, 100)
    # neural(train_x, train_y, test_x, test_y, 4, 100)
    # neural(train_x, train_y, test_x, test_y, 5, 100)

    print "analysis complete total time %s" % (datetime.datetime.now() - start_analysis)

if __name__=="__main__":
    main()