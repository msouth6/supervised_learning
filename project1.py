__author__ = 'main'

from sklearn import tree
from sklearn import svm
import random
import pandas as pd
import datetime
import logging
NUM_ITERATIONS = 1000

def size(tree_obj):
    return tree_obj.tree_.node_count

def decision_tree(train_x, train_y, test_x, test_y):
    file = open("dt_results.csv", "w")
    file.write("Depth, Iteration, Instances, Train Time, Test Time, # Nodes, Training Accuracy, Testing Accuracy")
    logging.info("Starting Decision Tree Analysis")
    outer_time = datetime.datetime.now()
    for depth in range(2, 20, 1):
        print "Starting Depth %d" % depth
        inner_time = datetime.datetime.now()
        dt = tree.DecisionTreeClassifier(max_depth=depth)
        for i in range(NUM_ITERATIONS):
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

def support_vector_machine(train_x, train_y, test_x, test_y, kern, iterations):
    name = "svm_"+kern+"_results.csv"
    file = open(name, "w")
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

    logname = "log"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +".txt"
    logging.basicConfig(filename=logname, level=logging.DEBUG)
    # create the training & test sets, skipping the header row with [2:]
    train_x = pd.read_csv(open('Data/train_x.csv', 'r'), delimiter=',').as_matrix()[:, :]
    train_y = pd.read_csv(open('Data/train_y.csv','r')).as_matrix()[:]

    test_x = pd.read_csv(open('Data/XY_test_17.csv', 'r'), skiprows=2, dtype='f', delimiter=',').as_matrix()[:, 1:]
    test_y = pd.read_csv(open('Data/XY_test_17.csv', 'r'), skiprows=2, dtype='f', delimiter=',').as_matrix()[:,0]

    support_vector_machine(train_x, train_y, test_x, test_y, 'linear', 1000)
    support_vector_machine(train_x, train_y, test_x, test_y, 'sigmoid', 1000)

if __name__=="__main__":
    main()