__author__ = 'main'

import matplotlib.pyplot as plt
import pandas as pd
import fnmatch
import os
from numpy import savetxt
import numpy as np

def plot_boosting():
    boosting_50 = pd.read_csv(open('Results/adaboost_50_results20150917-225844.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]

    plt.figure()
    plt.scatter(boosting_50[:, 0], boosting_50[:, 1], c='blue', label='Training')
    plt.scatter(boosting_50[:, 0], boosting_50[:, 2], c='red', label='Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('Adaboost w/ 50 weak learners')
    plt.legend()
    plt.grid(True)
    plt.savefig('Plots/adaboost_50_learning_curve_subject.png', format='png')
    plt.figure()

    boosting_100 = pd.read_csv(open('Results/adaboost_100_results20150917-230150.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]

    plt.scatter(boosting_100[:, 0], boosting_100[:, 1], c='blue', label='Training')
    plt.scatter(boosting_100[:, 0], boosting_100[:, 2], c='red', label='Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('Adaboost w/ 100 weak learners')
    plt.legend()
    plt.grid(True)
    plt.savefig('Plots/adaboost_100_learning_curve_subject.png', format='png')
    plt.figure()

    boosting_150 = pd.read_csv(open('Results/adaboost_150_results20150917-230820.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]

    plt.scatter(boosting_150[:, 0], boosting_150[:, 1], c='blue', label='Training')
    plt.scatter(boosting_150[:, 0], boosting_150[:, 2], c='red', label='Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('Adaboost w/ 150 weak learners')
    plt.legend()
    plt.grid(True)
    plt.savefig('Plots/adaboost_150_learning_curve_subject.png', format='png')
    plt.figure()

    # Combined Learning Curve Graph
    plt.scatter(boosting_50[:, 0], boosting_50[:, 1], c='b', label='Boost 50 - Training')
    plt.scatter(boosting_50[:, 0], boosting_50[:, 2], c='g', label='Boost 50- Testing')
    plt.scatter(boosting_100[:, 0], boosting_100[:, 1], c='r', label='Boost 100 - Training')
    plt.scatter(boosting_100[:, 0], boosting_100[:, 2], c='c', label='Boost 100 - Testing')
    plt.scatter(boosting_150[:, 0], boosting_150[:, 1], c='m', label='Boost 150 - Training')
    plt.scatter(boosting_150[:, 0], boosting_150[:, 2], c='y', label='Boost 150 - Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('Boosting w/ 50, 100, 150 weak learners')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('Plots/boosting_learning_curve_subject.png', format='png')

def plot_svm():
    svm_linear = pd.read_csv(open('Results/svm_linear_results.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]

    plt.figure()
    plt.scatter(svm_linear[:, 0], svm_linear[:, 1], c='blue', label='Training')
    plt.scatter(svm_linear[:, 0], svm_linear[:, 2], c='red', label='Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('SVM, Linear Kernel')
    plt.legend()
    plt.grid(True)
    plt.savefig('Plots/svm_linear_learning_curve_subject.png', format='png')

    # svm_sigmoid = pd.read_csv(open('Results/svm_sigmoid_results.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]
    # plt.figure()
    # plt.scatter(svm_sigmoid[:, 0], svm_sigmoid[:, 1], c='blue', label='Training')
    # plt.scatter(svm_sigmoid[:, 0], svm_sigmoid[:, 2], c='red', label='Testing')
    # plt.xlabel('# of Training Instances')
    # plt.ylabel('Accuracy')
    # plt.title('SVM, Sigmoid Kernel')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('Plots/svm_sigmoid_learning_curve_subject.png', format='png')

    svm_rbf = pd.read_csv(open('Results/svm_rbf_results.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]
    plt.figure()
    plt.scatter(svm_rbf[:, 0], svm_rbf[:, 1], c='blue', label='Training')
    plt.scatter(svm_rbf[:, 0], svm_rbf[:, 2], c='red', label='Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('SVM, RBF Kernel')
    plt.legend()
    plt.grid(True)
    plt.savefig('Plots/svm_rbf_learning_curve_subject.png', format='png')

def plot_neural():
    neural_2 = pd.read_csv(open('Results/neural_2_results20150920-215951.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]
    neural_3 = pd.read_csv(open('Results/neural_3_results20150920-220431.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]
    neural_4 = pd.read_csv(open('Results/neural_4_results20150920-221110.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]
    neural_5 = pd.read_csv(open('Results/neural_5_results20150920-221944.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]

    # Neural 2 epochs Learning Curve
    plt.figure()
    plt.scatter(neural_2[:, 0], neural_2[:, 1], c='b', label='2 Epochs - Training')
    plt.scatter(neural_2[:, 0], neural_2[:, 2], c='g', label='2 Epochs - Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('MLP Epochs = 2')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('Plots/neural_2_learning_curve_subject.png', format='png')
    plt.figure()

    # neural_3 epochs Learning Curve
    plt.scatter(neural_3[:, 0], neural_3[:, 1], c='r', label='3 Epochs - Training')
    plt.scatter(neural_3[:, 0], neural_3[:, 2], c='c', label='3 Epochs - Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('MLP Epochs = 3')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('Plots/neural_3_learning_curve_subject.png', format='png')
    plt.figure()

    # neural_4 epochs Learning Curve
    plt.scatter(neural_4[:, 0], neural_4[:, 1], c='m', label='4 Epochs - Training')
    plt.scatter(neural_4[:, 0], neural_4[:, 2], c='y', label='4 Epochs - Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('MLP Epochs = 4')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('Plots/neural_4_learning_curve_subject.png', format='png')
    plt.figure()

    # Neural 2 epochs Learning Curve
    plt.scatter(neural_5[:, 0], neural_5[:, 1], c='m', label='5 Epochs - Training')
    plt.scatter(neural_5[:, 0], neural_5[:, 2], c='y', label='5 Epochs - Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('MLP Epochs = 5')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('Plots/neural_5_learning_curve_subject.png', format='png')
    plt.figure()

    # Combined Learning Curve Graph
    plt.scatter(neural_2[:, 0], neural_2[:, 1], c='b', label='2 Epochs - Training')
    plt.scatter(neural_2[:, 0], neural_2[:, 2], c='g', label='2 Epochs - Testing')
    plt.scatter(neural_3[:, 0], neural_3[:, 1], c='r', label='3 Epochs - Training')
    plt.scatter(neural_3[:, 0], neural_3[:, 2], c='c', label='3 Epochs - Testing')
    plt.scatter(neural_4[:, 0], neural_4[:, 1], c='m', label='4 Epochs - Training')
    plt.scatter(neural_4[:, 0], neural_4[:, 2], c='y', label='4 Epochs - Testing')
    plt.scatter(neural_5[:, 0], neural_5[:, 1], c='b', label='5 Epochs - Training')
    plt.scatter(neural_5[:, 0], neural_5[:, 2], c='0.5', label='5 Epochs - Testing')

    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('MLP w/ 2, 3, 4, and 5 Epochs')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('Plots/mlp_learning_curve.png', format='png')

def plot_dt():
    decision = pd.read_csv(open('Results/dt_results.csv', 'r'), skiprows=0, usecols=[0, 2, 6, 7], delimiter=',').as_matrix()[:, :]

    plt.figure()
    for i in range(3,10,1):
        dplot = np.where(decision[:, 0]==i)
        plt.scatter(decision[dplot, 1], decision[dplot, 2], c=np.random.rand(3,1), label='Decision Training')
        plt.xlabel('# of Training Instances')
        plt.ylabel('Accuracy')
        plt.title('Decision Tree Training')
        # plt.legend(loc=4)
        plt.grid(True)
    plt.savefig('Plots/dt_train_learning_curve_subject.png', format='png')
    plt.figure()
    for i in range(3,10,1):
        dplot = np.where(decision[:, 0]==i)
        plt.scatter(decision[dplot, 1], decision[dplot, 3], c=np.random.rand(3,1), label='Decision Training')
        plt.xlabel('# of Training Instances')
        plt.ylabel('Accuracy')
        plt.title('Decision Tree Test')
        # plt.legend(loc=4)
        plt.grid(True)
    plt.savefig('Plots/dt_test_learning_curve_subject.png', format='png')


def knn_plot():
    knn_2 = pd.read_csv(open('Results/knn_2_results20150917-225745.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]
    knn_5 = pd.read_csv(open('Results/knn_5_results20150917-225822.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]
    knn_10 = pd.read_csv(open('Results/knn_10_results20150917-224501.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]

    # K = 2 Learning Curve
    plt.figure()
    plt.scatter(knn_2[:, 0], knn_2[:, 1], c='b', label='KNN 2 - Training')
    plt.scatter(knn_2[:, 0], knn_2[:, 2], c='g', label='KNN 2 - Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('KNN K = 2')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('Plots/knn_2_learning_curve_subject.png', format='png')
    plt.figure()

    # K = 5 Learning Curve
    plt.scatter(knn_5[:, 0], knn_5[:, 1], c='r', label='KNN 5 - Training')
    plt.scatter(knn_5[:, 0], knn_5[:, 2], c='c', label='KNN 5 - Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('KNN K = 3')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('Plots/knn_5_learning_curve_subject.png', format='png')
    plt.figure()

    # K = 10 Learning Curve
    plt.scatter(knn_10[:, 0], knn_10[:, 1], c='m', label='KNN 10 - Training')
    plt.scatter(knn_10[:, 0], knn_10[:, 2], c='y', label='KNN 10 - Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('KNN K = 5')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('Plots/knn_10_learning_curve_subject.png', format='png')
    plt.figure()

    # Combined Learning Curve Graph
    plt.scatter(knn_2[:, 0], knn_2[:, 1], c='b', label='KNN 2 - Training')
    plt.scatter(knn_2[:, 0], knn_2[:, 2], c='g', label='KNN 2 - Testing')
    plt.scatter(knn_5[:, 0], knn_5[:, 1], c='r', label='KNN 5 - Training')
    plt.scatter(knn_5[:, 0], knn_5[:, 2], c='c', label='KNN 5 - Testing')
    plt.scatter(knn_10[:, 0], knn_10[:, 1], c='m', label='KNN 10 - Training')
    plt.scatter(knn_10[:, 0], knn_10[:, 2], c='y', label='KNN 10 - Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('KNN K = 2, 5, 10')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('Plots/knn_learning_curve.png', format='png')

def main():
    #knn_plot()
    # plot_svm()
    # plot_boosting()
    plot_neural()
    # plot_dt()


if __name__=="__main__":
    main()