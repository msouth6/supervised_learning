__author__ = 'main'

import matplotlib.pyplot as plt
import pandas as pd
import fnmatch
import os
from numpy import savetxt

def plot_svm():
    svm_linear = pd.read_csv(open('Results/svm_linear_results.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]

    plt.scatter(svm_linear[:, 0], svm_linear[:, 1], c='blue', label='Training')
    plt.scatter(svm_linear[:, 0], svm_linear[:, 2], c='red', label='Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('SVM, Linear Kernel')
    plt.legend()
    plt.grid(True)
    plt.savefig('Plots/svm_linear_learning_curve.png', format='png')

    svm_sigmoid = pd.read_csv(open('Results/svm_sigmoid_results.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]
    plt.figure()
    plt.scatter(svm_sigmoid[:, 0], svm_sigmoid[:, 1], c='blue', label='Training')
    plt.scatter(svm_sigmoid[:, 0], svm_sigmoid[:, 2], c='red', label='Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('SVM, Sigmoid Kernel')
    plt.legend()
    plt.grid(True)
    plt.savefig('Plots/svm_sigmoid_learning_curve.png', format='png')

    svm_rbf = pd.read_csv(open('Results/svm_rbf_results.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]
    plt.figure()
    plt.scatter(svm_rbf[:, 0], svm_rbf[:, 1], c='blue', label='Training')
    plt.scatter(svm_rbf[:, 0], svm_rbf[:, 2], c='red', label='Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('SVM, RBF Kernel')
    plt.legend()
    plt.grid(True)
    plt.savefig('Plots/svm_rbf_learning_curve.png', format='png')

def knn_plot():
    knn_2 = pd.read_csv(open('Results/knn_2_results20150917-225439.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]
    knn_5 = pd.read_csv(open('Results/knn_3_results20150917-225456.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]
    knn_10 = pd.read_csv(open('Results/knn_5_results20150917-225515.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]

    # K = 2 Learning Curve
    plt.figure()
    plt.scatter(knn_2[:, 0], knn_2[:, 1], c='b', label='KNN 2 - Training')
    plt.scatter(knn_2[:, 0], knn_2[:, 2], c='g', label='KNN 2 - Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('KNN K = 2')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('Plots/knn_2_learning_curve.png', format='png')
    plt.figure()

    # K = 5 Learning Curve
    plt.scatter(knn_5[:, 0], knn_2[:, 1], c='r', label='KNN 5 - Training')
    plt.scatter(knn_5[:, 0], knn_2[:, 2], c='c', label='KNN 5 - Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('KNN K = 5')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('Plots/knn_5_learning_curve.png', format='png')
    plt.figure()

    # K = 10 Learning Curve
    plt.scatter(knn_10[:, 0], knn_2[:, 1], c='m', label='KNN 10 - Training')
    plt.scatter(knn_10[:, 0], knn_2[:, 2], c='y', label='KNN 10 - Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('KNN K = 10')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('Plots/knn_10_learning_curve.png', format='png')
    plt.figure()

    # Combined Learning Curve Graph
    plt.scatter(knn_2[:, 0], knn_2[:, 1], c='b', label='KNN 2 - Training')
    plt.scatter(knn_2[:, 0], knn_2[:, 2], c='g', label='KNN 2 - Testing')
    plt.scatter(knn_5[:, 0], knn_2[:, 1], c='r', label='KNN 5 - Training')
    plt.scatter(knn_5[:, 0], knn_2[:, 2], c='c', label='KNN 5 - Testing')
    plt.scatter(knn_10[:, 0], knn_2[:, 1], c='m', label='KNN 10 - Training')
    plt.scatter(knn_10[:, 0], knn_2[:, 2], c='y', label='KNN 10 - Testing')
    plt.xlabel('# of Training Instances')
    plt.ylabel('Accuracy')
    plt.title('KNN K = 2, 5, 10')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('Plots/knn_learning_curve.png', format='png')

def main():
    knn_plot()



if __name__=="__main__":
    main()