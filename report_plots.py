__author__ = 'main'

import matplotlib.pyplot as plt
import pandas as pd
from numpy import savetxt



def main():


    svm_linear = pd.read_csv(open('Results/svm_linear_results.csv', 'r'), skiprows=0, usecols=[1, 4, 5], delimiter=',').as_matrix()[:, :]


if __name__=="__main__":
    main()