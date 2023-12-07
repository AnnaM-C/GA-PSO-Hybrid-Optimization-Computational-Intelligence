
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_NSGA_vs_SGD(NSGA_file, SGD_file):

    df_NSGA=pd.read_csv(NSGA_file)
    NSGA_avg_accuracy_column=df_NSGA["max"]
    NSGA_max_accuracy_column=df_NSGA["max"]


    plt.plot(range(100), NSGA_max_accuracy_column, label="NSGA-II Maximum", color="blue")
    plt.plot(range(100), NSGA_avg_accuracy_column, label="NSGA-II Average", color="orange")

    plt.xlabel('Generations')
    plt.ylabel('Maximum Accuracy')
    plt.title('Comparison of NSGA-II vs Genetic Algorithm Accuracy over Generations')

    plt.legend()
    plt.show()

def plot_NSGA_vs_GA(NSGA_file, GA_file):
    df_NSGA=pd.read_csv(NSGA_file)
    df_GA=pd.read_csv(GA_file)

    NSGA_max_accuracy_column=df_NSGA["max"]
    GA_max_accuracy_column=df_GA["Max Fitness"]

    plt.plot(range(100), NSGA_max_accuracy_column, label="NSGA-II", color="tab:blue")
    plt.plot(range(100), GA_max_accuracy_column, label="GA", color="tab:orange")

    plt.xlabel('Generations')
    plt.ylabel('Maximum Accuracy')
    plt.title('Comparison of NSGA-II vs Genetic Algorithm Accuracy over Generations')

    plt.legend()
    plt.show()



if __name__ == "__main__":
    NSGA_file   = "NSGA/training_log_NSGA_1bounds.csv"
    SSPSO       = "SSPSO/accuracyTrack-070.csv"
    GA_file     = "GA/evolution_stats.csv"
    # plot_NSGA_vs_SGD(NSGA_file, NSGA_file)
    plot_NSGA_vs_GA(NSGA_file, GA_file)

