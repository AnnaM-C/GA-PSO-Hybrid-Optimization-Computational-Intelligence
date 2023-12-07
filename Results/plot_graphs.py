
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_NSGA_vs_SGD(NSGA_file, SSPSO_file):

    df_NSGA=pd.read_csv(NSGA_file)
    NSGA_avg_accuracy_column=df_NSGA["avg"]
    NSGA_max_accuracy_column=df_NSGA["max"]

    df_SSPSO=pd.read_csv(SSPSO_file)



    plt.plot(range(100), NSGA_max_accuracy_column, label="NSGA-II Maximum", color="blue")
    # plt.plot(range(100), NSGA_avg_accuracy_column, label="NSGA-II Average", color="orange")
    plt.plot(range(100), transposed, color="pink")

    plt.xlabel('Generations')
    plt.ylabel('Maximum Accuracy')
    plt.title('Comparison of NSGA-II vs Genetic Algorithm Accuracy over Generations')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    NSGA_file= "NSGA/training_log_6.csv"
    SSPSO="SSPSO/accuracyTrack-070.csv"
    plot_NSGA_vs_SGD(NSGA_file, SSPSO)
