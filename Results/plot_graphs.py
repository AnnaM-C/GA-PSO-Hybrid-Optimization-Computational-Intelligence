
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_NSGA_vs_SGD(NSGA_file, SGD_file):

    df_NSGA=pd.read_csv(NSGA_file)
    NSGA_max_accuracy_column=df_NSGA["max"]

    df_SGD=pd.read_csv(SGD_file)
    SGD_max_accuracy_column=df_SGD["accuracy"]

    plt.plot(range(100), SGD_max_accuracy_column)
    plt.plot(range(100), NSGA_max_accuracy_column)

    plt.show()



if __name__ == "__main__":
    NSGA_file= "training_log_3.csv"
    SGD_file="SDG_training_log.csv"
    plot_NSGA_vs_SGD(NSGA_file, SGD_file)
