
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


def plot_NSGA_vs_GA(NSGA_file, GA_file):
    df_NSGA=pd.read_csv(NSGA_file)
    df_GA=pd.read_csv(GA_file)

    NSGA_max_accuracy_column=df_NSGA["max"]
    GA_max_accuracy_column=df_GA["Max Fitness"]

    plt.plot(range(100), NSGA_max_accuracy_column, label="NSGA II", color="tab:blue")
    plt.plot(range(100), GA_max_accuracy_column, label="GA", color="tab:orange")

    plt.xlabel('Generations')
    plt.ylabel('Maximum Accuracy')
    plt.title('Comparison of NSGA II vs Genetic Algorithm Maximum Accuracy')

    plt.legend()
    plt.show()

def plot_NSGA_vs_SSPSO(NSGA_file, SSPSO_file):
    df_NSGA=pd.read_csv(NSGA_file)
    NSGA_max_accuracy_column=df_NSGA["max"]


    with open(SSPSO_file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        SSPSO = list(csv_reader)
        SSPSOVals = [eval(i) for i in SSPSO[4]]

    plt.plot(range(100), NSGA_max_accuracy_column, label="NSGA II", color="tab:blue")
    plt.plot(range(100), SSPSOVals, label="SSPSO", color="tab:orange")

    plt.xlabel('Generations')
    plt.ylabel('Maximum Accuracy')
    plt.title('NSGA II vs SSPSO Algorithm Maximum Accuracy')

    plt.legend()
    plt.savefig("nsga-vs-SSPSO.pdf", bbox_inches='tight')
    plt.savefig("nsga-vs-SSPSO.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_GA_vs_SLPSO(GA_file, SLPSO_file):

    df_GA=pd.read_csv(GA_file)
    GA_max_accuracy_column=df_GA["Max Fitness"]

    with open(SLPSO_file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        SLPSO = list(csv_reader)
        SLPSOVals = [eval(i) for i in SLPSO[4]]

    plt.plot(range(100), GA_max_accuracy_column, label="GA", color="tab:blue")
    plt.plot(range(100), SLPSOVals, label="SLPSO", color="tab:orange")
    plt.xlabel('Generations')
    plt.ylabel('Maximum Accuracy')
    plt.title('GA vs SLPSO Maximum Accuracy')

    plt.legend()
    plt.savefig("ga-vs-SLPSO.pdf", bbox_inches='tight')
    plt.savefig("ga-vs-SLPSO.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_big_sigma_vs_small_sigma(SSPSO_big_sigma_file, SSPSO_small_sigma_file):
    with open(SSPSO_big_sigma_file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        SSPSO_big_sigma = list(csv_reader)
        bigSigmaVals = [eval(i) for i in SSPSO_big_sigma[4]]

    with open(SSPSO_small_sigma_file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        SSPSO_small_sigma = list(csv_reader)
        smallSigmaVals = [eval(i) for i in SSPSO_small_sigma[4]]

    plt.plot(range(100), smallSigmaVals, label="SLPSO", color="tab:blue")
    plt.plot(range(100), bigSigmaVals, label="SSPSO Big Sigma", color="tab:orange")
    plt.xlabel('Generations')
    plt.ylabel('Maximum Accuracy')
    plt.title('SSPSO Big Sigma vs SLPSO Maximum Accuracy')

    plt.legend()
    plt.savefig("big-vs-small.pdf", bbox_inches='tight')
    plt.savefig("big-vs-small.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # We have create a seperate notebook for the optimisers to produce
    # the below csv
    NSGA_file               = "training_log_NSGA_1bounds2.csv"
    GA_file                 = "evolution_stats.csv"
    SLPSO_file              = "SLPSOLogbook.csv"
    SSPSO_big_sigma_file    = "bigSigmaSSPSOLogbook.csv"
    SSPSO_small_sigma_file  = "smallSigmaSSPSOLogbook.csv"

    plot_NSGA_vs_GA(NSGA_file, GA_file)
    plot_NSGA_vs_SSPSO(NSGA_file, SSPSO_small_sigma_file)
    plot_GA_vs_SLPSO(GA_file, SLPSO_file)
    plot_big_sigma_vs_small_sigma(SSPSO_big_sigma_file, SSPSO_small_sigma_file)

