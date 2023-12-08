
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

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

def plot_NSGA_vs_SSPSO(NSGA_file, SSPSO_file):
    df_NSGA=pd.read_csv(NSGA_file)
    NSGA_max_accuracy_column=df_NSGA["max"]


    with open(SSPSO_file, 'r') as read_obj: # read csv file as a list of lists
        csv_reader = csv.reader(read_obj) # pass the file object to reader() to get the reader object
        SSPSO = list(csv_reader) # Pass reader object to list() to get a list of lists
        SSPSOVals = [eval(i) for i in SSPSO[4]]

    plt.plot(range(100), NSGA_max_accuracy_column, label="NSGA II", color="tab:blue")
    plt.plot(range(100), SSPSOVals, label="SSPSO", color="tab:orange")

    plt.xlabel('Generations')
    plt.ylabel('Maximum Accuracy')
    plt.title('NSGA II vs Genetic Algorithm Maximum Accuracy')

    plt.legend()
    plt.savefig("nsga-vs-SSPSO.pdf", bbox_inches='tight')
    plt.savefig("nsga-vs-SSPSO.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_GA_vs_SLPSO(GA_file, SLPSO_file):

    df_GA=pd.read_csv(GA_file)
    GA_max_accuracy_column=df_GA["Max Fitness"]

    with open(SLPSO_file, 'r') as read_obj: # read csv file as a list of lists
        csv_reader = csv.reader(read_obj) # pass the file object to reader() to get the reader object
        SLPSO = list(csv_reader) # Pass reader object to list() to get a list of lists
        SLPSOVals = [eval(i) for i in SLPSO[4]]

    plt.plot(range(100), GA_max_accuracy_column, label="GA", color="tab:blue")
    plt.plot(range(100), SLPSOVals, label="SLPSO", color="tab:orange")
    plt.xlabel('Generations')
    plt.ylabel('Maximum Accuracy')
    plt.title('GA vs SLPSO')

    plt.legend()
    plt.savefig("ga-vs-SLPSO.pdf", bbox_inches='tight')
    plt.savefig("ga-vs-SLPSO.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    NSGA_file   = "/Users/annacarter/Library/Mobile Documents/com~apple~CloudDocs/Documents/Semester One/COMPUTATIONAL INTELLIGENCE COM3013/CW/computational-intelligence/Final Models/NSGA2/training_log_NSGA_1bounds2.csv"
    # NSGA_file   = "/Users/annacarter/Library/Mobile Documents/com~apple~CloudDocs/Documents/Semester One/COMPUTATIONAL INTELLIGENCE COM3013/CW/computational-intelligence/Results/NSGA/training_log_NSGA_1bounds.csv"
    SGD         = "SSPSO/accuracyTrack-070.csv"
    GA_file     = "GA/evolution_stats.csv"
    SSPSO_file  = "/Users/annacarter/Library/Mobile Documents/com~apple~CloudDocs/Documents/Semester One/COMPUTATIONAL INTELLIGENCE COM3013/CW/computational-intelligence/Final Models/SSPSO/0-SSPSO/bestLogbookNE.csv"
    SLPSO_file  = "/Users/annacarter/Library/Mobile Documents/com~apple~CloudDocs/Documents/Semester One/COMPUTATIONAL INTELLIGENCE COM3013/CW/computational-intelligence/Final Models/SLPSO/0-SLPSO/bestLogbookNE.csv"
    # plot_NSGA_vs_SGD(NSGA_file, NSGA_file)
    # plot_NSGA_vs_GA(NSGA_file, GA_file)
    # plot_NSGA_vs_SSPSO(NSGA_file, SSPSO_file)
    plot_GA_vs_SLPSO(GA_file, SLPSO_file)

