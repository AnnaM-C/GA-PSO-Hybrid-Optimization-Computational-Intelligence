import matplotlib.pyplot as plt
import numpy as np
import csv
import os

generationCols = 1

generationList = []
averageLoss = []

SLGens = []
SLavg = []

SLminloss = []

minLoss = []

with open('logbook-070.csv', mode = 'r')as file:
    csvFile = csv.reader(file)

    rows = list(csvFile)

    for i in rows[0]:
        generationList.append(int(i))

    for i in rows[2]:
        averageLoss.append(float(i))

    for i in rows[4]:
        minLoss.append(float(i))


# with open('logbook-SLPSO.csv', mode = 'r')as file:
#     csvFile = csv.reader(file)

#     rows = list(csvFile)

#     for i in rows[0]:
#         SLGens.append(int(i))

#     for i in rows[2]:
#         SLavg.append(float(i))

#     for i in rows[4]:
#         SLminloss.append(float(i))

generationsList = np.array(generationList)
averageLoss = np.array(averageLoss)

plt.plot(generationList, averageLoss, label='SS-PSO')
# plt.plot(SLGens, SLavg, label='SL-PSO')
plt.xlabel('Generation')
plt.ylabel('Average Loss')
# plt.ylim(0, 30)
plt.title('Average Loss vs. Generation')
plt.legend()
plt.show()


plt.plot(generationList, minLoss, label = 'SS-PSO')
# plt.plot(SLGens, SLminloss, label = 'SL-PSO')
plt.xlabel('Generation')
plt.ylabel('Minimum Loss')
# plt.ylim(0, 30)
plt.title('Minimum Loss vs. Generation')
plt.legend()
plt.show()


accList = []

SLaccList = []

with open('accuracyTrack-070.csv', mode = 'r')as file:
    CSV2 = csv.reader(file)

    rows = list(CSV2)

    for i in rows[0]:
        accList.append(float(i))

# with open('accuracy-SLPSO.csv', mode = 'r')as file:
#     CSV2 = csv.reader(file)

#     rows = list(CSV2)

#     for i in rows[0]:
#         SLaccList.append(float(i))


#print(accList)

plt.plot(generationList, accList, label = 'SS-PSO')
# plt.plot(SLGens, SLaccList, label = 'SL-PSO')
plt.xlabel('Generation')
plt.ylabel('Accuracy (%)')
plt.title('Training accuracy vs. Generation')
plt.legend()
plt.show()


# SLScore = 38.53
# SSScore = 51.54

# Names = ['SS-PSO', 'SL-PSO']
# Scores = [51.54, 38.53]
# colours = ['blue', 'orange']

# plt.bar(Names, Scores, color = colours)
# plt.xlabel('Optimisers')
# plt.ylabel('Accuracy (%)')
# plt.title('Test set accuracy')
# plt.show()



NSGAAcc = []
NSGAGen = []


with open('training_log_6.csv', mode = 'r')as file:
    reader = csv.DictReader(file)

    # column_values = [row['accuracy'] for row in reader]

    NSGAAcc.append([row['max'] for row in reader])

    # # row = list(reader)

    # for i in row['accuracy']:
    #     SGDAcc.append(float(i))


NSGAAcc = NSGAAcc[0]
NSGAAcc = [float(value) for value in NSGAAcc]
print(NSGAAcc)


with open('training_log_6.csv', mode = 'r')as file:
    reader = csv.DictReader(file)

    # column_values = [row['accuracy'] for row in reader]

    NSGAGen.append([row['gen'] for row in reader])


NSGAGen = NSGAGen[0]
NSGAGen = [int(value) for value in NSGAGen]
print(NSGAGen)

plt.plot(generationList, accList, label = 'SSPSO')
plt.plot(NSGAGen, NSGAAcc, label = 'NSGA-II')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Generation')
plt.legend()
plt.show()


# plt.plot(generationList, accList, label = 'Accuracy')
# plt.xlabel('Generation')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs. Generation')
# plt.legend()
# plt.show()



