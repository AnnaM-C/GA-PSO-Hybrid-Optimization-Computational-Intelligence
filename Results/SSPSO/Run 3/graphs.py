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

with open('logbook-050.csv', mode = 'r')as file:
    csvFile = csv.reader(file)

    rows = list(csvFile)

    for i in rows[0]:
        generationList.append(int(i))

    for i in rows[2]:
        averageLoss.append(float(i))

    for i in rows[4]:
        minLoss.append(float(i))


with open('logbook-SLPSO.csv', mode = 'r')as file:
    csvFile = csv.reader(file)

    rows = list(csvFile)

    for i in rows[0]:
        SLGens.append(int(i))

    for i in rows[2]:
        SLavg.append(float(i))

    for i in rows[4]:
        SLminloss.append(float(i))

generationsList = np.array(generationList)
averageLoss = np.array(averageLoss)

plt.plot(generationList, averageLoss, label='SS-PSO')
plt.plot(SLGens, SLavg, label='SL-PSO')
plt.xlabel('Generation')
plt.ylabel('Average Loss')
# plt.ylim(0, 30)
plt.title('Average Loss vs. Generation')
plt.legend()
plt.show()


plt.plot(generationList, minLoss, label = 'SS-PSO')
plt.plot(SLGens, SLminloss, label = 'SL-PSO')
plt.xlabel('Generation')
plt.ylabel('Minimum Loss')
# plt.ylim(0, 30)
plt.title('Minimum Loss vs. Generation')
plt.legend()
plt.show()


accList = []

SLaccList = []

with open('accuracyTrack-050.csv', mode = 'r')as file:
    CSV2 = csv.reader(file)

    rows = list(CSV2)

    for i in rows[0]:
        accList.append(float(i))

with open('accuracy-SLPSO.csv', mode = 'r')as file:
    CSV2 = csv.reader(file)

    rows = list(CSV2)

    for i in rows[0]:
        SLaccList.append(float(i))


#print(accList)

plt.plot(generationList, accList, label = 'SS-PSO')
plt.plot(SLGens, SLaccList, label = 'SL-PSO')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Generation')
plt.legend()
plt.show()


SLScore = 38.53
SSScore = 51.54

Names = ['SS-PSO', 'SL-PSO']
Scores = [51.54, 38.53]
colours = ['blue', 'orange']

plt.bar(Names, Scores, color = colours)
plt.xlabel('Optimisers')
plt.ylabel('Score (%)')
plt.title('Optimiser score on unseen Test Set')
plt.show()



# SGDAcc = []
# SGDGen = []


# with open('SDG_training.csv', mode = 'r')as file:
#     reader = csv.DictReader(file)

#     # column_values = [row['accuracy'] for row in reader]

#     SGDAcc.append([row['accuracy'] for row in reader])

#     # # row = list(reader)

#     # for i in row['accuracy']:
#     #     SGDAcc.append(float(i))


# SGDAcc = SGDAcc[0]
# SGDAcc = [float(value) for value in SGDAcc]
# print(SGDAcc)


# with open('SDG_training.csv', mode = 'r')as file:
#     reader = csv.DictReader(file)

#     # column_values = [row['accuracy'] for row in reader]

#     SGDGen.append([row['nepochs'] for row in reader])


# SGDGen = SGDGen[0]
# SGDGen = [int(value) for value in SGDGen]
# print(SGDGen)

# plt.plot(generationList, accList, label = 'Accuracy')
# plt.plot(SGDGen, SGDAcc, label = 'Accuracy')
# plt.xlabel('Generation')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs. Generation')
# plt.legend()
# plt.show()


# plt.plot(generationList, accList, label = 'Accuracy')
# plt.xlabel('Generation')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs. Generation')
# plt.legend()
# plt.show()



