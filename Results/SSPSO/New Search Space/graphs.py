import matplotlib.pyplot as plt
import numpy as np
import csv
import os

generationCols = 1

generationList = []
averageLoss = []

minLoss = []

with open('logbook-040.csv', mode = 'r')as file:
    csvFile = csv.reader(file)

    rows = list(csvFile)

    for i in rows[0]:
        generationList.append(int(i))

    for i in rows[2]:
        averageLoss.append(float(i))

    for i in rows[4]:
        minLoss.append(float(i))

    #print(generationList)
    #print(averageLoss)
    #print(minLoss)

generationsList = np.array(generationList)
averageLoss = np.array(averageLoss)

plt.plot(generationList, averageLoss, label='Average Loss')
plt.xlabel('Generation')
plt.ylabel('Average Loss')
# plt.ylim(0, 30)
plt.title('Average Loss vs. Generation')
plt.legend()
plt.show()


plt.plot(generationList, minLoss, label = 'Minimum Loss')
plt.xlabel('Generation')
plt.ylabel('Minimum Loss')
# plt.ylim(0, 30)
plt.title('Minimum Loss vs. Generation')
plt.legend()
plt.show()


accList = []

with open('accuracyTrack-040.csv', mode = 'r')as file:
    CSV2 = csv.reader(file)

    rows = list(CSV2)

    for i in rows[0]:
        accList.append(float(i))


#print(accList)

plt.plot(generationList, accList, label = 'Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Generation')
plt.legend()
plt.show()


SGDAcc = []
SGDGen = []


with open('SDG_training.csv', mode = 'r')as file:
    reader = csv.DictReader(file)

    # column_values = [row['accuracy'] for row in reader]

    SGDAcc.append([row['accuracy'] for row in reader])

    # # row = list(reader)

    # for i in row['accuracy']:
    #     SGDAcc.append(float(i))


SGDAcc = SGDAcc[0]
SGDAcc = [float(value) for value in SGDAcc]
print(SGDAcc)


with open('SDG_training.csv', mode = 'r')as file:
    reader = csv.DictReader(file)

    # column_values = [row['accuracy'] for row in reader]

    SGDGen.append([row['nepochs'] for row in reader])


SGDGen = SGDGen[0]
SGDGen = [int(value) for value in SGDGen]
print(SGDGen)

plt.plot(generationList, accList, label = 'Accuracy')
plt.plot(SGDGen, SGDAcc, label = 'Accuracy')
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



