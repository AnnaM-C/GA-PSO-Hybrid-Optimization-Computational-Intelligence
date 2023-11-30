import matplotlib.pyplot as plt
import numpy as np
import csv

generationCols = 1

generationList = []
averageLoss = []

minLoss = []

with open('logbook-030.csv', mode = 'r')as file:
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
plt.title('Average Loss vs. Generation')
plt.legend()
plt.show()


plt.plot(generationList, minLoss, label = 'Minimum Loss')
plt.xlabel('Generation')
plt.ylabel('Minimum Loss')
plt.title('Minimum Loss vs. Generation')
plt.legend()
plt.show()
