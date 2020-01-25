import matplotlib.pyplot as plt
from math import sqrt , exp


dt_x = []
dt_y = []

lr_x = []
lr_y = []

with open("results/decisionTree.txt", "r") as file:
    while True:
        s1 = file.readline()
        if len(s1) == 0:
            break
        s2 = file.readline()
        if len(s2) == 0:
            break
        dt_x.append(float(s1))
        dt_y.append(float(s2) * 100)


with open("results/logisticRegression.txt", "r") as file:
    while True:
        s1 = file.readline()
        if len(s1) == 0:
            break
        s2 = file.readline()
        if len(s2) == 0:
            break
        lr_x.append(float(s1))
        lr_y.append(float(s2) * 100)

plt.plot(dt_x, dt_y, label = 'Decision tree')
plt.plot(lr_x, lr_y, label = 'Logistic regression')

plt.ylabel('Procent dobrych klasyfikacji')
plt.xlabel('Procent danych testowych')

plt.title('Porownanie')
plt.legend()

plt.show()