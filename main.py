import pandas as pd 
from scipy.stats import uniform
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import tree, linear_model
from sklearn.linear_model import LogisticRegression 
from logisticRegression import logisticRegression
from decisionTree import decisionTree

def compare(arg1, arg2):
    sum = 0
    for i in range(len(arg1)):
        if arg1[i] == arg2[i]:
            sum += 1
    return sum/len(arg1)

dt_X = []
dt_Y = []

lr_X = []
lr_Y = []

data = pd.read_csv("data/data.csv")

y = data['Y']
x = data.drop('Y', axis = 1)

for testSize in [0.1, 0.3, 0.5, 0.7, 0.9]:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testSize, random_state = 0)

    dt_X.append(100 * testSize)
    dt_Y.append(
            compare(
                decisionTree(x_train = x_train, y_train = y_train, x_test = x_test),
                list(y_test)
            )
    )

    # lr_X.append(100 * testSize)
    # lr_Y.append(
    #         compare(
    #             logisticRegression(x_train = x_train, y_train = y_train, x_test = x_test),
    #             list(y_test)
    #         )
    # )
    print(testSize)

with open("results/decisionTree.txt", "w") as file:
    for i in range(len(dt_X)):
        file.write( str(dt_X[i]) + '\n')
        file.write( str(dt_Y[i]) + '\n')

# with open("results/logisticRegression.txt", "w") as file:
#     for i in range(len(lr_X)):
#         file.write( str(lr_X[i]) + '\n')
#         file.write( str(lr_Y[i]) + '\n')
