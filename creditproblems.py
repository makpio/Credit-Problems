import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import tree

data = pd.read_csv("data.csv")

y = data['Y']
x = data.drop('Y', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

print(clf.predict(x_test))


