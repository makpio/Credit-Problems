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

parameters = {
'splitter':('best', 'random'), 
'criterion':('gini', 'entropy'), 
'min_samples_splitint':[1, 10], 
'min_samples_leafint' :[1,10],
'max_featuresint':(“auto”, “sqrt”, “log2”)
}