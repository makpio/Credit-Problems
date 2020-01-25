import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree, linear_model


def compare(arg1, arg2):
    sum = 0
    for i in range(len(arg1)):
        if arg1[i] == arg2[i]:
            sum += 1
    return sum/len(arg1)

data = pd.read_csv("data.csv")

y = data['Y']
x = data.drop('Y', axis = 1)

# for i in (0.33, 0.5, 0.75):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = i, random_state = 0)
#     parameters = {
#         'splitter' : ('best', 'random'), 
#         'criterion' : ('gini', 'entropy'),
#         'max_depth' : [1, None],
#         'min_samples_split' : [2, 5],
#         'min_samples_leaf' : [1, 5],
#         'min_weight_fraction_leaf' : [0.0, 0.5],
#         'max_features' : [1, None],
#         # 'random_state' : [1, None],
#         # 'max_leaf_nodes' : [2, 10],
#         'ccp_alpha' : [0.0, 0.5]
#     }

#     dtc = tree.DecisionTreeClassifier()

#     clf = GridSearchCV(dtc, parameters)
#     clf.fit(x_train, y_train)

#     print (compare(list(clf.predict(x_test)), list(y_test)))



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


params = {
    'penalty':['l1', 'l2', 'elasticnet'],        # l1 is Lasso, l2 is Ridge
    'solver':['liblinear']
    # 'C': np.linspace(0.00002,1,100)
}

lr = linear_model.LogisticRegression()
lr_gs = GridSearchCV(lr, params).fit(x_train, y_train)

print (compare(list(lr_gs.predict(x_test)), list(y_test)))
# parameters = {
#     'penalty' : ('l1', 'l2', 'elasticnet', 'none'),
#     'dual' : (True, False), 
#     'C' : [0.5, 2.0], 
#     'fit_intercept' : (True, False)
# }



# lr = linear_model.LogisticRegression(random_state=0, max_iter=30000)

# clf = GridSearchCV(lr, parameters)
# clf.fit(x_train, y_train)

# print (compare(list(clf.predict(x_test)), list(y_test)))


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.7, random_state = 0)
# clf = linear_model.LogisticRegression(random_state=0, max_iter=30000).fit(x_train, y_train)

# print(compare(list(clf.predict(x_test)), list(y_test)))
