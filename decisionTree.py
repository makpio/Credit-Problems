from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def decisionTree(x_train, y_train, x_test):
    
    parameters = {
        'splitter' : ('best', 'random'), 
        'criterion' : ('gini', 'entropy'),
        'max_depth' : [1, None],
        'min_samples_split' : [2, 5],
        'min_samples_leaf' : [1, 5],
        'min_weight_fraction_leaf' : [0.0, 0.5],
        'max_features' : [1, None],
        'ccp_alpha' : [0.0, 0.5]
    }
    dt = DecisionTreeClassifier()
    classifier = GridSearchCV(dt, parameters)
    classifier.fit(x_train, y_train)

    return list(classifier.predict(x_test))






