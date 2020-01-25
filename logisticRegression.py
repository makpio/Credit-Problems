from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression 

def logisticRegression(x_train, y_train, x_test):

    parameters = {
        'penalty' : ('l1', 'l2'),
        'C' : [0.5, 1.5], 
        'fit_intercept' : (True, False),
        'class_weight' : ('balanced', None),
        'n_jobs' : [1, 4],
        'warm_start' : (True, False),
        'random_state' : [0, 5]
    }
    lr = LogisticRegression(solver='saga', tol=1e-2, multi_class='ovr')
    classifier = GridSearchCV(lr, parameters)
    classifier.fit(x_train, y_train)

    return list(classifier.predict(x_test))