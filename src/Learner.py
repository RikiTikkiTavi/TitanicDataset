from __future__ import annotations
from sklearn.model_selection import GridSearchCV
from sklearn import svm


class Learner:
    def __init__(self, processed_data: __doc__):
        self.data = processed_data
        self.Y = self.data['Survived']
        self.X = self.data.drop('Survived', axis=1)

    def check_svm(self):
        parameters = {'C': [1, 5, 10, 15], 'tol': [0.00001, 0.0001, 0.001]}
        svc = svm.LinearSVC(max_iter=5000)
        clf = GridSearchCV(estimator=svc,
                           param_grid=parameters,
                           cv=5)
        clf.fit(self.X, self.Y)
        print(clf.best_params_)
        print(clf.best_score_)
        print(clf.best_estimator_)
