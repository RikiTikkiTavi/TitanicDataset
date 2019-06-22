from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.pipeline import Pipeline


class Learner:
    def __init__(self, processed_data: __doc__):
        self.data = processed_data
        self.Y = self.data['Survived']
        self.X = self.data.drop('Survived', axis=1)
        self.grid_search: GridSearchCV = None

    def selection(self) -> Learner:
        pipe = Pipeline([('classifier', DummyClassifier())])
        search_space = [
            {
                'classifier': [svm.LinearSVC(max_iter=10000, dual=False)],
                'classifier__C': range(1, 10, 2),
                'classifier__tol': [0.00001, 0.0001, 0.001]
            },
            {
                'classifier': [RandomForestClassifier()],
                'classifier__n_estimators': [80, 100, 120],
                'classifier__max_features': [3, 5, None]
            }
        ]
        self.grid_search = GridSearchCV(pipe, search_space, cv=5, verbose=0)
        self.grid_search.fit(self.X, self.Y)
        return self

    def check_svm(self):
        parameters = {'C': [1, 5, 10, 15], 'tol': [0.00001, 0.0001, 0.001]}
        svc = svm.LinearSVC(max_iter=5000)
        clf = GridSearchCV(estimator=svc,
                           param_grid=parameters,
                           cv=5,
                           n_jobs=-1)
        clf.fit(self.X, self.Y)
        print(clf.best_score_)
        print(clf.best_estimator_)
        return self

    def pick_best(self):
        print(self.grid_search.best_score_)
        print(self.grid_search.best_estimator_)
        return self.grid_search.best_estimator_
