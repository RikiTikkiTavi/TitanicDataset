from __future__ import annotations

from joblib import dump, load

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.pipeline import Pipeline


class Learner:
    def __init__(self, processed_data: __doc__):
        self.data = processed_data
        self.Y = self.data['Survived']
        self.X = self.data.drop('Survived', axis=1)
        self.grid_search: GridSearchCV = None

    @staticmethod
    def __build_svc_space() -> dict:
        return {
            'classifier': [svm.LinearSVC(max_iter=10000, dual=False)],
            'classifier__C': range(1, 10, 2),
            'classifier__tol': [0.00001, 0.0001, 0.001]
        }

    @staticmethod
    def __build_rf_space() -> dict:
        return {
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [80, 100, 120],
            'classifier__max_features': [3, 5, None]
        }

    @staticmethod
    def __build_xgb_space() -> dict:
        return {
            'classifier': [GradientBoostingClassifier()],
            'classifier__n_estimators': [100, 120, 140],
            'classifier__learning_rate': [0.05, 0.1, 0.15, 0.2]
        }

    def selection(self) -> Learner:
        pipe = Pipeline([('classifier', DummyClassifier())])
        search_space = [
            Learner.__build_svc_space(),
            Learner.__build_rf_space(),
            Learner.__build_xgb_space()
        ]
        self.grid_search = GridSearchCV(pipe, search_space, cv=5, verbose=0)
        self.grid_search.fit(self.X, self.Y)
        return self

    def pick_best(self):
        print(self.grid_search.best_score_)
        print(self.grid_search.best_estimator_)
        return self.grid_search.best_estimator_

    def save_best(self, path):
        dump(self.grid_search.best_estimator_, path)
        return self
