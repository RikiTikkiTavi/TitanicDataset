from __future__ import annotations

import pandas as pd
from src.constants import src_train
import matplotlib.pyplot as plt


class Preparer:
    def __init__(self):
        self.data: __doc__ = pd.read_csv(src_train, index_col="PassengerId")

    def show_insights(self) -> Preparer:
        print("----------- Info:")
        print(self.data.info())
        print("----------- Insights:")
        print(self.data.describe())
        print("----------- Missing values:")
        print(len(self.data) - self.data.count())
        print("----------- Some Non numeric examples")
        self.data.groupby(['Survived', 'Sex']).size().unstack().plot(kind='bar', stacked=True)
        self.data.groupby(['Embarked', 'Pclass']).size().unstack().plot(kind='bar', stacked=True)
        plt.show()
        return self

    def remove_redundant_cols(self):
        self.data.drop(["Name", "Cabin"], axis="columns")
        return self

    def handle_nan(self):
        self.data['Age'].fillna(self.data['Age'].mean())
        return self

    @staticmethod
    def gender_encoding(gender: str):
        return gender == "male"

    @staticmethod
    def encode_categorical(series: pd.Series):
        series["Sex"] = Preparer.gender_encoding(series["Sex"])

    def handle_categorical(self):
        self.data.apply(Preparer.encode_categorical, axis=1)
        return self

    def build(self):
        return self.data


data_prepared = Preparer() \
    .show_insights() \
    .remove_redundant_cols() \
    .handle_nan() \
    .handle_categorical() \
    .build()
