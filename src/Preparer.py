from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.constants import src_train
import matplotlib.pyplot as plt


# noinspection PyMethodMayBeStatic
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
        print("----------- Some of Tickets:")
        print(self.data["Ticket"].head())
        self.data.groupby(['Survived', 'Sex']).size().unstack().plot(kind='bar', stacked=True)
        self.data.groupby(['Embarked', 'Pclass']).size().unstack().plot(kind='bar', stacked=True)
        self.data.plot(kind='scatter', x='Survived', y='Fare')
        plt.show()
        return self

    def remove_redundant_cols(self):
        self.data.drop(["Name", "Ticket", "Cabin"], axis="columns", inplace=True)
        return self

    def handle_nan(self):
        self.data['Age'].fillna(self.data['Age'].mean(), inplace=True)
        self.data['Embarked'].fillna(self.data['Embarked'].mode()[0], inplace=True)
        return self

    def handle_categorical(self):
        self.data[["Sex", "Embarked"]] = self.data[["Sex", "Embarked"]] \
            .transform(lambda col: LabelEncoder().fit_transform(col))
        return self

    def handle_tickets(self):
        def get_ticket_number(ticket: str) -> float:
            return float(ticket.split()[-1])

        self.data["Ticket"] = self.data["Ticket"].apply(get_ticket_number)
        print(self.data["Ticket"].head())
        return self

    def handle_cabin(self):
        pass

    def build(self):
        print(self.data.columns)
        return self.data
