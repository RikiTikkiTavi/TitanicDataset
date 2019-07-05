from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from src.core.Explorer import Explorer
from src.constants import src_train


# noinspection PyMethodMayBeStatic
class Preparer:
    def __init__(self):
        self.data: __doc__ = pd.read_csv(src_train, index_col="PassengerId")

    def show_insights(self) -> Preparer:
        Explorer(self.data) \
            .basic_info() \
            .plotting()
        return self

    def remove_redundant_cols(self) -> Preparer:
        self.data.drop(["Ticket", "Cabin"], axis="columns", inplace=True)
        return self

    def handle_nan(self) -> Preparer:
        self.data['Age'].fillna(self.data['Age'].mean(), inplace=True)
        self.data['Embarked'].fillna(self.data['Embarked'].mode()[0], inplace=True)
        return self

    def handle_categorical(self) -> Preparer:
        self.data[["Name", "Sex", "Embarked"]] = self.data[["Name", "Sex", "Embarked"]] \
            .transform(lambda col: LabelEncoder().fit_transform(col))
        return self

    def handle_tickets(self) -> Preparer:
        def get_ticket_number(ticket: str) -> float:
            return float(ticket.split()[-1])

        self.data["Ticket"] = self.data["Ticket"].apply(get_ticket_number)
        print(self.data["Ticket"].head())
        return self

    def handle_name(self) -> Preparer:
        def get_name_prefix(name: str) -> str:
            return name.split(",")[1].split()[0]

        self.data["Name"] = self.data["Name"].apply(get_name_prefix)
        return self

    def handle_siblings(self) -> Preparer:
        self.data["Siblings"] = self.data["Parch"] + self.data["SibSp"]
        self.data.loc[self.data['Siblings'] > 0, 'travelled_alone'] = 0
        self.data.loc[self.data['Siblings'] == 0, 'travelled_alone'] = 1
        self.data.drop(["Parch", "SibSp", "Siblings"], axis="columns", inplace=True)
        return self

    def scale(self) -> Preparer:
        self.data["Name"] = StandardScaler().fit_transform(self.data["Name"])
        return self

    def handle_cabin(self):
        pass

    def build(self):
        print(self.data.columns)
        return self.data
