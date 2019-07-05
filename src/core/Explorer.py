import pandas_profiling
from src.constants import src_generated
import matplotlib.pyplot as plt


class Explorer:
    def __init__(self, data):
        self.data = data

    def _heading(self, t):
        print(print(f"----------- ${t}:"))
        return self

    def basic_info(self):
        self._heading("Info")
        print(self.data.info())
        self._heading("Insights")
        print(self.data.describe())
        self._heading("Missing values")
        print(len(self.data) - self.data.count())
        return self

    def profiling(self):
        profile = pandas_profiling.ProfileReport(self.data)
        profile.to_file(output_file=src_generated + "Titanic data profiling.html")
        return self

    def plotting(self):
        self.data.groupby(['Survived', 'Sex']).size().unstack().plot(kind='bar', stacked=True)
        self.data.groupby(['Embarked', 'Pclass']).size().unstack().plot(kind='bar', stacked=True)
        self.data.plot(kind='scatter', x='Survived', y='Fare')
        plt.show()
