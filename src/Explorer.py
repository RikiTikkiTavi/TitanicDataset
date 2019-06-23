class Explorer:
    def __init__(self, data):
        self.data = data;

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
