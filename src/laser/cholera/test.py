from matplotlib.figure import Figure


class Eradication:
    def __init__(self, model) -> None:
        self.model = model

        return

    def check(self):
        return

    def __call__(self, model, tick: int) -> None:
        if tick == 1:
            model.people.S[tick] += model.people.Isym[tick] + model.people.Iasym[tick]
            model.people.Isym[tick] = 0
            model.people.Iasym[tick] = 0

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        yield
        return
