import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Recovered:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Recovered: model needs to have a 'agents' attribute."
        assert hasattr(model.agents, "R"), "Recovered: model agents needs to have a 'R' (recovered) attribute."
        assert hasattr(model.agents, "I"), "Recovered: model agents needs to have a 'I' (infectious) attribute."
        assert hasattr(model, "params"), "Recovered: model needs to have a 'params' attribute."
        assert hasattr(model.params, "gamma"), "Recovered: model params needs to have a 'gamma' (recovery rate) parameter."
        assert hasattr(model.params, "epsilon"), "Recovered: model params needs to have a 'epsilon' (recovery rate) parameter."
        assert hasattr(model, "patches"), "Recovered: model needs to have a 'patches' attribute."
        assert hasattr(model.patches, "mortrate"), "Recovered: model patches needs to have a 'mortrate' parameter."

        return

    def __call__(self, model, tick: int) -> None:
        R = model.agents.R[tick]
        Rprime = model.agents.R[tick + 1]
        I = model.agents.I[tick]  # noqa: E741
        Iprime = model.agents.I[tick + 1]
        Sprime = model.agents.S[tick + 1]

        Rprime[:] = R
        newly_recovered = model.prng.binomial(I, model.params.gamma).astype(Rprime.dtype)  # rate or probability?
        Iprime -= newly_recovered
        Rprime += newly_recovered
        waned = model.prng.binomial(R, model.params.epsilon).astype(Rprime.dtype)  # rate or probability?
        Sprime += waned
        Rprime -= waned
        # non-disease mortality
        Rprime -= model.prng.binomial(R, model.patches.mortrate).astype(Rprime.dtype)  # rate or probability?

        return

    def plot(self, fig: Figure = None):
        _fig = Figure(figsize=(12, 9), dpi=128) if fig is None else fig
        ipatch = self.model.patches.initpop.argmax()
        plt.title(f"Recovered in Patch {ipatch}")
        plt.plot(self.model.agents.R[:, ipatch], color="purple", label="Recovered")
        plt.xlabel("Tick")

        yield
        return
