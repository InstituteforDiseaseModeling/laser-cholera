import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import beta


class Environmental:
    def __init__(self, model) -> None:
        self.model = model

        assert hasattr(model, "patches"), "Environmental: model needs to have a 'patches' attribute."

        model.patches.add_vector_property("W", length=model.params.nticks + 1, dtype=np.float32, default=0.0)
        assert hasattr(model, "params"), "Environmental: model needs to have a 'params' attribute."
        assert hasattr(model.params, "psi_jt"), (
            "Environmental: model params needs to have a 'psi_jt' (environmental contamination rate) parameter."
        )
        psi = model.params.psi_jt  # convenience
        # TODO - use newer laser_core with add_array_property and psi.shape
        model.patches.add_vector_property("delta_jt", length=psi.shape[0], dtype=np.float32, default=0.0)

        assert "decay_days_short" in model.params, (
            "Environmental: model params needs to have a 'decay_days_short' (maximum environmental decay) parameter."
        )
        assert "decay_days_long" in model.params, (
            "Environmental: model params needs to have a 'decay_days_long' (minimum environmental decay) parameter."
        )
        assert "decay_shape_1" in self.model.params, (
            "Environmental: model params needs to have a 'decay_shape_1' (beta function parameter 1) parameter."
        )
        assert "decay_shape_2" in self.model.params, (
            "Environmental: model params needs to have a 'decay_shape_2' (beta function parameter 2) parameter."
        )

        model.patches.delta_jt[:, :] = map_suitability_to_decay(
            fast=model.params.decay_days_short,
            slow=model.params.decay_days_long,
            suitability=model.params.psi_jt,
            beta_a=model.params.decay_shape_1,
            beta_b=model.params.decay_shape_2,
        )

        return

    def check(self):
        assert hasattr(self.model, "people"), "Environmental: model needs to have a 'people' attribute."
        assert hasattr(self.model.people, "Isym"), "Environmental: model people needs to have a 'Isym' (symptomatic) attribute."
        assert hasattr(self.model.people, "Iasym"), "Environmental: model people needs to have a 'Iasym' (asymptomatic) attribute."
        assert "zeta_1" in self.model.params, "Environmental: model params needs to have a 'zeta_1' (symptomatic shedding rate) parameter."
        assert "zeta_2" in self.model.params, "Environmental: model params needs to have a 'zeta_2' (asymptomatic shedding rate) parameter."
        assert "theta_j" in self.model.params, (
            "Environmental: model params needs to have a 'theta_j' (fraction of population with WASH) attribute."
        )

        return

    def __call__(self, model, tick: int) -> None:
        W = model.patches.W[tick]
        W_next = model.patches.W[tick + 1]
        W_next[:] = W

        Isym = model.people.Isym[tick]
        Iasym = model.people.Iasym[tick]

        # -decay
        # Use np.minimum() to make sure we don't go negative
        decay = np.minimum(model.prng.poisson(model.patches.delta_jt[tick] * W), W).astype(W_next.dtype)
        W_next -= decay

        # +shedding from Isymptomatic
        shedding_sym = model.prng.poisson(model.params.zeta_1 * Isym).astype(W_next.dtype)
        W_next += (1 - model.params.theta_j) * shedding_sym

        # +shedding from Iasymptomatic
        shedding_asym = model.prng.poisson(model.params.zeta_2 * Iasym).astype(W_next.dtype)
        W_next += (1 - model.params.theta_j) * shedding_asym

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Environmental Reservoir") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.patches.W[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Environmental Reservoir")
        plt.legend()

        yield "Environmental Reservoir"

        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Environmental Decay Rate") if fig is None else fig

        plt.imshow(self.model.patches.delta_jt.T, aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Environmental Decay Rate")
        plt.xlabel("Tick")
        plt.ylabel("Patch")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "Environmental Decay Rate"

        title = "Suitability to Decay Mapping"
        _fig = plt.figure(figsize=(12, 9), dpi=128, num=title) if fig is None else fig

        x = np.linspace(0, 1, self.model.params.psi_jt.shape[0])
        y = map_suitability_to_decay(
            self.model.params.decay_days_short,
            self.model.params.decay_days_long,
            x,
            self.model.params.decay_shape_1,
            self.model.params.decay_shape_2,
        )
        plt.plot(x, y, label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("psi_jt - suitability")
        plt.ylabel("delta_jt - decay rate")

        yield title
        return


# Put this is its own function so mapping plot is sure to use the same calculation.
def map_suitability_to_decay(fast, slow, suitability, beta_a, beta_b):
    """
    Map suitability to decay using a beta distribution.

    $ \\delta_{jt} = \frac { 1 } { \text {days}_short + f( \\psi_{jt}) ( \text {days}_long  - \text {days}_short ) } $

    We use a parameterized beta distribution to map suitability values [0, 1] to [0, 1] in a, potentially, non-linear way.

    The resulting suitability factor determines a decay rate that is large when suitability is low,
    i.e., when suitability is 0, the decay rate is 1/fast and since fast is a short time or small number of days, 1/fast is relatively large,
    and small when suitability is high,
    i.e. when suitability is 1, the decay rate is 1/slow and since slow is a long time or larger number of days, 1/slow is relatively small.

    Parameters
    ----------
    fast : float
        Fast decay time.
    slow : float
        Slow decay time.
    suitability : np.ndarray
        Suitability values.
    beta_a : float
        Alpha parameter for the beta distribution.
    beta_b : float
        Beta parameter for the beta distribution.

    Returns
    -------
    np.ndarray
        Decay rates corresponding to the suitability values.
    """
    return 1.0 / (fast + beta.cdf(suitability, beta_a, beta_b) * (slow - fast))
