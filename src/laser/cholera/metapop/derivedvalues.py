"""End-of-run derived diagnostics: spatial hazard and inter-location coupling.

[`DerivedValues`][laser.cholera.metapop.derivedvalues.DerivedValues]
runs every tick but only does its real work on the final tick, computing:

- `patches.spatial_hazard` — per-patch, per-tick infection-pressure
  metric incorporating local susceptibility, emigration / immigration
  via `pi_ij`, and the seasonality envelope `beta_jt_human`.
- `patches.coupling` — `(npatches × npatches)` Pearson correlation
  matrix of the per-tick prevalence-fraction series across patch pairs.

Module-level helpers
[`calculate_spatial_hazard`][laser.cholera.metapop.derivedvalues.calculate_spatial_hazard]
and
[`calculate_coupling`][laser.cholera.metapop.derivedvalues.calculate_coupling]
do the math; the per-model wrappers exist primarily so R callers can
invoke them with a single argument.
"""

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from laser.cholera.metapop.model import Model
from laser.cholera.metapop.utils import check_attr
from laser.cholera.metapop.utils import check_key

logger = logging.getLogger("laser.cholera")


class DerivedValues:
    """Computes end-of-run diagnostics: spatial hazard and coupling matrix.

    Attributes:
        model: The parent `Model` instance.
    """

    def __init__(self, model: "Model") -> None:
        """Allocate `spatial_hazard` and `coupling` arrays on `model.patches`.

        Args:
            model: The `Model` instance. Must have `patches` and
                `params` set up.

        Raises:
            AttributeError: When `model.patches` or `model.params` is
                missing.
        """
        self.model = model

        check_attr(model, "patches", "DerivedValues: model needs to have a 'patches' attribute.")
        check_attr(model, "params", "DerivedValues: model needs to have a 'params' attribute.")

        # There's no spatial hazard calculation at the start of the simulation, but we will allocate nticks + 1 to match other outputs.
        model.patches.add_vector_property("spatial_hazard", length=model.params.nticks + 1, dtype=np.float32, default=0.0)
        model.patches.add_array_property("coupling", shape=(model.patches.count, model.patches.count), dtype=np.float32, default=0.0)

        return

    def check(self):
        """Validate every upstream attribute and parameter the final-tick math needs.

        Verifies `model.people` has `S`/`Isym`/`Iasym`; `model.patches`
        has `N`/`beta_jt_human`/`pi_ij`; and `params` provides
        `beta_j0_hum`, `p`, and `tau_i`. These are populated by
        [`Susceptible`][laser.cholera.metapop.susceptible.Susceptible],
        [`Infectious`][laser.cholera.metapop.infectious.Infectious],
        [`Census`][laser.cholera.metapop.census.Census], and
        [`HumanToHuman`][laser.cholera.metapop.humantohuman.HumanToHuman]
        earlier in the pipeline.

        Raises:
            AttributeError: When `model.people` (with `S` / `Isym` /
                `Iasym`) or `model.patches` (with `N` / `beta_jt_human` /
                `pi_ij`) is missing.
            ValueError: When `params.beta_j0_hum`, `params.p`, or
                `params.tau_i` is missing.
        """
        check_attr(self.model, "people", "DerivedValues: model needs to have an 'people' attribute.")
        check_attr(self.model.people, "S", "DerivedValues: model.people needs to have 'S' attribute.")
        check_attr(self.model.people, "Isym", "DerivedValues: model.people needs to have 'Isym' attribute.")
        check_attr(self.model.people, "Iasym", "DerivedValues: model.people needs to have 'Iasym' attribute.")

        check_attr(self.model, "patches", "DerivedValues: model needs to have a 'patches' attribute.")
        check_attr(self.model.patches, "N", "DerivedValues: model.patches needs to have 'N' attribute.")
        check_attr(self.model.patches, "beta_jt_human", "DerivedValues: model.patches needs to have 'beta_jt_human' attribute.")
        check_attr(self.model.patches, "pi_ij", "DerivedValues: model.patches needs to have 'pi_ij' attribute.")

        check_key(self.model.params, "beta_j0_hum", "DerivedValues: model.params needs to have 'beta_j0_hum' attribute.")
        check_key(self.model.params, "p", "DerivedValues: model.params needs to have 'p' attribute.")
        check_key(self.model.params, "tau_i", "DerivedValues: model.params needs to have 'tau_i' attribute.")

        return

    def __call__(self, model: "Model", tick: int) -> None:
        r"""Calculate derived values for the model.

        Spatial hazard and coupling are calculated at the end of the
        simulation.

        Spatial hazard per location and tick:

        $$
        h(j,t) = \frac {\beta^{hum}_{jt} (1 - e^{-((1 - \tau_j) (S_{jt} / N_{jt})) \sum_{\forall i \ne j} \pi_{ij} \tau_i ((I^{sym}_{it} + I^{asym}_{it}) / N_{it})})} {1/(1 + \beta^{hum}_{jt}(1 - \tau_j) S_{jt})}
        $$

        Prevalence fraction and its time-average per location:

        $$
        y_{it} = \frac {I^{sym}_{it} + I^{asym}_{it}} {N_{it}}
        $$

        $$
        \bar y_{i} = \frac 1 T \sum_{t=1}^{T} y_{it}
        $$

        Inter-location coupling — Pearson correlation between prevalence
        fractions:

        $$
        C_{ij} = \frac { \sum_{t=1}^T {(y_{it} - \bar y_i) (y_{jt} - \bar y_j)} } { \sqrt {\sum_{t=1}^T {(y_{it} - \bar y_i)}^2} \sqrt {\sum_{t=1}^T {(y_{jt} - \bar y_j)}^2} }
        $$

        Equivalently:

        $$
        C_{ij} = \frac {(y_{it} - \bar y_{i}) (y_{jt} - \bar y_{j})} { \sqrt {\text{var}(y_{i}) \text{var}(y_{j})} }
        $$

        Args:
            model: The parent `Model` instance.
            tick: Current simulation tick. The computation is a no-op
                for every tick except the last
                (`model.params.nticks - 1`).
        """
        if tick == model.params.nticks - 1:
            calculate_spatial_hazard(
                model.params.nticks,
                model.patches.beta_jt_human.T,
                model.params.tau_i,
                model.people.S[1:, :].T,
                model.patches.N[1:, :].T,
                model.patches.pi_ij,
                model.people.Iasym[1:, :].T,
                model.people.Isym[1:, :].T,
                model.patches.spatial_hazard[1:, :].T,
            )

            calculate_coupling(model.people.Isym, model.people.Iasym, model.patches.N, model.patches.coupling)

        return

    def plot(self, fig: Optional[Figure] = None) -> Iterator[str]:  # pragma: no cover
        """Yield one Matplotlib heatmap of `spatial_hazard` (patch × tick).

        Args:
            fig: Optional existing Matplotlib `Figure` to draw into.

        Yields:
            The string label `"Spatial Hazard by Location Over Time"`.
        """
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Spatial Hazard by Location Over Time") if fig is None else fig

        plt.imshow(self.model.patches.spatial_hazard.T, aspect="auto", cmap="Reds", interpolation="nearest")
        plt.colorbar(label="Spatial Hazard")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "Spatial Hazard by Location Over Time"

        return


def calculate_spatial_hazard_for_model(model):
    """Calculate the spatial hazard for the model.

    Helpful for calling from R without needing to specify all the parameters.
    """
    calculate_spatial_hazard(
        model.params.nticks,
        model.patches.beta_jt_human.T,
        model.params.tau_i,
        model.people.S[1:, :].T,
        model.patches.N[1:, :].T,
        model.patches.pi_ij,
        model.people.Iasym[1:, :].T,
        model.people.Isym[1:, :].T,
        model.patches.spatial_hazard[1:, :].T,
    )

    return


def calculate_spatial_hazard(nticks, beta_jt_human, tau_i, S, Njt, pi_ij, Iasym, Isym, spatial_hazard):
    r"""Calculate the spatial hazard for each location at each time step.

    The spatial hazard is calculated using the formula
    ([reference][hazard-ref]):

    $$
    h(j,t) = \frac {\beta^{hum}_{jt} (1 - e^{-((1 - \tau_j) (S_{jt} / N_{jt})) \sum_{\forall i \ne j} \pi_{ij} \tau_i ((I^{sym}_{it} + I^{asym}_{it}) / N_{it})})} {1/(1 + \beta^{hum}_{jt}(1 - \tau_j) S_{jt})}
    $$

    Note: To simplify coding and debugging, all arrays are passed in R
    order: `[j, t]`.

    [hazard-ref]: https://institutefordiseasemodeling.github.io/MOSAIC-docs/model-description.html#the-spatial-hazard
    """

    beta = beta_jt_human

    # Reference implementation:

    # values for comparison/debugging
    # S_star = np.zeros_like(S, dtype=np.float64)
    # x = np.zeros_like(S, dtype=np.float64)
    # y_bar = np.zeros_like(S, dtype=np.float64)

    # nlocs = Njt.shape[0]
    # for t in range(nticks):
    #     Nkt = Njt[:, t].sum()
    #     for j in range(nlocs):
    #         S_star_jt = (1.0 - tau_i[j]) * (S[j, t])
    #         # S_star[j, t] = S_star_jt
    #         x_jt = S_star_jt / Njt[j, t]
    #         # x[j, t] = x_jt
    #         sum_tau_pi_i = 0.0
    #         for i in range(nlocs):
    #             if i != j:
    #                 sum_tau_pi_i += tau_i[i] * pi_ij[i, j] * (Isym[i, t] + Iasym[i, t])
    #         y_bar_jt = ((1.0 - tau_i[j]) * (Isym[j, t] + Iasym[j, t]) + sum_tau_pi_i) / Nkt
    #         # y_bar[j, t] = y_bar_jt
    #         H_jt = beta[j, t] * S_star_jt * (1.0 - np.exp(-x_jt * y_bar_jt)) / (1.0 + beta[j, t] * S_star_jt)
    #         spatial_hazard[j, t] = H_jt

    # Opimized implementation:

    Nt = Njt.sum(axis=0)  # Total population at time t
    xfer_ij = (tau_i * pi_ij.T).T  # fraction emmigrating from i * fraction going to j = fraction of i going to j
    S_star_jt = ((1.0 - tau_i) * S.T).T
    x_jt = S_star_jt / Njt  # S_star / N for each location and time step
    beta_S_star_jt = beta * S_star_jt  # beta_jt_human * S_star for each location and time step

    Ijt = Isym + Iasym
    # (Remaining) infections in location j at time t accounting for the fraction of emmigrants by location
    Iloc_jt = ((1.0 - tau_i) * Ijt.T).T

    for t in range(nticks):
        I_local = Iloc_jt[:, t]  # Local infections at time t, syntactic sugar
        # Calculate the incoming infections at time t
        I_incoming = np.dot(Ijt[:, t], xfer_ij)  # @, matmul, raises RuntimeWarnings on MacOS, but works fine on Linux
        y_bar_t = (I_local + I_incoming) / Nt[t]
        H_t = beta_S_star_jt[:, t] * (-np.expm1(-x_jt[:, t] * y_bar_t)) / (1.0 + beta_S_star_jt[:, t])
        spatial_hazard[:, t] = H_t

    return


def calculate_coupling_for_model(model):
    """Calculate the coupling for the model.

    Helpful for calling from R without needing to specify all the parameters.
    """
    calculate_coupling(model.people.Isym, model.people.Iasym, model.patches.N, model.patches.coupling)

    return


def calculate_coupling(Isym, Iasym, N, C):
    r"""Calculate the coupling between locations.

    Pearson correlation between per-tick prevalence fractions, computed
    as ([reference][coupling-ref]):

    $$
    C_{ij} = \frac {(y_{it} - \bar y_{i}) (y_{jt} - \bar y_{j})} { \sqrt {\text{var}(y_{i}) \text{var}(y_{j})} }
    $$

    [coupling-ref]: https://institutefordiseasemodeling.github.io/MOSAIC-docs/model-description.html#coupling-among-locations
    """
    assert Isym.shape == Iasym.shape, "Isym and Iasym must have the same shape."
    assert Isym.shape == N.shape, "Isym and N must have the same shape."
    _T, L = N.shape
    assert C.shape == (L, L), "C must be a square matrix of shape (L, L)."

    y = (Isym + Iasym) / N

    # PERF: the manual `for i / for j in range(i, L)` double loop computed
    # the Pearson correlation between every pair of patch columns of `y`.
    # `np.corrcoef(y, rowvar=False)` is the canonical NumPy equivalent and
    # delegates the cov-and-divide to BLAS-backed reductions. Constant
    # columns (zero variance) produce NaN entries — same as the original
    # `denominator == 0` branch.
    #
    # Note: results may differ from the original by a few ULPs in
    # individual cells because the BLAS reduction order is not the same
    # as `np.sum`'s pairwise reduction; `coupling` is a final-tick
    # diagnostic that nothing else in the simulation reads, so the drift
    # is fully isolated.
    #
    # y_bar = np.mean(y, axis=0)  # mean over time (axis=0) for each location
    # diff = y - y_bar  # difference from mean
    # for i in range(L):
    #     for j in range(i, L):
    #         numerator = np.sum(diff[:, i] * diff[:, j])
    #         denominator = np.sqrt(np.sum(diff[:, i] ** 2) * np.sum(diff[:, j] ** 2))
    #         if denominator != 0:
    #             C_ij = numerator / denominator
    #         else:
    #             C_ij = np.nan
    #         C[i, j] = C[j, i] = C_ij
    #
    # `np.corrcoef` produces NaN for any column with zero variance (a patch
    # whose prevalence was constant for the whole run — typically because
    # `I_j_initial = 0` and no force-of-infection reached it). That's a
    # legitimate model state, not an error: Pearson correlation with a
    # constant series is mathematically undefined (0 / 0), and the original
    # nested-loop branch above returned NaN explicitly. Naively calling
    # `np.corrcoef(y, rowvar=False)` works but emits a `RuntimeWarning` for
    # the underlying divide. Detect the constant columns up front and run
    # corrcoef only on the variable subset so the warning never fires;
    # constant rows/cols are filled with NaN explicitly (and logged at INFO
    # so the situation is visible in run logs).
    constant_cols = y.var(axis=0) == 0
    if constant_cols.any():
        n_const = int(constant_cols.sum())
        logger.info(
            f"calculate_coupling: {n_const} of {L} patches have constant prevalence over the simulation; "
            f"their rows and columns in `coupling` are NaN (correlation with a constant series is undefined)."
        )
        keep = ~constant_cols
        C[:] = np.nan
        if keep.any():
            C[np.ix_(keep, keep)] = np.corrcoef(y[:, keep], rowvar=False)
    else:
        C[:] = np.corrcoef(y, rowvar=False)

    return
