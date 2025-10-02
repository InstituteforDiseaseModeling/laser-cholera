import matplotlib.pyplot as plt
import numpy as np


class DerivedValues:
    def __init__(self, model) -> None:
        self.model = model

        assert hasattr(model, "patches"), "DerivedValues: model needs to have a 'patches' attribute."
        assert hasattr(model, "params"), "DerivedValues: model needs to have a 'params' attribute."

        # There's no spatial hazard calculation at the start of the simulation, but we will allocate nticks + 1 to match other outputs.
        model.patches.add_vector_property("spatial_hazard", length=model.params.nticks + 1, dtype=np.float32, default=0.0)
        model.patches.add_array_property("coupling", shape=(model.patches.count, model.patches.count), dtype=np.float32, default=0.0)

        return

    def check(self):
        assert hasattr(self.model, "people"), "DerivedValues: model needs to have an 'people' attribute."
        assert hasattr(self.model.people, "S"), "DerivedValues: model.people needs to have 'S' attribute."
        assert hasattr(self.model.people, "Isym"), "DerivedValues: model.people needs to have 'Isym' attribute."
        assert hasattr(self.model.people, "Iasym"), "DerivedValues: model.people needs to have 'Iasym' attribute."
        assert hasattr(self.model.people, "V1sus"), "DerivedValues: model.people needs to have 'V1sus' attribute."
        assert hasattr(self.model.people, "V2sus"), "DerivedValues: model.people needs to have 'V2sus' attribute."

        assert hasattr(self.model, "patches"), "DerivedValues: model needs to have a 'patches' attribute."
        assert hasattr(self.model.patches, "N"), "DerivedValues: model.patches needs to have 'N' attribute."
        assert hasattr(self.model.patches, "beta_jt_human"), "DerivedValues: model.patches needs to have 'beta_jt_human' attribute."
        assert hasattr(self.model.patches, "pi_ij"), "DerivedValues: model.patches needs to have 'pi_ij' attribute."

        assert "beta_j0_hum" in self.model.params, "DerivedValues: model.params needs to have 'beta_j0_hum' attribute."
        assert "p" in self.model.params, "DerivedValues: model.params needs to have 'p' attribute."
        assert "tau_i" in self.model.params, "DerivedValues: model.params needs to have 'tau_i' attribute."

        return

    def __call__(self, model, tick: int) -> None:
        """Calculate derived values for the model.

        Spatial hazard and coupling are calculated at the end of the simulation.

        .. math::

            h(j,t) = \\frac {\\beta^{hum}_{jt} (1 - e^{-((1 - \\tau_j) (S_{jt} + V^{sus}_{1,jt} + V^{sus}_{2,jt}) / N_{jt}) \\sum_{\\forall i \\ne j} \\pi_{ij} \\tau_i ((I^{sym}_{it} + I^{asym}_{it}) / N_{it})})} {1/(1 + \\beta^{hum}_{jt}(1 - \\tau_j) (S_{jt} + V^{sus}_{1,jt} + V^{sus}_{2,jt}))}

        .. math::

            y_{it} = \\frac {I^{sym}_{it} + I^{asym}_{it}} {N_{it}}

        .. math::

            \\bar y_{i} = \\frac 1 T \\sum_{t=1}^{T} y_{it}

        .. math::

            C_{ij} = \\frac { \\sum_{t=1}^T {(y_{it} - \\bar y_i) (y_{jt} - \\bar y_j)} } { \\sqrt {\\sum_{t=1}^T {(y_{it} - \\bar y_i)}^2} \\sqrt {\\sum_{t=1}^T {(y_{jt} - \\bar y_j)}^2} }


        .. math::

            C_{ij} = \\frac {(y_{it} - \\bar y_{i}) (y_{jt} - \\bar y_{j})} { \\sqrt {\\text {var}(y_{i}) \\text {var}(y_{j})} }

        """
        if tick == model.params.nticks - 1:
            calculate_spatial_hazard(
                model.params.nticks,
                model.patches.beta_jt_human.T,
                model.params.tau_i,
                model.people.S[1:, :].T,
                model.people.V1sus[1:, :].T,
                model.people.V2sus[1:, :].T,
                model.patches.N[1:, :].T,
                model.patches.pi_ij,
                model.people.Iasym[1:, :].T,
                model.people.Isym[1:, :].T,
                model.patches.spatial_hazard[1:, :].T,
            )

            calculate_coupling(model.people.Isym, model.people.Iasym, model.patches.N, model.patches.coupling)

        return

    def plot(self, fig=None):  # pragma: no cover
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
        model.people.V1sus[1:, :].T,
        model.people.V2sus[1:, :].T,
        model.patches.N[1:, :].T,
        model.patches.pi_ij,
        model.people.Iasym[1:, :].T,
        model.people.Isym[1:, :].T,
        model.patches.spatial_hazard[1:, :].T,
    )

    return


def calculate_spatial_hazard(nticks, beta_jt_human, tau_i, S, V1sus, V2sus, Njt, pi_ij, Iasym, Isym, spatial_hazard):
    """Calculate the spatial hazard for each location at each time step.
    The spatial hazard is calculated using the formula:

    https://institutefordiseasemodeling.github.io/MOSAIC-docs/model-description.html#the-spatial-hazard

    .. math::

        h(j,t) = \\frac {\\beta^{hum}_{jt} (1 - e^{-((1 - \\tau_j) (S_{jt} + V^{sus}_{1,jt} + V^{sus}_{2,jt}) / N_{jt}) \\sum_{\\forall i \\ne j} \\pi_{ij} \\tau_i ((I^{sym}_{it} + I^{asym}_{it}) / N_{it})})} {1/(1 + \\beta^{hum}_{jt}(1 - \\tau_j) (S_{jt} + V^{sus}_{1,jt} + V^{sus}_{2,jt}))}

    Note: To simplify coding and debugging, all arrays are passed in R order: [j, t]

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
    #         S_star_jt = (1.0 - tau_i[j]) * (S[j, t] + V1sus[j, t] + V2sus[j, t])
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
    S_star_jt = ((1.0 - tau_i) * (S + V1sus + V2sus).T).T
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
    """Calculate the coupling between locations.

    https://institutefordiseasemodeling.github.io/MOSAIC-docs/model-description.html#coupling-among-locations

    .. math::

        C_{ij} = \\frac {(y_{it} - \\bar y_{i}) (y_{jt} - \\bar y_{j})} { \\sqrt {\\text {var}(y_{i}) \\text {var}(y_{j})} }

    """
    assert Isym.shape == Iasym.shape, "Isym and Iasym must have the same shape."
    assert Isym.shape == N.shape, "Isym and N must have the same shape."
    _T, L = N.shape
    assert C.shape == (L, L), "C must be a square matrix of shape (L, L)."

    y = (Isym + Iasym) / N

    y_bar = np.mean(y, axis=0)  # mean over time (axis=0) for each location

    diff = y - y_bar  # difference from mean

    for i in range(L):
        for j in range(i, L):
            numerator = np.sum(diff[:, i] * diff[:, j])
            denominator = np.sqrt(np.sum(diff[:, i] ** 2) * np.sum(diff[:, j] ** 2))
            if denominator != 0:
                C_ij = numerator / denominator
            else:
                C_ij = np.nan
            C[i, j] = C[j, i] = C_ij

    return
