import matplotlib.pyplot as plt
import numpy as np


class DerivedValues:
    def __init__(self, model) -> None:
        self.model = model

        assert hasattr(model, "patches"), "DerivedValues: model needs to have a 'patches' attribute."
        assert hasattr(model, "params"), "DerivedValues: model needs to have a 'params' attribute."

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
        assert hasattr(self.model.patches, "beta_j_seasonality"), (
            "DerivedValues: model.patches needs to have 'beta_j_seasonality' attribute."
        )
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
                model.params.beta_j0_hum,
                model.patches.beta_j_seasonality,
                model.params.p,
                model.params.tau_i,
                model.people.S,
                model.people.V1sus,
                model.people.V2sus,
                model.patches.N,
                model.patches.pi_ij,
                model.people.Iasym,
                model.people.Isym,
                model.patches.spatial_hazard,
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
        model.params.beta_j0_hum,
        model.patches.beta_j_seasonality,
        model.params.p,
        model.params.tau_i,
        model.people.S,
        model.people.V1sus,
        model.people.V2sus,
        model.patches.N,
        model.patches.pi_ij,
        model.people.Iasym,
        model.people.Isym,
        model.patches.spatial_hazard,
    )

    return


def calculate_spatial_hazard(nticks, beta_j0_hum, beta_j_seasonality, p, tau_i, S, V1sus, V2sus, N, pi_ij, Iasym, Isym, spatial_hazard):
    """Calculate the spatial hazard for each location at each time step.
    The spatial hazard is calculated using the formula:

    https://www.mosaicmod.org/model-description.html#the-spatial-hazard

    .. math::

        h(j,t) = \\frac {\\beta^{hum}_{jt} (1 - e^{-((1 - \\tau_j) (S_{jt} + V^{sus}_{1,jt} + V^{sus}_{2,jt}) / N_{jt}) \\sum_{\\forall i \\ne j} \\pi_{ij} \\tau_i ((I^{sym}_{it} + I^{asym}_{it}) / N_{it})})} {1/(1 + \\beta^{hum}_{jt}(1 - \\tau_j) (S_{jt} + V^{sus}_{1,jt} + V^{sus}_{2,jt}))}

    """

    for t in range(nticks):
        beta_j = beta_j0_hum * (1.0 + beta_j_seasonality[t % p, :])
        tau_j = tau_i
        S_j = S[t] + V1sus[t] + V2sus[t]  # Use S_j where S_j = S + V1sus + V2sus
        N_i = N_j = N[t]
        # pi_ij = pi_ij
        # tau_i = tau_i
        I_i = Isym[t] + Iasym[t]  # Use I_i where I_i = Isym + Iasym

        S_effective = (1 - tau_j) * S_j / N_j  # Use S_effective where S_effective = (1 - tau_j) * S_j / N_j
        I_incoming = (pi_ij * tau_i * I_i / N_i).sum(axis=1)
        rate = S_effective * I_incoming
        probability = -np.expm1(-rate)  # expm1 = exp(x) - 1, ∴ -expm1(-x) = 1 - exp(-x)
        denominator = 1 / (1 + beta_j * (1 - tau_j) * S_j)
        hazard = (beta_j * probability) / denominator
        spatial_hazard[t] = hazard

    return


def calculate_coupling_for_model(model):
    """Calculate the coupling for the model.

    Helpful for calling from R without needing to specify all the parameters.
    """
    calculate_coupling(model.people.Isym, model.people.Iasym, model.patches.N, model.patches.coupling)

    return


def calculate_coupling(Isym, Iasym, N, C):
    """Calculate the coupling between locations.

    https://www.mosaicmod.org/model-description.html#coupling-among-locations

    .. math::

        C_{ij} = \\frac {(y_{it} - \\bar y_{i}) (y_{jt} - \\bar y_{j})} { \\sqrt {\\text {var}(y_{i}) \\text {var}(y_{j})} }

    """
    assert Isym.shape == Iasym.shape, "Isym and Iasym must have the same shape."
    assert Isym.shape == N.shape, "Isym and N must have the same shape."
    T, L = N.shape
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
