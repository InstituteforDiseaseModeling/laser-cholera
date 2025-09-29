import gzip
import io
import json
import logging
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Optional
from typing import Union

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from laser_core.propertyset import PropertySet
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class PseEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, PropertySet):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return f"{obj:%Y-%m-%d}"
        else:
            return super().default(obj)


class PropertySetEx(PropertySet):
    def __init__(self, *kvps):
        super().__init__(*kvps)

        return

    def __str__(self) -> str:
        """Return a string representation of the PropertySet.

        Include converters for datetime and NumPy types.
        """
        return json.dumps(self.to_dict(), cls=PseEncoder, indent=4)


def get_parameters(
    paramsource: Optional[Union[str, Path, dict]] = None, do_validation: bool = True, overrides: Optional[dict] = None
) -> PropertySetEx:
    fn_map = {
        (".json",): load_json_parameters,
        (".json", ".gz"): load_compressed_json_parameters,
        (".h5",): load_hdf5_parameters,
        (".hdf",): load_hdf5_parameters,
        (".hdf5",): load_hdf5_parameters,
        (".h5", ".gz"): load_compressed_hdf5_parameters,
        (".hdf", ".gz"): load_compressed_hdf5_parameters,
        (".hdf5", ".gz"): load_compressed_hdf5_parameters,
    }

    if isinstance(paramsource, (str, Path, type(None))):
        file_path = Path(paramsource) if paramsource is not None else Path(__file__).parent / "data" / "default_parameters.json"
        suffixes = [suffix.lower() for suffix in file_path.suffixes]
        load_fn = fn_map[tuple(suffixes)]

        logger.info(f"Loading parameters from `{file_path}`…")
        params = load_fn(file_path)

    elif isinstance(paramsource, dict):
        params = dict_to_propertysetex(paramsource)

    else:
        raise ValueError(f"Invalid parameter source type: {type(paramsource)}")

    if overrides is not None:
        # Update the parameters with the overrides
        params += overrides

        logger.info("Updated/overrode file parameters with overrides:")
        for k, v in overrides.items():
            logger.info(f"  '{k}': {v}")

    if do_validation:
        validate_parameters(params)

    if "visualize" not in params:
        params.visualize = False
    if "pdf" not in params:
        params.pdf = False
    if "quiet" not in params:
        params.quiet = True

    return params


def load_json_parameters(filename: Union[str, Path]) -> PropertySetEx:
    file_path = Path(filename)
    with file_path.open("r") as file:
        parameters = json.load(file)

    return dict_to_propertysetex(parameters)


def load_compressed_json_parameters(filename: Union[str, Path]) -> PropertySetEx:
    file_path = Path(filename)
    with gzip.open(file_path, "rb") as gz_file:
        with io.BytesIO(gz_file.read()) as file:
            parameters = json.load(file)

    return dict_to_propertysetex(parameters)


def as_ndarray(input, dtype):
    retval = None

    if isinstance(input, np.ndarray):
        # Don't make yet another NumPy array ...
        retval = input
    elif isinstance(input, list):
        # Convert lists to NumPy arrays
        retval = np.array(input, dtype=dtype)
    else:
        # Convert other (assumed to be scalar) to a single entry NumPy array
        retval = np.array([input], dtype=dtype)

    return retval


def dict_to_propertysetex(parameters: dict) -> PropertySetEx:
    # Note the following canonicalizes the order of the locations based on the
    # order in the JSON file.
    # We might consider either
    # a) alphabetical order by location name or
    # b) order by int(ID)

    params = PropertySetEx(parameters)

    # No processing of "seed"

    params.date_start = datetime.strptime(params.date_start, "%Y-%m-%d") if isinstance(params.date_start, (str,)) else params.date_start  # noqa: DTZ007
    params.date_stop = datetime.strptime(params.date_stop, "%Y-%m-%d") if isinstance(params.date_stop, (str,)) else params.date_stop  # noqa: DTZ007
    params.nticks = (params.date_stop - params.date_start).days + 1
    logger.info(f"Simulation calendar dates: {params.date_start} to {params.date_stop} ({params.nticks} ticks)")

    # Handle single location instance (incoming data is scalar rather than list)
    if not isinstance(params.location_name, list):
        params.location_name = [params.location_name]

    num_ticks = params.nticks
    num_nodes = len(params.location_name)

    # IDs are 1-based not 0-based like indices
    # Map string IDs to names and names to indices (0-based)

    params.N_j_initial = as_ndarray(params.N_j_initial, dtype=np.uint32)
    params.S_j_initial = as_ndarray(params.S_j_initial, dtype=np.uint32)
    params.E_j_initial = as_ndarray(params.E_j_initial, dtype=np.uint32)
    params.I_j_initial = as_ndarray(params.I_j_initial, dtype=np.uint32)
    params.R_j_initial = as_ndarray(params.R_j_initial, dtype=np.uint32)
    params.V1_j_initial = as_ndarray(params.V1_j_initial, dtype=np.uint32)
    params.V2_j_initial = as_ndarray(params.V2_j_initial, dtype=np.uint32)

    params.b_jt = np.array(params.b_jt, dtype=np.float32)
    if params.b_jt.shape == (num_nodes, num_ticks):
        params.b_jt = np.array(params.b_jt.T)  # index on time, then location
    params.d_jt = np.array(params.d_jt, dtype=np.float32)
    if params.d_jt.shape == (num_nodes, num_ticks):
        params.d_jt = np.array(params.d_jt.T)  # index on time, then location

    params.nu_1_jt = np.array(params.nu_1_jt, dtype=np.float32)
    if params.nu_1_jt.shape == (num_nodes, num_ticks):
        params.nu_1_jt = np.array(params.nu_1_jt.T)  # index on time, then location
    params.nu_2_jt = np.array(params.nu_2_jt, dtype=np.float32)
    if params.nu_2_jt.shape == (num_nodes, num_ticks):
        params.nu_2_jt = np.array(params.nu_2_jt.T)  # index on time, then location

    # No processing of "phi_1"
    # No processing of "phi_2"
    # No processing of "omega_1"
    # No processing of "omega_2"
    # No processing of "iota"
    # No processing of "gamma_1"
    # No processing of "gamma_2"
    # No processing of "epsilon"

    params.mu_jt = np.array(params.mu_jt, dtype=np.float32)
    if params.mu_jt.shape == (num_nodes, num_ticks):
        params.mu_jt = np.array(params.mu_jt.T)  # index on time, then location

    # No processing of "rho"
    # No processing of "sigma"

    params.latitude = as_ndarray(params.latitude, dtype=np.float32)
    params.longitude = as_ndarray(params.longitude, dtype=np.float32)

    # No processing of "mobility_omega"
    # No processing of "mobility_gamma"

    params.tau_i = as_ndarray(params.tau_i, dtype=np.float32)
    assert np.all((params.tau_i >= 0.0) & (params.tau_i <= 1.0)), "tau_i values must be in the range [0, 1]"

    params.beta_j0_hum = as_ndarray(params.beta_j0_hum, dtype=np.float32)

    params.a_1_j = as_ndarray(params.a_1_j, dtype=np.float32)
    params.b_1_j = as_ndarray(params.b_1_j, dtype=np.float32)
    params.a_2_j = as_ndarray(params.a_2_j, dtype=np.float32)
    params.b_2_j = as_ndarray(params.b_2_j, dtype=np.float32)

    assert int(params.p) == params.p, f"p must be an integer, but got {params.p}"
    params.p = int(params.p)

    # No processing of "alpha_1"
    # No processing of "alpha_2"

    params.beta_j0_env = as_ndarray(params.beta_j0_env, dtype=np.float32).reshape(-1, 1)
    params.theta_j = as_ndarray(params.theta_j, dtype=np.float32)

    params.psi_jt = np.array(params.psi_jt, dtype=np.float32)
    if params.psi_jt.shape == (num_nodes, num_ticks):
        params.psi_jt = np.array(params.psi_jt.T)  # index on time, then location

    # No processing of "zeta_1"
    # No processing of "zeta_2"
    # No processing of "kappa"
    # No processing of "decay_days_short"
    # No processing of "decay_days_long"
    # No processing of "decay_shape_1"
    # No processing of "decay_shape_2"

    def convert(item):
        try:
            return int(item)
        except ValueError:
            pass
        return np.nan

    # If it's a list (of lists), convert to a numpy array. Otherwise assume it's already a numpy array.
    if isinstance(params.reported_cases, list):
        params.reported_cases = np.array([[convert(element) for element in row] for row in params.reported_cases])
    else:
        assert isinstance(params.reported_cases, np.ndarray), "reported_cases must be a list of lists or a numpy array"

    # If it's a list (of lists), convert to a numpy array. Otherwise assume it's already a numpy array.
    if isinstance(params.reported_deaths, list):
        params.reported_deaths = np.array([[convert(element) for element in row] for row in params.reported_deaths])
    else:
        assert isinstance(params.reported_deaths, np.ndarray), "reported_deaths must be a list of lists or a numpy array"

    # No processing of "return"

    return params


def load_hdf5_parameters(filename: Union[str, Path]) -> PropertySetEx:
    with h5.File(filename, "r") as h5file:
        parameters = load_hdf5(h5file)

    return parameters


def load_compressed_hdf5_parameters(filename: Union[str, Path]) -> PropertySetEx:
    with gzip.open(filename, "rb") as gz_file:
        with io.BytesIO(gz_file.read()) as file:
            with h5.File(file, "r") as h5file:
                parameters = load_hdf5(h5file)

    return parameters


def load_hdf5(h5file) -> PropertySetEx:
    ps = PropertySetEx()

    # date_start and date_stop
    start = h5file["date_start"][()][0]
    stop = h5file["date_stop"][()][0]
    epoch = datetime(year=1970, month=1, day=1)
    ps.date_start = epoch + timedelta(days=start)
    ps.date_stop = epoch + timedelta(days=stop)
    nticks = ps.nticks = (ps.date_stop - ps.date_start).days + 1  # +1 to include stop date

    # scalars
    for scalar in [
        "phi_1",
        "phi_2",
        "omega_1",
        "omega_2",
        "iota",
        "gamma_1",
        "gamma_2",
        "epsilon",
        "rho",
        "sigma",
        "alpha_1",
        "alpha_2",
        "zeta_1",
        "zeta_2",
        "kappa",
        "decay_days_short",
        "decay_days_long",
        "decay_shape_1",
        "decay_shape_2",
    ]:
        ps[scalar] = h5file[scalar][()][0]

    # per location vectors
    for vector in [
        "location_name",
        "S_j_initial",
        "E_j_initial",
        "I_j_initial",
        "R_j_initial",
        "V1_j_initial",
        "V2_j_initial",
        "beta_j0_hum",
        "beta_j0_env",
        "tau_i",
        "theta_j",
    ]:
        ps[vector] = h5file[vector][()]

    npatches = len(ps.location_name)

    # per tick, per location arrays
    for array in ["b_jt", "d_jt", "nu_1_jt", "nu_2_jt", "mu_jt", "psi_jt"]:
        temp = np.zeros((nticks, npatches), dtype=np.float32)
        for ipatch in range(ps.location_name):
            temp[:, ipatch] = h5file[array][str(ipatch + 1)][()]
        ps[array] = temp

    return ps


def validate_parameters(params: PropertySetEx) -> None:
    # date_start and date_stop
    assert params.date_stop >= params.date_start, f"date_stop ({params.date_stop}) must be >= date_start ({params.date_start})"

    npatches = len(params.location_name)

    assert params.S_j_initial.shape == (npatches,), (
        f"Number of S_j_initial values ({len(params.S_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.S_j_initial >= 0), "S_j_initial values must be non-negative"
    assert params.E_j_initial.shape == (npatches,), (
        f"Number of E_j_initial values ({len(params.E_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.E_j_initial >= 0), "E_j_initial values must be non-negative"
    assert params.I_j_initial.shape == (npatches,), (
        f"Number of I_j_initial values ({len(params.I_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.I_j_initial >= 0), "I_j_initial values must be non-negative"
    assert params.R_j_initial.shape == (npatches,), (
        f"Number of R_j_initial values ({len(params.R_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.R_j_initial >= 0), "R_j_initial values must be non-negative"
    assert params.V1_j_initial.shape == (npatches,), (
        f"Number of V1_j_initial values ({len(params.V1_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.V1_j_initial >= 0), "V1_j_initial values must be non-negative"
    assert params.V2_j_initial.shape == (npatches,), (
        f"Number of V2_j_initial values ({len(params.V2_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.V2_j_initial >= 0), "V2_j_initial values must be non-negative"

    nticks = params.nticks

    # shape of b_jt = (nticks, npatches)
    assert params.b_jt.shape == (nticks, npatches), (
        f"Shape of b_jt {params.b_jt.shape} does not match (nticks, npatches) = ({nticks}, {npatches})"
    )
    # 0 <= b_jt <= 0.0001369863 (5% annual / 365 days)
    assert np.all(params.b_jt >= 0.0), "b_jt rate values must be positive"

    # shape of b_jt = (nticks, npatches)
    assert params.d_jt.shape == (nticks, npatches), (
        f"Shape of d_jt {params.d_jt.shape} does not match (nticks, npatches) = ({nticks}, {npatches})"
    )
    # 0 <= d_jt <= 0.0002739726 (10% annual / 365 days)
    assert np.all(params.d_jt >= 0.0), "d_jt rate values must be positive"

    # shape of nu_1_jt = (nticks, npatches)
    assert params.nu_1_jt.shape == (nticks, npatches), (
        f"Shape of nu_1_jt {params.nu_1_jt.shape} does not match (nticks, npatches) = ({nticks}, {npatches})"
    )
    # # nu_1_jt - no daily value can be larger than the country population (N_j_initial) / 7
    # assert np.all(params.nu_1_jt <= params.N_j_initial[np.newaxis, :] / 7), (
    #     "nu_1_jt values must not exceed N_j_initial / 7 for any location"
    # )

    # shape of nu_2_jt = (nticks, npatches)
    assert params.nu_2_jt.shape == (nticks, npatches), (
        f"Shape of nu_2_jt {params.nu_2_jt.shape} does not match (nticks, npatches) = ({nticks}, {npatches})"
    )
    # # nu_2_jt - no daily value can be larger than the country population (N_j_initial) / 7
    # assert np.all(params.nu_2_jt <= params.N_j_initial[np.newaxis, :] / 7), (
    #     "nu_2_jt values must not exceed N_j_initial / 7 for any location"
    # )

    # phi_1 and phi_2 must be between 0 (completely ineffective) and 1 (fully effective)
    assert (params.phi_1 >= 0.0) & (params.phi_1 <= 1.0), "phi_1 value must be in the range [0, 1]"
    assert (params.phi_2 >= 0.0) & (params.phi_2 <= 1.0), "phi_2 value must be in the range [0, 1]"

    # omega_1 and omega_2 must be above zero
    assert (params.omega_1 >= 0.0), "omega_1 value must be positive"
    assert (params.omega_2 >= 0.0), "omega_2 value must be positive"

    # iota must be above zero
    assert (params.iota >= 0.0), "iota value must be positive"

    # gamma_1 and gamma_2 must be positive
    assert (params.gamma_1 >= 0.0), "gamma_1 value must be positive"
    assert (params.gamma_2 >= 0.0), "gamma_2 value must be positive"

    # epsilon must be positive
    assert (params.epsilon >= 0.0), "epsilon value must be positive"

    # shape of mu_jt = (nticks, npatches)
    assert params.mu_jt.shape == (nticks, npatches), (
        f"Shape of mu_jt {params.mu_jt.shape} does not match (nticks, npatches) = ({nticks}, {npatches})"
    )
    # all mu_jt must be above zero
    assert np.all((params.mu_jt >= 0.0)), "mu_jt values must be positive"

    # rho must be between 0 (all false positives) and 1 (no false positives)
    assert (params.rho >= 0.0) & (params.rho <= 1.0), "rho value must be in the range [0, 1]"

    # sigma must be between 0 (all asymptomatic) and 1 (all symptomatic)
    assert (params.sigma >= 0.0) & (params.sigma <= 1.0), "sigma value must be in the range [0, 1]"

    # Number of lat/long values must match number of patches
    assert len(params.latitude) == npatches, (
        f"Number of latitude values ({len(params.latitude)}) does not match number of locations ({npatches})"
    )
    assert len(params.longitude) == npatches, (
        f"Number of longitude values ({len(params.longitude)}) does not match number of locations ({npatches})"
    )
    # omega and gamma required to build pi_ij matrix with "power_norm"
    assert "mobility_omega" in params, "Parameters: 'mobility_omega' not found in parameters"
    assert "mobility_gamma" in params, "Parameters: 'mobility_gamma' not found in parameters"

    # Number of seasonality parameters must match number of patches
    assert len(params.a_1_j) == npatches, f"Number of a_1_j values ({len(params.a_1_j)}) does not match number of locations ({npatches})"
    assert len(params.b_1_j) == npatches, f"Number of b_1_j values ({len(params.b_1_j)}) does not match number of locations ({npatches})"
    assert len(params.a_2_j) == npatches, f"Number of a_2_j values ({len(params.a_2_j)}) does not match number of locations ({npatches})"
    assert len(params.b_2_j) == npatches, f"Number of b_2_j values ({len(params.b_2_j)}) does not match number of locations ({npatches})"
    assert "p" in params, "Parameters: 'p' (seasonality phase) not found in parameters"

    # length of beta_j0_hum must be equal to number of locations
    assert len(params.beta_j0_hum) == npatches, (
        f"Number of beta_j0_hum values ({len(params.beta_j0_hum)}) does not match number of locations ({npatches})"
    )
    # beta_j0_hum must be >= 0
    assert np.all(params.beta_j0_hum >= 0.0), "beta_j0_hum values must be >= 0"

    # length of tau_i must be equal to number of locations
    assert len(params.tau_i) == npatches, f"Number of tau_i values ({len(params.tau_i)}) does not match number of locations ({npatches})"
    # tau_i must be between 0 (no emigration) and 1 (all emigration)
    assert np.all((params.tau_i >= 0.0) & (params.tau_i <= 1.0)), "tau_i values must be in the range [0, 1]"

    # alpha_1 and alpha_2
    # TODO - TBD

    # alpha_1 must be above 0 (zero population mixing) and below 1 (full mass action), cannot equal zero
    assert (params.alpha_1 > 0.0) & (params.alpha_1 <= 1.0), "alpha_1 value must be in the range [0, 1]"

    # alpha_2 must be between 0 (full density dependence) and 1 (full frequency dependence)
    assert (params.alpha_2 >= 0.0) & (params.alpha_2 <= 1.0), "alpha_1 value must be in the range [0, 1]"

    # length of beta_j0_env must be equal to number of locations
    assert len(params.beta_j0_env) == npatches, (
        f"Number of beta_j0_env values ({len(params.beta_j0_env)}) does not match number of locations ({npatches})"
    )
    # beta_j0_env must be >= 0
    assert np.all(params.beta_j0_env >= 0.0), "beta_j0_env values must be >= 0"

    # length of theta_j must be equal to number of locations
    assert len(params.theta_j) == npatches, (
        f"Number of theta_j values ({len(params.theta_j)}) does not match number of locations ({npatches})"
    )
    # theta_j must be between 0 (no WASH intervention) and 1 (full WASH protection)
    assert np.all((params.theta_j >= 0.0) & (params.theta_j <= 1.0)), "theta_j values must be in the range [0, 1]"

    # shape of psi_jt = (nticks, npatches)
    assert params.psi_jt.shape == (nticks, npatches), (
        f"Shape of psi_jt {params.psi_jt.shape} does not match (nticks, npatches) = ({nticks}, {npatches})"
    )

    # psi_jt
    # TODO - TBD

    # zeta_1 and zeta_2 must be >= 0
    assert params.zeta_1 >= 0.0, "zeta_1 value must be >= 0"
    assert params.zeta_2 >= 0.0, "zeta_2 value must be >= 0"
    # TODO - TBD any other limits

    # kappa must be >= 0
    assert params.kappa >= 0.0, "kappa value must be >= 0"

    # decay_days_short > 0.0
    assert params.decay_days_short > 0.0, f"decay_days_short value must be > 0 {params.decay_days_short=}"
    # decay_days_short <= decay_days_long
    assert params.decay_days_short <= params.decay_days_long, (
        f"decay_days_short ({params.decay_days_short}) value must be <= decay_days_long ({params.decay_days_long})"
    )

    return


class Parameters:
    def __init__(self, model) -> None:
        self.model = model

        return

    def check(self):
        # assert hasattr(self.model, "patches"), "Parameters: model needs to have a 'patches' attribute."
        # assert hasattr(self.model, "people"), "Parameters: model needs to have a 'people' attribute."
        assert hasattr(self.model, "params"), "Parameters: model needs to have a 'params' attribute."

        return

    def __call__(self, _model, _tick):
        pass

    def plot(self, fig: Figure = None):  # pragma: no cover
        # Stacked bar chart of initial populations
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Initial Populations by Category") if fig is None else fig

        categories = ["S_j_initial", "E_j_initial", "I_j_initial", "R_j_initial", "V1_j_initial", "V2_j_initial"]
        data = [getattr(self.model.params, category) for category in categories]

        x = np.arange(len(self.model.params.location_name))
        bottom = np.zeros(len(self.model.params.location_name))

        for category, values in zip(categories, data):
            plt.bar(x, values, bottom=bottom, label=category)
            bottom += values

        plt.xticks(x, self.model.params.location_name, rotation=45, ha="right")
        plt.xlabel("Location Name")
        plt.ylabel("Population")
        plt.legend()

        yield "Initial Populations by Category"

        # Birth rates by location over time
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Birth Rates by Location Over Time") if fig is None else fig

        plt.imshow(self.model.params.b_jt.T, aspect="auto", cmap="Blues", interpolation="nearest")
        plt.colorbar(label="Birth Rate")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "Birth Rates by Location Over Time"

        # Mortality rates by location over time
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Non-Disease Mortality Rates by Location Over Time") if fig is None else fig

        plt.imshow(self.model.params.d_jt.T, aspect="auto", cmap="Reds", interpolation="nearest")
        plt.colorbar(label="Mortality Rate")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "Non-Disease Mortality Rates by Location Over Time"

        # Vaccination (first dose) rates by location over time
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="First Dose Vaccination Counts by Location Over Time") if fig is None else fig

        plt.imshow(self.model.params.nu_1_jt.T, aspect="auto", cmap="Greens", interpolation="nearest")
        plt.colorbar(label="Vaccination Count")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "First Dose Vaccination Counts by Location Over Time"

        # Vaccination (second dose) rates by location over time
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Second Dose Vaccination Counts by Location Over Time") if fig is None else fig

        plt.imshow(self.model.params.nu_2_jt.T, aspect="auto", cmap="Greens", interpolation="nearest")
        plt.colorbar(label="Vaccination Count")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "Second Dose Vaccination Counts by Location Over Time"

        # Disease mortality rate over time
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Disease Mortality Rate by Location Over Time") if fig is None else fig

        plt.imshow(self.model.params.mu_jt.T, aspect="auto", cmap="Reds", interpolation="nearest")
        plt.colorbar(label="Disease Mortality Rate")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "Disease Mortality Rate by Location Over Time"

        # Emmigration probability rates by location
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Emigration Probabilities by Location") if fig is None else fig

        plt.scatter(self.model.params.location_name, self.model.params.tau_i, marker="x", color="purple")
        plt.xlabel("Location Name")
        plt.ylabel("Emigration Probability")
        plt.xticks(rotation=45, ha="right")

        yield "Emigration Probabilities by Location"

        # WASH fraction by location
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="WASH Coverage by Location") if fig is None else fig

        plt.scatter(self.model.params.location_name, self.model.params.theta_j, marker="x", color="purple")
        plt.xlabel("Location Name")
        plt.ylabel("WASH Coverage")
        plt.xticks(rotation=45, ha="right")

        yield "WASH Coverage by Location"

        # Environmental suitability factor by location over time
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Environmental Suitability Factor by Location Over Time") if fig is None else fig

        plt.imshow(self.model.params.psi_jt.T, aspect="auto", cmap="Blues", interpolation="nearest")
        plt.colorbar(label="Environmental Suitability Factor")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "Environmental Suitability Factor by Location Over Time"

        return
