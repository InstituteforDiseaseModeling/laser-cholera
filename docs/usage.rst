=====
Usage
=====

Computing the model log-likelihood
----------------------------------

The :func:`laser.cholera.calc_model_likelihood.calc_model_likelihood` function
scores a model fit against observed cases and deaths. It accepts four 2-D
arrays of shape ``(n_locations, n_time_steps)`` and returns a single
log-likelihood scalar.

Minimal core call (no shape terms)
----------------------------------

The minimal call uses only the Negative Binomial core; all four shape weights
default to ``0`` (off):

.. code-block:: python

    >>> import numpy as np
    >>> from laser.cholera.calc_model_likelihood import calc_model_likelihood
    >>> obs_cases = np.array([[5, 8, 12, 7], [3, 6, 9, 4]], dtype=float)
    >>> est_cases = np.array([[6, 9, 11, 7], [4, 6, 8, 5]], dtype=float)
    >>> obs_deaths = np.zeros_like(obs_cases)
    >>> est_deaths = np.zeros_like(est_cases)
    >>> ll = calc_model_likelihood(obs_cases, est_cases, obs_deaths, est_deaths)
    >>> ll < 0
    True

A perfect match returns the maximum (least-negative) log-likelihood; any
deviation between observed and estimated values reduces it.

Enabling the shape terms
------------------------

Set any of the shape-term weights to a positive value to enable additional
calibration signals on top of the NB core. A weight of ``0.25`` contributes
roughly 25 % as much as the NB core (the terms are T-normalized internally):

* ``weight_peak_timing`` — Normal prior on the per-location peak timing offset
  in weeks; requires ``epidemic_peaks``, ``date_start``, and ``date_stop``.
* ``weight_peak_magnitude`` — log-Normal with adaptive sigma on the
  observed-vs-estimated peak ratio; same requirements as peak timing.
* ``weight_cumulative_total`` — NB log-likelihood on cumulative sums at
  fractional timepoints (defaults to 25 %, 50 %, 75 %, 100 % of the series).
* ``weight_wis`` — negated Weighted Interval Score on a set of quantile levels.

The ``epidemic_peaks`` argument is a pandas DataFrame with columns
``iso_code``, ``peak_date``, and ``loc_idx`` (a 0-based integer index into the
rows of the observation arrays). Construction of this DataFrame is handled
automatically by :func:`laser.cholera.metapop.params.get_parameters` when the
incoming JSON has an ``epidemic_peaks`` entry.

Integration with the metapopulation model
-----------------------------------------

In normal use the function is invoked from
:class:`laser.cholera.metapop.analyzer.Analyzer` on the final tick of the
simulation. The analyzer reads ``model.params.reported_cases`` /
``reported_deaths`` for the observations, ``model.results.reported_cases`` /
``model.patches.reported_deaths`` for the estimates, and any per-shape-term
weights from ``model.params``.

To compute the likelihood against a finished model directly:

.. code-block:: python

    from laser.cholera.calc_model_likelihood import calc_model_likelihood

    nreports = min(
        model.params.reported_cases.shape[1],
        model.patches.incidence.shape[0] - 1,
    )
    ll = calc_model_likelihood(
        obs_cases=model.params.reported_cases[:, :nreports],
        est_cases=model.results.reported_cases[:, :nreports],
        obs_deaths=model.params.reported_deaths[:, :nreports],
        est_deaths=model.patches.reported_deaths[:, :nreports],
        epidemic_peaks=model.params.epidemic_peaks,
        date_start=model.params.date_start,
        date_stop=model.params.date_stop,
        weight_peak_timing=0.25,
    )

See the API reference for :func:`~laser.cholera.calc_model_likelihood.calc_model_likelihood`
for the full argument list and the per-location assembly formula.
