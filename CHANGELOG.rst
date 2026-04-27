
Changelog
=========

0.12.5 (unreleased)
-------------------

* Add src/laser/cholera/calc_log_likelihood_distributions.py: Python translation of calc_log_likelihood_distributions.R (Beta, Binomial, Gamma, NegBin, Normal, Poisson)
* Rename src/laser/cholera/spring_likelihood.py → calc_model_likelihood.py
* Add tests/test_calc_log_likelihood_negbin.py: Python translation of test_calc_log_likelihood_negbin.R; update import to calc_log_likelihood_distributions
* Add tests/test_calc_model_likelihood.py: Python translation of test_calc_model_likelihood.R
* Add tests/test_calc_model_likelihood_extreme.py: Python translation of test_calc_model_likelihood_extreme.R
* Add tests/test_calc_model_likelihood_reference.py: Python translation of test_calc_model_likelihood_reference.R
* Add tests/test_compute_wis_parametric_row.py: Python translation of test_compute_wis_parametric_row.R
* Add tests/test_ll_cumulative_progressive_nb.py: Python translation of test_ll_cumulative_progressive_nb.R
* Add tests/test_nb_size_from_obs_weighted.py: Python translation of test_nb_size_from_obs_weighted.R

0.10.1 (2026-01-16)
-------------------

* Add tests for new IFR implementation

0.10.0 (2026-01-15)
-------------------

* New IFR model (Infection Fatality Ratio)
* Update observation process with rho and chi based on infectious prevalence and diagnostic rates
* Update default_parameters.json and LICENSE copyright dates
* Test fixes for NumPy scalar serialization
* Remove MacOS x86_64 from test matrix
* Linter issues and GitHub runner fixes

0.9.1 (2025-10-02)
------------------

* Fix typo infective -> ineffective
* Add checks against populations going negative
* Expose new_symptomatic
* Only print if verbose is True in parameters
* Skip likelihood check unless "calc_likelihood" is in parameters
* Address linter issues
* Bugfix for parameter constraints (alphas)

0.9.0 (2025-08-19)
------------------

* Support single location configuration

0.8.0 (2025-07-24)
------------------

* Spatial hazard computation fix (don't transpose pi_ij in model.results)

0.7.11 (2025-07-11)
-------------------

* Trim and transpose for convenience in MOSAIC

0.7.10 (2025-07-10)
-------------------

* Fix bug in double counting Vxinf
* Fix bug in suitability to decay calculations
* Update default parameters
* Fix indexing for human daily seasonality

0.7.9 (2025-06-06)
------------------

* Rename beta_env to beta_jt_env
* Rename beta_j_seasonality to beta_jt_human and use directly in spatial hazard calculation
* Update pre-commit
* Fix handling of pi_ij matrix math
* Track vaccine doses delivered
* Births should be Poisson rather than binomial
* Rename estimated to simulated for clarity
* Switch from 'agents' to 'people' terminology
* Fix coupling calculation for denominator == 0

0.7.8 (2025-05-16)
------------------

* Likelihood cleanup for NaNs and all zeros

0.7.7 (2025-05-13)
------------------

* Calculate log likelihood at end of simulation

0.7.6 (2025-05-13)
------------------

* Fix logging setup and np.var() usage

0.7.5 (2025-05-13)
------------------

* Add Python implementation of R tests for likelihood functions

0.7.4 (2025-05-07)
------------------

* Fix up reading JSON files back into memory (handle actual NaN vs "NA" or "NaN")
* Record incidence (total and per source)
* Adding likelihood functions
* Adding likelihood function tests
* More consistent variable names

0.7.3 (2025-04-30)
------------------

* Gate file output on hdf5_output and "return" config parameters
* Clean up console output with logging infrastructure
* Add "quiet" parameter to suppress console progress bar (defaults to False for CLI, True for programmatic interface)
* Update GHA to run tests on push to main
* Support params from R (numeric values come in as doubles, but we need an integer for p)

0.7.2 (2025-04-24)
------------------

* Support for passing dict to get_parameters()
* Tests for run_model() function
* Additional tests for tracking vital statistics (births, non-disease deaths, disease deaths)

0.7.1 (2025-04-24)
------------------

* Minor version bump

0.7.0 (2025-04-23)
------------------

* Initial alpha release
* Support passing parameter dictionary to run_model()
* Fix mapping of environmental suitability (psi_jt) to decay parameter (delta_jt)
* Update default_parameters.json with matrices
* Update parameter loading for matrices
* Clean up plotting and fix seasonality phase
* Handle command line parameter overrides
* Enable parameter overrides correctly
* Allow test parameter sets to skip validation
* Pin numpy, numba, and llvmlite versions
* Remove subpackages
* Update laser-core dependency
* Return model from run_model()
* Require Numba that supports NumPy>=2.0
* Update shedding to environment based on theta_j
* Updated parameters including switch from delta_min/delta_max to decay_days_fast/decay_days_slow
* Use decay_shape_1 and decay_shape_2 to parameterize scipy.stats.beta.cdf
* Add version bump, build, and release GHA
* Metapop implementation work-in-progress commits

0.0.0 (2024-09-30)
------------------

* First release on PyPI
