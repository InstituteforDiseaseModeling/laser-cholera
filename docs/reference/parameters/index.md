# Parameter reference

Every top-level field in [`default_parameters.json`](https://github.com/InstituteforDiseaseModeling/laser-cholera/blob/main/src/laser/cholera/metapop/data/default_parameters.json) is documented on one of the ten parameter group pages. Each parameter has a definition, shape, dtype, valid range, a link to the model section that consumes it, and — most importantly — the value that turns the feature it gates off. The grouped table below lets you skim a feature area (vaccination, mobility, environmental transmission, ...) and jump straight to the entries on that page; the alphabetical table at the bottom is the lookup index for "I have a parameter name, where is it?".

Every parameter listed below is exercised by the coverage test `tests/test_docs_param_coverage.py`, which fails if a `default_parameters.json` key is undocumented or appears on more than one group page. The test is the source of truth for completeness; this index is the source of truth for navigation.

## Grouped tables

### Run identity and calendar

Anchors the run: PRNG seed, calendar window, and the ordered patch labels that define `npatches` and every per-patch vector's element order.

| Parameter | One-line description |
|---|---|
| [`date_start`](run-identity.md#date_start) | Calendar date of simulation tick 0; together with date_stop it defines the run length (nticks = (date_stop - date_start).days + 1). |
| [`date_stop`](run-identity.md#date_stop) | Calendar date of the final simulation tick; with date_start it bounds the run window and sets nticks. |
| [`location_name`](run-identity.md#location_name) | Ordered list of patch/admin-unit labels; its length defines npatches and its order defines every per-patch vector's element ordering. |
| [`seed`](run-identity.md#seed) | Integer seed for the model's pseudo-random number generator; setting it pins all stochastic draws to a reproducible sequence. |

### Initial populations

Per-patch seed counts (and unused proportions) for the seven live compartments at tick 0.

| Parameter | One-line description |
|---|---|
| [`E_j_initial`](initial-populations.md#e_j_initial) | Per-patch initial number of exposed (latent, not yet infectious) people seeded into the E compartment at tick 0. |
| [`I_j_initial`](initial-populations.md#i_j_initial) | Per-patch initial number of infectious people; split by sigma into symptomatic (Isym) and asymptomatic (Iasym) at tick 0. |
| [`N_j_initial`](initial-populations.md#n_j_initial) | Total initial population per patch (used as the gravity-model destination mass when reconstructing the mobility matrix, though the live code now sums the per-compartment initials instead). |
| [`R_j_initial`](initial-populations.md#r_j_initial) | Per-patch initial number of recovered (immune) people seeded into the R compartment at tick 0. |
| [`S_j_initial`](initial-populations.md#s_j_initial) | Per-patch initial number of susceptible people seeded into the S compartment at tick 0. |
| [`V1_j_initial`](initial-populations.md#v1_j_initial) | Per-patch initial number of one-dose-vaccinated people seeded into the V1 compartment at tick 0. |
| [`V2_j_initial`](initial-populations.md#v2_j_initial) | Per-patch initial number of two-dose-vaccinated people seeded into the V2 compartment at tick 0. |
| [`prop_E_initial`](initial-populations.md#prop_e_initial) | Per-patch initial fraction of population in the E compartment (informational / upstream-MOSAIC artifact; not read by the model). |
| [`prop_I_initial`](initial-populations.md#prop_i_initial) | Per-patch initial fraction of population in the I compartment (informational / upstream-MOSAIC artifact; not read by the model). |
| [`prop_R_initial`](initial-populations.md#prop_r_initial) | Per-patch initial fraction of population in the R compartment (informational / upstream-MOSAIC artifact; not read by the model). |
| [`prop_S_initial`](initial-populations.md#prop_s_initial) | Per-patch initial fraction of population in the S compartment (informational / upstream-MOSAIC artifact; not read by the model itself). |
| [`prop_V1_initial`](initial-populations.md#prop_v1_initial) | Per-patch initial fraction of population in the V1 compartment (informational / upstream-MOSAIC artifact; not read by the model). |
| [`prop_V2_initial`](initial-populations.md#prop_v2_initial) | Per-patch initial fraction of population in the V2 compartment (informational / upstream-MOSAIC artifact; not read by the model). |

### Vital dynamics

Per-tick birth, non-disease mortality, and disease-mortality rates applied to the live compartments.

| Parameter | One-line description |
|---|---|
| [`b_jt`](vital-dynamics.md#b_jt) | Per-patch, per-tick crude birth rate; the Poisson mean for new susceptibles each day is `N * b_jt[tick]`. |
| [`d_jt`](vital-dynamics.md#d_jt) | Per-patch, per-tick non-disease (background) mortality rate applied to every live compartment (S, E, Isym, Iasym, R, V1, V2). |
| [`mu_j_baseline`](vital-dynamics.md#mu_j_baseline) | Per-patch baseline disease (cholera) mortality rate applied to symptomatic infectious; the multiplicative anchor of the `mu_jt = mu_baseline * (1 + slope*t) * (1 + epidemic_factor*flag)` formula. |
| [`mu_j_epidemic_factor`](vital-dynamics.md#mu_j_epidemic_factor) | Per-patch multiplicative boost on disease mortality when the patch is flagged as being in an epidemic regime (reported symptomatic infectious exceeds `epidemic_threshold * N`). |
| [`mu_j_slope`](vital-dynamics.md#mu_j_slope) | Per-patch linear time-trend on disease mortality; `mu_jt` is scaled by `(1 + mu_j_slope * tick/nticks)` so positive values ramp mortality up over the run and negative values ramp it down. |
| [`mu_jt`](vital-dynamics.md#mu_jt) | Disease mortality time-series stored on the parameter set for diagnostic plotting only — the simulation loop computes its own per-tick mu_jt from baseline/slope/epidemic_factor and ignores this array. |

### Vaccination

First- and second-dose campaigns: delivery counts, dose effectiveness, waning, and the source-compartment list.

| Parameter | One-line description |
|---|---|
| [`nu_1_jt`](vaccination.md#nu_1_jt) | Number of first-dose vaccinations to deliver per tick and per patch. |
| [`nu_2_jt`](vaccination.md#nu_2_jt) | Number of second-dose vaccinations to deliver per tick and per patch (moves V1 to V2). |
| [`nu_jt_sources`](vaccination.md#nu_jt_sources) | List of compartment names from which first-dose vaccinations are drawn (proportional sampling across the listed compartments). |
| [`omega_1`](vaccination.md#omega_1) | Daily waning rate from V1 back to S; converted to a per-tick probability via 1 - exp(-omega_1). |
| [`omega_2`](vaccination.md#omega_2) | Daily waning rate from V2 back to S; converted to a per-tick probability via 1 - exp(-omega_2). |
| [`phi_1`](vaccination.md#phi_1) | Effectiveness (take fraction) of a first dose: only this fraction of delivered doses moves the recipient into V1. |
| [`phi_2`](vaccination.md#phi_2) | Effectiveness (take fraction) of a second dose: only this fraction of V1 recipients moves into V2. |

### Disease progression

Per-day rates for the SEIR backbone: latency, recovery, waning natural immunity, and the symptomatic/asymptomatic split.

| Parameter | One-line description |
|---|---|
| [`epsilon`](disease-progression.md#epsilon) | Per-day rate at which natural immunity wanes (R -> S). |
| [`gamma_1`](disease-progression.md#gamma_1) | Per-day recovery rate for symptomatic infectious individuals (I_sym -> R). |
| [`gamma_2`](disease-progression.md#gamma_2) | Per-day recovery rate for asymptomatic infectious individuals (I_asym -> R). |
| [`iota`](disease-progression.md#iota) | Per-day rate at which exposed (E) individuals progress to becoming infectious (I_sym or I_asym). |
| [`sigma`](disease-progression.md#sigma) | Fraction of new infections that become symptomatic (the rest go to the asymptomatic compartment). |

### Reporting

Observed-data inputs (case / death series) and the under-reporting + lag knobs that turn modeled cases into reported cases.

| Parameter | One-line description |
|---|---|
| [`delta_reporting_cases`](reporting.md#delta_reporting_cases) | Integer tick lag between when a new symptomatic case occurs and when it is added to reported_cases. |
| [`delta_reporting_deaths`](reporting.md#delta_reporting_deaths) | Integer tick lag between when a disease death occurs and when it is added to reported_deaths. |
| [`reported_cases`](reporting.md#reported_cases) | Observed per-patch-per-time-step case counts used as the target series when computing the model likelihood. |
| [`reported_deaths`](reporting.md#reported_deaths) | Observed per-patch-per-time-step death counts used as the target series when computing the model likelihood. |
| [`rho`](reporting.md#rho) | Per-case reporting probability for symptomatic cases — fraction of new symptomatic infections that show up in the reported_cases series. |
| [`rho_deaths`](reporting.md#rho_deaths) | Per-death reporting probability — fraction of disease deaths that show up in the reported_deaths series. |

### Regime switching

Endemic-vs-epidemic regime: threshold, reporting-rate divisors per regime, and the observed peak dates the likelihood scorer uses.

| Parameter | One-line description |
|---|---|
| [`chi_endemic`](regime-switching.md#chi_endemic) | Reporting-rate divisor used when local symptomatic-infected fraction is below epidemic_threshold (endemic regime); inflates estimated cases via division. |
| [`chi_epidemic`](regime-switching.md#chi_epidemic) | Reporting-rate divisor used when local symptomatic-infected fraction is at or above epidemic_threshold (epidemic regime); inflates estimated cases via division. |
| [`epidemic_peaks`](regime-switching.md#epidemic_peaks) | Observed epidemic peak dates per ISO code used by the likelihood scorer to evaluate peak-timing and peak-magnitude shape terms; not consumed by the simulation dynamics. |
| [`epidemic_threshold`](regime-switching.md#epidemic_threshold) | Per-patch (or global) symptomatic-infected fraction above which the patch is treated as in an epidemic regime, switching both the disease-mortality multiplier and the reporting-rate divisor. |

### Geography and mobility

Per-patch coordinates, the master spatial-mixing fraction `tau_i`, and the gravity-model exponents that build `pi_ij`.

| Parameter | One-line description |
|---|---|
| [`latitude`](geography-and-mobility.md#latitude) | Per-patch latitude coordinate; combined with longitude for inter-patch distance in the gravity-mobility kernel (and used to order patches in one plot). |
| [`longitude`](geography-and-mobility.md#longitude) | Per-patch longitude coordinate; used together with latitude to compute great-circle distances that feed the gravity-model mobility matrix. |
| [`mobility_gamma`](geography-and-mobility.md#mobility_gamma) | Exponent on inter-patch distance (with sign reversed) in the gravity model; larger values penalize long-distance movement more strongly when building pi_ij. |
| [`mobility_omega`](geography-and-mobility.md#mobility_omega) | Exponent on the destination population in the gravity model that builds the patch-to-patch coupling matrix pi_ij; larger values pull more flow toward larger patches. |
| [`tau_i`](geography-and-mobility.md#tau_i) | Per-patch fraction of infectious (and, for environmental exposure, susceptible) individuals that move out of their home patch; the master switch for spatial mixing. |

### Human-to-human transmission

Per-patch baseline transmission, the two-harmonic seasonal envelope, and the mixing-nonlinearity exponents.

| Parameter | One-line description |
|---|---|
| [`a_1_j`](human-transmission.md#a_1_j) | Per-patch amplitude of the first (annual) cosine term of the seasonal-transmission harmonic. |
| [`a_2_j`](human-transmission.md#a_2_j) | Per-patch amplitude of the second (semi-annual) cosine term of the seasonal-transmission harmonic. |
| [`alpha_1`](human-transmission.md#alpha_1) | Exponent on the effective-infected count in the human-to-human force-of-infection (mixing nonlinearity in the numerator). |
| [`alpha_2`](human-transmission.md#alpha_2) | Exponent on the patch population in the FOI denominator; alpha_2=1 = frequency-dependent mixing, alpha_2=0 = density-dependent. |
| [`b_1_j`](human-transmission.md#b_1_j) | Per-patch amplitude of the first (annual) sine term of the seasonal-transmission harmonic. |
| [`b_2_j`](human-transmission.md#b_2_j) | Per-patch amplitude of the second (semi-annual) sine term of the seasonal-transmission harmonic. |
| [`beta_j0_hum`](human-transmission.md#beta_j0_hum) | Per-patch baseline human-to-human transmission rate; multiplied by the seasonal harmonic to produce the per-tick, per-patch transmission envelope. |
| [`beta_j0_tot`](human-transmission.md#beta_j0_tot) | Per-patch combined baseline transmission rate (legacy / metadata field carried from upstream MOSAIC R config; not consumed by the Python simulation loop). |
| [`p`](human-transmission.md#p) | Period (in ticks/days) of the seasonal-transmission harmonic; typically 365 for an annual cycle. |
| [`p_beta`](human-transmission.md#p_beta) | Per-patch split fraction between human-to-human and environmental transmission in the upstream R model (legacy metadata; not consumed by the Python simulation loop). |

### Environmental transmission

Environmental-reservoir transmission to humans, WASH coverage, shedding rates, suitability forcing, and reservoir decay.

| Parameter | One-line description |
|---|---|
| [`beta_j0_env`](environmental-transmission.md#beta_j0_env) | Per-patch baseline environmental-reservoir-to-human transmission rate; multiplier on the W/(kappa+W) saturation term in the env force of infection. |
| [`decay_days_long`](environmental-transmission.md#decay_days_long) | Longest reservoir half-life (in days) at the low end of the suitability-to-decay map; 1/decay_days_long is the minimum decay rate. |
| [`decay_days_short`](environmental-transmission.md#decay_days_short) | Shortest reservoir half-life (in days) at the high end of the suitability-to-decay map; 1/decay_days_short is the maximum decay rate. |
| [`decay_days_spread`](environmental-transmission.md#decay_days_spread) | MOSAIC-side scratch parameter describing the spread between fast and slow decay (days); carried in the parameter set but NOT consumed by any laser-cholera simulation code. |
| [`decay_shape_1`](environmental-transmission.md#decay_shape_1) | Alpha shape parameter of the Beta CDF used to map psi_jt in [0, 1] onto the [decay_days_short, decay_days_long] interval. |
| [`decay_shape_2`](environmental-transmission.md#decay_shape_2) | Beta shape parameter of the Beta CDF used to map psi_jt in [0, 1] onto the [decay_days_short, decay_days_long] interval. |
| [`kappa`](environmental-transmission.md#kappa) | Half-saturation constant in the env force-of-infection W / (kappa + W); larger kappa means more reservoir is needed before transmission saturates. |
| [`psi_jt`](environmental-transmission.md#psi_jt) | Per-tick, per-patch environmental suitability time series; modulates seasonal env transmission rate around its mean and drives the suitability-to-decay map for the reservoir. |
| [`psi_star_a`](environmental-transmission.md#psi_star_a) | Per-patch upstream-MOSAIC parameter for the suitability curve; carried in the parameter set but NOT consumed by any laser-cholera simulation code. |
| [`psi_star_b`](environmental-transmission.md#psi_star_b) | Per-patch upstream-MOSAIC parameter for the suitability curve; carried in the parameter set but NOT consumed by any laser-cholera simulation code. |
| [`psi_star_k`](environmental-transmission.md#psi_star_k) | Per-patch upstream-MOSAIC parameter for the suitability curve; carried in the parameter set but NOT consumed by any laser-cholera simulation code. |
| [`psi_star_z`](environmental-transmission.md#psi_star_z) | Per-patch upstream-MOSAIC parameter for the suitability curve; carried in the parameter set but NOT consumed by any laser-cholera simulation code. |
| [`theta_j`](environmental-transmission.md#theta_j) | Per-patch WASH (water/sanitation/hygiene) coverage fraction; (1 - theta_j) attenuates both env transmission to humans and infectious shedding into the reservoir. |
| [`zeta_1`](environmental-transmission.md#zeta_1) | Per-infectious-symptomatic-person shedding rate into the environmental reservoir W per tick (Poisson rate). |
| [`zeta_2`](environmental-transmission.md#zeta_2) | Per-infectious-asymptomatic-person shedding rate into the environmental reservoir W per tick (Poisson rate). |
| [`zeta_ratio`](environmental-transmission.md#zeta_ratio) | MOSAIC-side ratio used to derive zeta_1/zeta_2 upstream; carried into the parameter set but NOT consumed by any laser-cholera simulation code. |

## Alphabetical table

Every top-level field in `default_parameters.json`, sorted alphabetically. The "Group" column links to the section above; the parameter name links to its detailed entry.

| Parameter | Group | One-line description |
|---|---|---|
| [`E_j_initial`](initial-populations.md#e_j_initial) | Initial populations | Per-patch initial number of exposed (latent, not yet infectious) people seeded into the E compartment at tick 0. |
| [`I_j_initial`](initial-populations.md#i_j_initial) | Initial populations | Per-patch initial number of infectious people; split by sigma into symptomatic (Isym) and asymptomatic (Iasym) at tick 0. |
| [`N_j_initial`](initial-populations.md#n_j_initial) | Initial populations | Total initial population per patch (used as the gravity-model destination mass when reconstructing the mobility matrix, though the live code now sums the per-compartment initials instead). |
| [`R_j_initial`](initial-populations.md#r_j_initial) | Initial populations | Per-patch initial number of recovered (immune) people seeded into the R compartment at tick 0. |
| [`S_j_initial`](initial-populations.md#s_j_initial) | Initial populations | Per-patch initial number of susceptible people seeded into the S compartment at tick 0. |
| [`V1_j_initial`](initial-populations.md#v1_j_initial) | Initial populations | Per-patch initial number of one-dose-vaccinated people seeded into the V1 compartment at tick 0. |
| [`V2_j_initial`](initial-populations.md#v2_j_initial) | Initial populations | Per-patch initial number of two-dose-vaccinated people seeded into the V2 compartment at tick 0. |
| [`a_1_j`](human-transmission.md#a_1_j) | Human-to-human transmission | Per-patch amplitude of the first (annual) cosine term of the seasonal-transmission harmonic. |
| [`a_2_j`](human-transmission.md#a_2_j) | Human-to-human transmission | Per-patch amplitude of the second (semi-annual) cosine term of the seasonal-transmission harmonic. |
| [`alpha_1`](human-transmission.md#alpha_1) | Human-to-human transmission | Exponent on the effective-infected count in the human-to-human force-of-infection (mixing nonlinearity in the numerator). |
| [`alpha_2`](human-transmission.md#alpha_2) | Human-to-human transmission | Exponent on the patch population in the FOI denominator; alpha_2=1 = frequency-dependent mixing, alpha_2=0 = density-dependent. |
| [`b_1_j`](human-transmission.md#b_1_j) | Human-to-human transmission | Per-patch amplitude of the first (annual) sine term of the seasonal-transmission harmonic. |
| [`b_2_j`](human-transmission.md#b_2_j) | Human-to-human transmission | Per-patch amplitude of the second (semi-annual) sine term of the seasonal-transmission harmonic. |
| [`b_jt`](vital-dynamics.md#b_jt) | Vital dynamics | Per-patch, per-tick crude birth rate; the Poisson mean for new susceptibles each day is `N * b_jt[tick]`. |
| [`beta_j0_env`](environmental-transmission.md#beta_j0_env) | Environmental transmission | Per-patch baseline environmental-reservoir-to-human transmission rate; multiplier on the W/(kappa+W) saturation term in the env force of infection. |
| [`beta_j0_hum`](human-transmission.md#beta_j0_hum) | Human-to-human transmission | Per-patch baseline human-to-human transmission rate; multiplied by the seasonal harmonic to produce the per-tick, per-patch transmission envelope. |
| [`beta_j0_tot`](human-transmission.md#beta_j0_tot) | Human-to-human transmission | Per-patch combined baseline transmission rate (legacy / metadata field carried from upstream MOSAIC R config; not consumed by the Python simulation loop). |
| [`chi_endemic`](regime-switching.md#chi_endemic) | Regime switching | Reporting-rate divisor used when local symptomatic-infected fraction is below epidemic_threshold (endemic regime); inflates estimated cases via division. |
| [`chi_epidemic`](regime-switching.md#chi_epidemic) | Regime switching | Reporting-rate divisor used when local symptomatic-infected fraction is at or above epidemic_threshold (epidemic regime); inflates estimated cases via division. |
| [`d_jt`](vital-dynamics.md#d_jt) | Vital dynamics | Per-patch, per-tick non-disease (background) mortality rate applied to every live compartment (S, E, Isym, Iasym, R, V1, V2). |
| [`date_start`](run-identity.md#date_start) | Run identity and calendar | Calendar date of simulation tick 0; together with date_stop it defines the run length (nticks = (date_stop - date_start).days + 1). |
| [`date_stop`](run-identity.md#date_stop) | Run identity and calendar | Calendar date of the final simulation tick; with date_start it bounds the run window and sets nticks. |
| [`decay_days_long`](environmental-transmission.md#decay_days_long) | Environmental transmission | Longest reservoir half-life (in days) at the low end of the suitability-to-decay map; 1/decay_days_long is the minimum decay rate. |
| [`decay_days_short`](environmental-transmission.md#decay_days_short) | Environmental transmission | Shortest reservoir half-life (in days) at the high end of the suitability-to-decay map; 1/decay_days_short is the maximum decay rate. |
| [`decay_days_spread`](environmental-transmission.md#decay_days_spread) | Environmental transmission | MOSAIC-side scratch parameter describing the spread between fast and slow decay (days); carried in the parameter set but NOT consumed by any laser-cholera simulation code. |
| [`decay_shape_1`](environmental-transmission.md#decay_shape_1) | Environmental transmission | Alpha shape parameter of the Beta CDF used to map psi_jt in [0, 1] onto the [decay_days_short, decay_days_long] interval. |
| [`decay_shape_2`](environmental-transmission.md#decay_shape_2) | Environmental transmission | Beta shape parameter of the Beta CDF used to map psi_jt in [0, 1] onto the [decay_days_short, decay_days_long] interval. |
| [`delta_reporting_cases`](reporting.md#delta_reporting_cases) | Reporting | Integer tick lag between when a new symptomatic case occurs and when it is added to reported_cases. |
| [`delta_reporting_deaths`](reporting.md#delta_reporting_deaths) | Reporting | Integer tick lag between when a disease death occurs and when it is added to reported_deaths. |
| [`epidemic_peaks`](regime-switching.md#epidemic_peaks) | Regime switching | Observed epidemic peak dates per ISO code used by the likelihood scorer to evaluate peak-timing and peak-magnitude shape terms; not consumed by the simulation dynamics. |
| [`epidemic_threshold`](regime-switching.md#epidemic_threshold) | Regime switching | Per-patch (or global) symptomatic-infected fraction above which the patch is treated as in an epidemic regime, switching both the disease-mortality multiplier and the reporting-rate divisor. |
| [`epsilon`](disease-progression.md#epsilon) | Disease progression | Per-day rate at which natural immunity wanes (R -> S). |
| [`gamma_1`](disease-progression.md#gamma_1) | Disease progression | Per-day recovery rate for symptomatic infectious individuals (I_sym -> R). |
| [`gamma_2`](disease-progression.md#gamma_2) | Disease progression | Per-day recovery rate for asymptomatic infectious individuals (I_asym -> R). |
| [`iota`](disease-progression.md#iota) | Disease progression | Per-day rate at which exposed (E) individuals progress to becoming infectious (I_sym or I_asym). |
| [`kappa`](environmental-transmission.md#kappa) | Environmental transmission | Half-saturation constant in the env force-of-infection W / (kappa + W); larger kappa means more reservoir is needed before transmission saturates. |
| [`latitude`](geography-and-mobility.md#latitude) | Geography and mobility | Per-patch latitude coordinate; combined with longitude for inter-patch distance in the gravity-mobility kernel (and used to order patches in one plot). |
| [`location_name`](run-identity.md#location_name) | Run identity and calendar | Ordered list of patch/admin-unit labels; its length defines npatches and its order defines every per-patch vector's element ordering. |
| [`longitude`](geography-and-mobility.md#longitude) | Geography and mobility | Per-patch longitude coordinate; used together with latitude to compute great-circle distances that feed the gravity-model mobility matrix. |
| [`mobility_gamma`](geography-and-mobility.md#mobility_gamma) | Geography and mobility | Exponent on inter-patch distance (with sign reversed) in the gravity model; larger values penalize long-distance movement more strongly when building pi_ij. |
| [`mobility_omega`](geography-and-mobility.md#mobility_omega) | Geography and mobility | Exponent on the destination population in the gravity model that builds the patch-to-patch coupling matrix pi_ij; larger values pull more flow toward larger patches. |
| [`mu_j_baseline`](vital-dynamics.md#mu_j_baseline) | Vital dynamics | Per-patch baseline disease (cholera) mortality rate applied to symptomatic infectious; the multiplicative anchor of the `mu_jt = mu_baseline * (1 + slope*t) * (1 + epidemic_factor*flag)` formula. |
| [`mu_j_epidemic_factor`](vital-dynamics.md#mu_j_epidemic_factor) | Vital dynamics | Per-patch multiplicative boost on disease mortality when the patch is flagged as being in an epidemic regime (reported symptomatic infectious exceeds `epidemic_threshold * N`). |
| [`mu_j_slope`](vital-dynamics.md#mu_j_slope) | Vital dynamics | Per-patch linear time-trend on disease mortality; `mu_jt` is scaled by `(1 + mu_j_slope * tick/nticks)` so positive values ramp mortality up over the run and negative values ramp it down. |
| [`mu_jt`](vital-dynamics.md#mu_jt) | Vital dynamics | Disease mortality time-series stored on the parameter set for diagnostic plotting only — the simulation loop computes its own per-tick mu_jt from baseline/slope/epidemic_factor and ignores this array. |
| [`nu_1_jt`](vaccination.md#nu_1_jt) | Vaccination | Number of first-dose vaccinations to deliver per tick and per patch. |
| [`nu_2_jt`](vaccination.md#nu_2_jt) | Vaccination | Number of second-dose vaccinations to deliver per tick and per patch (moves V1 to V2). |
| [`nu_jt_sources`](vaccination.md#nu_jt_sources) | Vaccination | List of compartment names from which first-dose vaccinations are drawn (proportional sampling across the listed compartments). |
| [`omega_1`](vaccination.md#omega_1) | Vaccination | Daily waning rate from V1 back to S; converted to a per-tick probability via 1 - exp(-omega_1). |
| [`omega_2`](vaccination.md#omega_2) | Vaccination | Daily waning rate from V2 back to S; converted to a per-tick probability via 1 - exp(-omega_2). |
| [`p`](human-transmission.md#p) | Human-to-human transmission | Period (in ticks/days) of the seasonal-transmission harmonic; typically 365 for an annual cycle. |
| [`p_beta`](human-transmission.md#p_beta) | Human-to-human transmission | Per-patch split fraction between human-to-human and environmental transmission in the upstream R model (legacy metadata; not consumed by the Python simulation loop). |
| [`phi_1`](vaccination.md#phi_1) | Vaccination | Effectiveness (take fraction) of a first dose: only this fraction of delivered doses moves the recipient into V1. |
| [`phi_2`](vaccination.md#phi_2) | Vaccination | Effectiveness (take fraction) of a second dose: only this fraction of V1 recipients moves into V2. |
| [`prop_E_initial`](initial-populations.md#prop_e_initial) | Initial populations | Per-patch initial fraction of population in the E compartment (informational / upstream-MOSAIC artifact; not read by the model). |
| [`prop_I_initial`](initial-populations.md#prop_i_initial) | Initial populations | Per-patch initial fraction of population in the I compartment (informational / upstream-MOSAIC artifact; not read by the model). |
| [`prop_R_initial`](initial-populations.md#prop_r_initial) | Initial populations | Per-patch initial fraction of population in the R compartment (informational / upstream-MOSAIC artifact; not read by the model). |
| [`prop_S_initial`](initial-populations.md#prop_s_initial) | Initial populations | Per-patch initial fraction of population in the S compartment (informational / upstream-MOSAIC artifact; not read by the model itself). |
| [`prop_V1_initial`](initial-populations.md#prop_v1_initial) | Initial populations | Per-patch initial fraction of population in the V1 compartment (informational / upstream-MOSAIC artifact; not read by the model). |
| [`prop_V2_initial`](initial-populations.md#prop_v2_initial) | Initial populations | Per-patch initial fraction of population in the V2 compartment (informational / upstream-MOSAIC artifact; not read by the model). |
| [`psi_jt`](environmental-transmission.md#psi_jt) | Environmental transmission | Per-tick, per-patch environmental suitability time series; modulates seasonal env transmission rate around its mean and drives the suitability-to-decay map for the reservoir. |
| [`psi_star_a`](environmental-transmission.md#psi_star_a) | Environmental transmission | Per-patch upstream-MOSAIC parameter for the suitability curve; carried in the parameter set but NOT consumed by any laser-cholera simulation code. |
| [`psi_star_b`](environmental-transmission.md#psi_star_b) | Environmental transmission | Per-patch upstream-MOSAIC parameter for the suitability curve; carried in the parameter set but NOT consumed by any laser-cholera simulation code. |
| [`psi_star_k`](environmental-transmission.md#psi_star_k) | Environmental transmission | Per-patch upstream-MOSAIC parameter for the suitability curve; carried in the parameter set but NOT consumed by any laser-cholera simulation code. |
| [`psi_star_z`](environmental-transmission.md#psi_star_z) | Environmental transmission | Per-patch upstream-MOSAIC parameter for the suitability curve; carried in the parameter set but NOT consumed by any laser-cholera simulation code. |
| [`reported_cases`](reporting.md#reported_cases) | Reporting | Observed per-patch-per-time-step case counts used as the target series when computing the model likelihood. |
| [`reported_deaths`](reporting.md#reported_deaths) | Reporting | Observed per-patch-per-time-step death counts used as the target series when computing the model likelihood. |
| [`rho`](reporting.md#rho) | Reporting | Per-case reporting probability for symptomatic cases — fraction of new symptomatic infections that show up in the reported_cases series. |
| [`rho_deaths`](reporting.md#rho_deaths) | Reporting | Per-death reporting probability — fraction of disease deaths that show up in the reported_deaths series. |
| [`seed`](run-identity.md#seed) | Run identity and calendar | Integer seed for the model's pseudo-random number generator; setting it pins all stochastic draws to a reproducible sequence. |
| [`sigma`](disease-progression.md#sigma) | Disease progression | Fraction of new infections that become symptomatic (the rest go to the asymptomatic compartment). |
| [`tau_i`](geography-and-mobility.md#tau_i) | Geography and mobility | Per-patch fraction of infectious (and, for environmental exposure, susceptible) individuals that move out of their home patch; the master switch for spatial mixing. |
| [`theta_j`](environmental-transmission.md#theta_j) | Environmental transmission | Per-patch WASH (water/sanitation/hygiene) coverage fraction; (1 - theta_j) attenuates both env transmission to humans and infectious shedding into the reservoir. |
| [`zeta_1`](environmental-transmission.md#zeta_1) | Environmental transmission | Per-infectious-symptomatic-person shedding rate into the environmental reservoir W per tick (Poisson rate). |
| [`zeta_2`](environmental-transmission.md#zeta_2) | Environmental transmission | Per-infectious-asymptomatic-person shedding rate into the environmental reservoir W per tick (Poisson rate). |
| [`zeta_ratio`](environmental-transmission.md#zeta_ratio) | Environmental transmission | MOSAIC-side ratio used to derive zeta_1/zeta_2 upstream; carried into the parameter set but NOT consumed by any laser-cholera simulation code. |
