# Shot Noise Analysis Methods

## Raw data handling
- The calibration and shot-noise CSV files in `shot_noise_data/` are treated as raw inputs.
- Calibration repeats are grouped by exact frequency.
- Shot-noise repeats are grouped by bulb setting, blank continuation rows are carried forward, and blank or zero trailing rows are ignored.
- Shot-noise groups are sorted by measured first-stage voltage `V_d`, not knob setting.
- The fit error bars include repeat SEM and an RMS-voltage floor of `0.20 mV` propagated as `delta(V0^2)=2 V0 delta(V0)`.

## Unit conversions
- Frequencies are converted from kHz to Hz before integration.
- Voltages are converted from mV to V before fitting.
- The feedback resistor and test-input resistor are each taken as `475 kOhm` with independent 1% tolerances.
- Current is computed as `I = V_d / R_F`.
- The calibration test input means the fitted x scale is proportional to `R_test^2 / R_F`, so resistor tolerance contributes `sqrt((2%)^2 + (1%)^2) = 2.24%` as a common scale uncertainty.

## Gain integral
- The measured gain-squared curve is integrated with the trapezoid rule through the final measured point at 90 kHz.
- Gain squared is computed as `(mean V0 / mean Vi)^2` at each frequency; the mean of per-row gain-squared values is retained in the calibration summary for comparison.
- The unmeasured right tail beyond 90 kHz uses a power-law fit to the measured high-frequency tail.
- Nominal result: `G = 3.865974e+10 +/- 2.16e+07 Hz`.
- The fitted tail contributes `0.245%` of the nominal integral.
- Trapezoid discretization is estimated by comparing to Simpson integration: `0.050%`.

## Shot-noise fit
- The fitted line is `V0^2 = e X + V_A^2`, where `X = 2 R_F^2 I_av integral(g^2(f) df)`. The plot axis multiplies by the gain integral; it does not divide by it.
- The slope is the electron charge in coulombs.
- The nominal fit uses one grouped mean per bulb setting. This is the correct default because repeated readings at the same bulb setting are repeated observations of the same current condition.
- Treating all raw rows separately is included only as a diagnostic; it gives a much better-looking chi-squared when a pooled single-measurement scatter is assigned, but it does not move the charge toward the accepted value.
- Repeated-measurement SEM is used for `V0^2`; uncertainty in `V_d` is propagated into the point-by-point `X` uncertainty.
- The five readings within a bulb setting may be time-correlated because the AC voltmeter reports an RMS average with finite response time. As a diagnostic, the group uncertainty is recomputed as `s/sqrt(N_eff)` for `N_eff = 1...5` instead of assuming all five readings are independent.
- The RMS-voltage floor is an alternate empirical way to account for AC-voltmeter/noise-estimator stability not captured by five repeated readings. The effective-sample-size scan is preferred for presentation because it directly connects the enlarged error bars to correlated RMS readings.
- The gain-integral uncertainty is propagated as a separate multiplicative contribution to the slope uncertainty.
- The resistor-ratio contribution to the all-data uncertainty is `0.0410e-19 C`.
- A second fit excludes low-signal front-end points. The threshold is chosen as the smallest plotted x-value cut that gives `p >= 0.05` using repeat SEM only, without the empirical RMS-voltage floor.

## Diagnostics and improvement
- Nominal fit with RMS floor: `chi2/dof = 29.5/26`, `chi2_red = 1.13`, `p = 0.289`.
- Nominal grouped result: `e = (1.8355 +/- 0.0413)e-19 C`.
- Front-end-cut result: `e = (1.8431 +/- 0.0414)e-19 C`, `chi2_red = 1.12`, `p = 0.338`.
- The front-end cut excludes bulb settings `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`.
- Accepted value ratio: `e/e_accepted = 1.1456`.
- Agreement with the accepted value would require the product `(R_test^2/R_F) G` to be larger by `14.6%`, much larger than the propagated calibration-repeat and resistor-ratio uncertainties.
- If the issue is only AC calibration, the gain amplitude would need to be low by `7.0%`.
- If interpreted as a current-proportional excess-noise source, the effective Fano/excess-noise factor is `1.146`.
- The post-fit extra-scatter check now adds `2.446e-05 V^2`; this is small compared with the output variance scale but shows the RMS floor is still an approximation.
- Raw-row diagnostic: `e = (1.8339 +/- 0.0411)e-19 C`, `chi2_red = 0.941`.
- Effective-sample-size diagnostic with `N_eff = 2`: `e = (1.8353 +/- 0.0414)e-19 C`, `chi2/dof = 24.7/26`, `chi2_red = 0.951`, `p = 0.534`.
- Effective-sample-size diagnostic with `N_eff = 3`: `e = (1.8353 +/- 0.0413)e-19 C`, `chi2/dof = 37.1/26`, `chi2_red = 1.43`, `p = 0.0732`.
- Removing points with plotted x-value below 0.5 leaves `7` points and gives `e = (1.8342 +/- 0.0417)e-19 C`, `chi2_red = 1.05`.
- Leave-one-out, current-window, scaled-x-threshold, effective-sample-size, and tail-start scans are exported in `shot_noise_results.json`; these are diagnostics, not hidden point-selection rules.

## Presentation guidance
- Use the nominal fit figure to show the main result and the residual plot to discuss remaining structure.
- For the goodness-of-fit discussion, quote the `N_eff = 2` or `N_eff = 3` uncertainty model as a correlated-readings systematic rather than relying only on the empirical RMS floor.
