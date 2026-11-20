# shera_planet_search.py
# Functional, RVSearch-style multi-planet search for along-axis separation residuals
# Per-planet parameterization: {P, sesw, secw, tau_pa, mass}
# where sesw = sqrt(e) * sin(omega), secw = sqrt(e) * cos(omega), tau_pa in [0,1).

import numpy as np
from scipy.optimize import minimize
from orbitize.kepler import calc_orbit
import astropy.units as u
import astropy.constants as const
from orbitize.system import radec2seppa, seppa2radec
from astropy.timeseries import LombScargle
from copy import deepcopy

_PROGRESS_SUFFIX = '                        \r'

def _prog(msg, suppress=True):
    """Print a one-line progress update that overwrites itself."""
    if not suppress:
        print(msg, end=_PROGRESS_SUFFIX, flush=True)
    else:
        return

def _done(msg, suppress=True):
    """Finish a progress line with a final message and newline."""
    if not suppress:
        print(msg, flush=True)
    else:
        return


def out_of_bounds_planets(planets, minP=None, maxP=None):
    """
    Return True if ANY planet violates the prior bounds:
      - P outside range [minP, maxP]
      - e >= 0.95
      - mass <= 0
    """
    for p in planets:
        P = p["P"]; sesw = p["sesw"]; secw = p["secw"]; tau = p["tau_pa"]; mass = p["mass"]
        if (minP is not None and P < minP) or (maxP is not None and P > maxP):
            return True
        if sesw**2 + secw**2 >= 0.95:
            return True
        if mass <= 0.0:
            return True
        if not (0.0 <= tau < 1.0):
            return True
    return False


def neg_logprior(planets, m0, rayleigh_ecc_sigma=0.06, mass_prior_bounds=(0.01, 100.0), sma_prior_bounds=(0.1, 10.0)):
    """The MCMC's planet priors: log-uniform sma and mass, Rayleigh eccentricity."""
    out = 0.0
    for p in planets:
        m, P = p["mass"], p["P"]
        sma = ((m0) * P ** 2) ** (1 / 3)          # AU, as in to_orbitize_params
        e = p["sesw"] ** 2 + p["secw"] ** 2
        if not (sma_prior_bounds[0] <= sma <= sma_prior_bounds[1]):
            return np.inf
        if not (mass_prior_bounds[0] <= m <= mass_prior_bounds[1]):
            return np.inf
        # log-uniform sma and mass
        out += np.log(sma) + np.log(m)
        # the fit steps in P, not sma: Jacobian |d sma / d P| = (2/3) sma / P
        out -= np.log((2.0 / 3.0) * sma / P)
        # Rayleigh eccentricity; uniform in (sesw, secw) carries no Jacobian
        out += -np.log(max(e, 1e-300)) + e ** 2 / (2 * rayleigh_ecc_sigma ** 2)
    return out


def _ls_coefficients_at_peak(ls, t, y_resid, yerr, f_peak):
    """
    Return (C0, A_cos, B_sin) for model y ≈ C0 + A_cos*cos(ωt) + B_sin*sin(ωt).
    Uses ls.model_parameters if available; otherwise falls back to a weighted solve.
    """
    omega = 2.0 * np.pi * f_peak
    try:
        # Preferred (public) API in modern Astropy
        params = ls.model_parameters(f_peak)
        # Depending on fit options, params order is typically:
        # [offset] + [sin/cos terms]. For the standard floating-mean single-term model,
        # it’s (offset, A_cos, B_sin) or (offset, A_sin, B_cos) depending on basis.
        # We can detect via the design matrix columns:
        DM = ls.design_matrix(f_peak, t)
        # If first non-offset column matches cos(ωt), map accordingly:
        col1 = DM[:, 1]
        col2 = DM[:, 2] if DM.shape[1] > 2 else None
        cos_col = np.cos(omega * t)
        sin_col = np.sin(omega * t)

        C0 = float(params[0])
        if np.allclose(col1, cos_col, rtol=0, atol=1e-10):
            A_cos = float(params[1])
            B_sin = float(params[2])
        else:
            # columns are [offset, sin, cos]
            A_cos = float(params[2])
            B_sin = float(params[1])
        return C0, A_cos, B_sin
    except Exception:
        # Fallback: weighted linear solve using the same basis
        w = 1.0 / (yerr**2)
        Ccol = np.ones_like(t, dtype=float)
        Ccos = np.cos(omega * t)
        Ssin = np.sin(omega * t)
        X = np.column_stack([Ccol, Ccos, Ssin])
        XtW = (X.T * w)
        H = XtW @ X
        g = XtW @ y_resid
        C0, A_cos, B_sin = np.linalg.solve(H, g)
        return float(C0), float(A_cos), float(B_sin)


def ls_seed_and_fap(
    t, y, yerr, pa_axis, tau_ref_epoch, binary_params,
    minP, maxP, planets_existing=None, samples_per_peak=5, harmonic_summing=False
):
    """
    Astropy Lomb–Scargle *with floating mean* (fit_mean=True) used only for detection+seed.

    Returns:
      dict with keys:
        'P', 'tau_pa', 'mass', 'sesw', 'secw', 'fap',
        'freq', 'power', 'P_grid'
    """
    if planets_existing is None:
        planets_existing = []

    # Base model & residuals
    y_base = separation_model(t, planets_existing, pa_axis, tau_ref_epoch, binary_params) if planets_existing else np.zeros_like(t)
    y_resid = y - y_base

    # Frequency window corresponding to [minP, maxP] (convert years → days)
    Pmin_d = (float(minP) * u.yr).to(u.day).value
    Pmax_d = (float(maxP) * u.yr).to(u.day).value
    fmin = 1.0 / Pmax_d
    fmax = 1.0 / Pmin_d
    spp = int(max(1, samples_per_peak))

    # Floating mean ON (this is the whole point here)
    ls = LombScargle(t, y_resid, yerr, fit_mean=True, center_data=False)

    # Power spectrum and peak
    T_days = float(np.max(t) - np.min(t))
    freq = np.arange(fmin, fmax, 1.0 / (2.0 * np.pi * T_days))
    power = ls.power(freq)
    
    i_peak = int(np.argmax(power))
    f_peak = float(freq[i_peak])               # [1/day]
    P_peak = float(1.0 / f_peak / (1*u.yr).to(u.day).value)      # [yr]
    try:
        refined_f_peak, _ = fit_max_peak(freq, power, i_peak, plot=False)
    except:
        refined_f_peak = deepcopy(f_peak)
    harmonic_power = power[np.argmin(np.abs(freq - (2*refined_f_peak)))] if harmonic_summing else 0

    C0, A_cos, B_sin = _ls_coefficients_at_peak(ls, t, y_resid, yerr, f_peak)
    
    A_prime = B_sin      # coefficient of sin
    B_prime = A_cos      # coefficient of cos
    phi = float(np.arctan2(-B_prime, A_prime))
    omega = 2.0 * np.pi * f_peak

    tau_pa = wrap_tau_pa((phi - omega * float(tau_ref_epoch)) / (2.0 * np.pi))

    # Unit-mass circular response at (P_peak, tau_pa) on top of base; solve mass
    unit_shape = {"P": P_peak, "sesw": 0.0, "secw": 0.0, "tau_pa": tau_pa, "mass": 1.0}
    y_with1 = separation_model(t, planets_existing + [unit_shape], pa_axis, tau_ref_epoch, binary_params)
    y_unit = y_with1 - y_base
    
    mhat, _ = analytic_mass(y_resid, y_unit, yerr)
    if mhat < 0:
        mhat = -mhat
        unit_shape["tau_pa"] = wrap_tau_pa(unit_shape["tau_pa"] + 0.5)

    # FAP for the peak power (floating mean enables valid FAP)
    try:
        fap = float(ls.false_alarm_probability(
            power[i_peak] + harmonic_power, method="baluev",
            minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=spp
        ))
    except Exception:
        fap = float(ls.false_alarm_probability(
            power[i_peak] + harmonic_power,
            minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=spp
        ))

    # FAP guide levels for plotting (5%, 1%, 0.1%)
    fap_probs = [0.05, 0.01, 0.001]
    try:
        fap_levels = ls.false_alarm_level(
            fap_probs, minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=spp
        )
        fap_levels = np.asarray(fap_levels, dtype=float).ravel()
    except Exception:
        fap_levels = None
        fap_probs = None

    # Period grid corresponding to the returned power array (in years) — FORCE 1-D
    freq = np.asarray(freq, dtype=float).ravel()
    power = np.asarray(power, dtype=float).ravel()
    P_grid = (1.0 / freq) / (1*u.yr).to(u.day).value

    return {
        "P": P_peak,
        "tau_pa": float(unit_shape["tau_pa"]),
        "mass": float(mhat),
        "sesw": 0.0,
        "secw": 0.0,
        "fap": fap,
        "freq": freq,
        "power": power,
        "P_grid": P_grid,
        "fap_levels": fap_levels,
        "fap_probs": fap_probs,
        "harmonic_power": harmonic_power
    }


def local_period_subgrid(P_star, P_neighbors=None, width_frac=0.01, npts=9):
    """
    Small local period sub-grid around P_star (neighbors if available, else ±width_frac).
    """
    if P_neighbors is not None:
        left, right = P_neighbors
        if left > 0 and right > 0 and left < right:
            return np.linspace(left, right, npts)
    return np.linspace(P_star * (1 - width_frac), P_star * (1 + width_frac), npts)


def polish_fixed_period_subgrid(t, y, yerr, pa_axis, tau_ref_epoch,
                                seed_new_planet, subgrid, binary_params,
                                planets_existing=None,
                                minP=None, maxP=None,
                                method="L-BFGS-B"):
    """
    Polish near-fixed P: for each P_trial in subgrid, free {sesw, secw, tau_pa},
    set mass by linear scaling against the unit-mass model on top of the base,
    and score by BIC. Return the best params for the NEW planet.
    """
    if planets_existing is None:
        planets_existing = []
    best = None
    n = len(t)
    k_base = 5 * len(planets_existing)

    # Base model once
    y_base = separation_model(t, planets_existing, pa_axis, tau_ref_epoch, binary_params) if planets_existing else np.zeros_like(t)
    nll_base = 0.5 * np.sum(((y - y_base) / yerr)**2)
    bic_base = 2.0 * nll_base + k_base * np.log(n)

    Ns = len(subgrid)
    step = max(1, Ns // 100)
    _prog(f"Polish sub-grid near P*: 0% (N={Ns})")
    for j, P_trial in enumerate(subgrid, 1):
        x0 = np.array([seed_new_planet["sesw"], seed_new_planet["secw"], seed_new_planet["tau_pa"]], dtype=float)

        def objective(xvec):
            sesw, secw, tau = float(xvec[0]), float(xvec[1]), wrap_tau_pa(float(xvec[2]))
            if sesw**2 + secw**2 >= 1.0:
                return np.inf
            shape = {"P": float(P_trial), "sesw": sesw, "secw": secw, "tau_pa": tau, "mass": 1.0}
            y_with1 = separation_model(t, planets_existing + [shape], pa_axis, tau_ref_epoch, binary_params)
            y_unit = y_with1 - y_base
            # Linear mass (amplitude ratio)
            w = 1.0 / (yerr**2)
            num = np.sum(w * (y - y_base) * y_unit)
            den = np.sum(w * y_unit * y_unit)
            if not np.isfinite(den) or den <= 0:
                return np.inf
            mhat = num / den
            # enforce positive mass via τ shift by +0.5 (equivalent signal)
            if mhat < 0:
                mhat = -mhat
                tau = wrap_tau_pa(tau + 0.5)
                shape["tau_pa"] = tau
                y_with1 = separation_model(t, planets_existing + [shape], pa_axis, tau_ref_epoch, binary_params)
                y_unit = y_with1 - y_base
            y_model = y_base + mhat * y_unit
            return 0.5 * np.sum(((y - y_model) / yerr)**2)

        res = minimize(objective, x0, method=method, options=dict(maxiter=200))
        sesw_opt, secw_opt, tau_opt = float(res.x[0]), float(res.x[1]), wrap_tau_pa(float(res.x[2]))
        if sesw_opt**2 + secw_opt**2 >= 1.0:
            if (j % step == 0) or (j == Ns):
                pct = int(100 * j / Ns)
                _prog(f"Polish sub-grid near P*: {pct}% (N={Ns})")
            continue

        shape = {"P": float(P_trial), "sesw": sesw_opt, "secw": secw_opt, "tau_pa": tau_opt, "mass": 1.0}
        y_with1 = separation_model(t, planets_existing + [shape], pa_axis, tau_ref_epoch, binary_params)
        y_unit = y_with1 - y_base
        w = 1.0 / (yerr**2)
        num = np.sum(w * (y - y_base) * y_unit)
        den = np.sum(w * y_unit * y_unit)
        if not np.isfinite(den) or den <= 0:
            if (j % step == 0) or (j == Ns):
                pct = int(100 * j / Ns)
                _prog(f"Polish sub-grid near P*: {pct}% (N={Ns})")
            continue
        mhat = num / den
        if mhat < 0:
            mhat = -mhat
            shape["tau_pa"] = wrap_tau_pa(shape["tau_pa"] + 0.5)
        shape["mass"] = float(mhat)

        y_model = y_base + shape["mass"] * y_unit
        nll = 0.5 * np.sum(((y - y_model) / yerr)**2)
        bic_cand = 2.0 * nll + (k_base + 5) * np.log(n)  # +5 new free params in the final model

        if (best is None) or (bic_cand < best["bic"]):
            best = {"params": shape, "bic": float(bic_cand), "P_trial": float(P_trial)}

        if (j % step == 0) or (j == Ns):
            pct = int(100 * j / Ns)
            _prog(f"Polish sub-grid near P*: {pct}% (N={Ns})")

    if best is not None:
        _done(f"Polish sub-grid near P*: 100% — best P={best['P_trial']:.6f} yr")
        return best["params"]
    _done("Polish sub-grid near P*: 100% — no valid candidate")
    return best  # may be None


def pack_params(planets):
    """
    Flatten planets into vector: [P1, sesw1, secw1, tau1, m1,  P2, ...].
    """
    vec = []
    for p in planets:
        vec += [p["P"], p["sesw"], p["secw"], wrap_tau_pa(p["tau_pa"]), max(p["mass"], 1e-12)]
    return np.array(vec, dtype=float)


def unpack_params(x):
    """
    Inverse of pack_params().
    """
    x = np.asarray(x, dtype=float)
    assert x.size % 5 == 0, "Param vector length must be multiple of 5."
    out = []
    for i in range(0, x.size, 5):
        out.append({
            "P": float(x[i+0]),
            "sesw": float(x[i+1]),
            "secw": float(x[i+2]),
            "tau_pa": wrap_tau_pa(float(x[i+3])),
            "mass": max(float(x[i+4]), 1e-12)
        })
    return out


def joint_map_optimize(t, y, yerr, pa_axis, tau_ref_epoch, planets_init, binary_params,
                       minP=None, maxP=None, method="L-BFGS-B"):
    """
    Final joint MAP: free all planets' {P, sesw, secw, tau_pa, mass}.
    Returns (planets_best, final_bic).
    """
    x0 = pack_params(planets_init)
    n = len(t)
    _prog(f"MAP optimization ({method}): starting (k={x0.size} params)")

    it_counter = [0]
    def _cb(_xk):
        it_counter[0] += 1
        _prog(f"MAP optimization ({method}): iter {it_counter[0]}")

    def objective(xvec):
        planets = unpack_params(xvec)
        if out_of_bounds_planets(planets, minP=minP, maxP=maxP):
            return np.inf
        y_model = separation_model(t, planets, pa_axis, tau_ref_epoch, binary_params)
        return 0.5 * np.sum(((y - y_model) / yerr)**2) + neg_logprior(planets, binary_params['M0'])

    res = minimize(objective, x0, method=method, callback=_cb, options=dict(maxiter=1000))
    planets_best = unpack_params(res.x)
    y_model = separation_model(t, planets_best, pa_axis, tau_ref_epoch, binary_params)
    nll = 0.5 * np.sum(((y - y_model) / yerr)**2)
    bic = 2.0 * nll + len(res.x) * np.log(n)
    _done(f"MAP optimization ({method}): done (nit={getattr(res,'nit',None)}, nfev={getattr(res,'nfev',None)}, BIC={bic:.3f})")
    return planets_best, bic


def search_planets(t, y, yerr, pa_axis,
                   minP, maxP, tau_ref_epoch, binary_params,
                   fap=1e-3, oversampling=1.0,
                   max_planets=5,
                   polish_width_frac=0.01, polish_npts=9, plot=False,
                   harmonic_summing=False):
    """
    baseline(model of accepted planets) → Astropy LS (floating mean) detection
    → FAP gate → polish(local P sub-grid) → joint MAP; iterate.

    Plots:
        After each LS scan: LS periodogram with FAP guide lines (5%, 1%, 0.1%).
        After each acceptance: dense model vs data across full span.
        At end: final LS periodogram from the last scan and final dense model.
    """
    planets = []
    diags = {"periodograms": [], "thresholds": [], "accepted": []}
    faps = []

    # Keep the last LS periodogram (and FAP levels) to show at the very end
    last_ls_periods = None
    last_ls_power   = None
    last_ls_levels  = None
    last_ls_probs   = None

    _done(f"Search pipeline: max_planets={max_planets}, oversampling={oversampling}, fap={fap:g}")
    for loop_idx in range(max_planets):
        idx = len(planets) + 1
        _done(f"\n[Planet {idx}]")

        # Lomb–Scargle detection & seed (floating mean)
        _prog(f"[Planet {idx}] LS scan (Astropy)...")
        ls_out = ls_seed_and_fap(
            t, y, yerr, pa_axis, tau_ref_epoch, binary_params,
            minP, maxP, planets_existing=planets, samples_per_peak=max(5, int(oversampling*5)),
            harmonic_summing=harmonic_summing
        )
        _done(f"[Planet {idx}] LS scan complete: P*={ls_out['P']:.6f} yr, FAP={ls_out['fap']:.3g}")

        # Save diagnostics and plot the LS periodogram (with FAP guide lines)
        diags["periodograms"].append((ls_out["P_grid"], ls_out["power"]))
        diags["thresholds"].append(fap)
        faps.append(ls_out['fap'])

        # Store for end-of-run final plot
        last_ls_periods = ls_out["P_grid"]
        last_ls_power   = ls_out["power"]
        last_ls_levels  = ls_out.get("fap_levels")
        last_ls_probs   = ls_out.get("fap_probs")
        if plot:
            try:
                ls_periodogram_plot(
                    ls_out["P_grid"], ls_out["power"],
                    fap_levels=ls_out.get("fap_levels"),
                    probs=ls_out.get("fap_probs"),
                    title=f"LS Periodogram — Planet {idx}"
                )
            except Exception as _plot_err:
                _done(f"[Planet {idx}] LS periodogram plotting skipped: {_plot_err}")

        # FAP gate
        if (not np.isfinite(ls_out["fap"])) or (ls_out["fap"] > fap):
            _done(f"[Planet {idx}] Peak fails FAP (FAP={ls_out['fap']:.3g} > {fap:g}). Stopping.")
            break
        _done(f"[Planet {idx}] Peak passes FAP (FAP={ls_out['fap']:.3g} ≤ {fap:g}). Promoting.")
        
        # Promote detection → build polish seed from LS outputs
        seed = {"P": ls_out["P"], "tau_pa": ls_out["tau_pa"], "mass": ls_out["mass"], "sesw": 0.0, "secw": 0.0}
        P_star = float(ls_out["P"])
        subgrid = local_period_subgrid(P_star, P_neighbors=None, width_frac=polish_width_frac, npts=polish_npts)
        if subgrid[-1] > maxP:
            subgrid -= (subgrid[-1] - (maxP-0.08))
        elif subgrid[0] < minP:
            subgrid += ((minP + 0.08) - subgrid[0])
        _done(f"[Planet {idx}] Polish sub-grid size: {len(subgrid)} around P*={P_star:.6f} yr")

        # Polish near P*: free {sesw,secw,tau_pa}, mass via linear scaling on top of base
        polished = polish_fixed_period_subgrid(
            t, y, yerr, pa_axis, tau_ref_epoch,
            seed, subgrid, binary_params,
            planets_existing=planets, minP=minP, maxP=maxP
        )
        
        # Final joint MAP with ALL planets free
        planets_plus = planets + [polished]
        planets, bic_map = joint_map_optimize(
            t, y, yerr, pa_axis, tau_ref_epoch, planets_plus, binary_params, minP=minP, maxP=maxP
        )
        
        diags["accepted"].append(planets[-1])
        pacc = planets[-1]
        ecc_now = pacc['sesw']**2 + pacc['secw']**2
        _done(f"[Planet {idx}] Accepted after MAP: P={pacc['P']:.6f} yr, mass={pacc['mass']:.6g}, tau_pa={pacc['tau_pa']:.6f}, e={ecc_now:.6f}, BIC={bic_map:.3f}")
        if plot:
            # Dense-only model plot after acceptance
            try:
                model_fit_plot_dense(
                    t, y, yerr,
                    separation_model, planets, pa_axis, tau_ref_epoch, binary_params,
                    npts=1500,
                    title=f"Planet {idx}: Model vs Data (dense curve)"
                )
            except Exception as _plot_err:
                _done(f"[Planet {idx}] Dense model plotting skipped: {_plot_err}")

    # Final BIC of the accepted model
    n = len(t)
    y_final = separation_model(t, planets, pa_axis, tau_ref_epoch, binary_params) if planets else np.zeros_like(t)
    nll_final = 0.5 * np.sum(((y - y_final) / yerr)**2)
    bic_final = 2.0 * nll_final + 5 * len(planets) * np.log(n)

    _done(f"\nSearch complete: accepted {len(planets)} planet(s); final BIC={bic_final:.3f}")

    # Final summary print (notebook-style)
    print("\n=== Accepted planets ===")
    if planets:
        for i, p in enumerate(planets, 1):
            e = p['sesw']**2 + p['secw']**2
            print(f"  #{i}: P={p['P']:.6f} yr  mass={p['mass']:.6g}  tau_pa={p['tau_pa']:.6f}  e={e:.6f}")
    else:
        print("  none")
    if plot:
        # Final plots: last LS periodogram (likely the failed detection) and dense final model
        try:
            if (last_ls_periods is not None) and (last_ls_power is not None):
                ls_periodogram_plot(
                    last_ls_periods, last_ls_power,
                    fap_levels=last_ls_levels, probs=last_ls_probs,
                    title="Final LS Periodogram (last scan)"
                )
        except Exception as _plot_err:
            _done(f"[Final] LS periodogram plotting skipped: {_plot_err}")
    
        try:
            model_fit_plot_dense(
                t, y, yerr,
                separation_model, planets, pa_axis, tau_ref_epoch, binary_params,
                npts=2000,
                title="Final Model vs Data (all accepted planets, dense curve)"
            )
        except Exception as _plot_err:
            _done(f"[Final] Dense model plotting skipped: {_plot_err}")

    return {"planets": planets, "bic": bic_final, "diagnostics": diags, 'faps': faps}


def to_orbitize_params(planets, binary_params, tau_ref_epoch, PA):
    """
    Map this module's params to orbitize parameters
    """
    param_arr = {}
    
    sma1_as = binary_params['sma_as1']
    plx = binary_params['plx']
    sma1 = sma1_as * 1000 / plx
    param_arr['sma1'] = sma1

    # Convert binary period to secondary mass (plus the mass of planetary companions)
    per1 = binary_params['P1']
    mtot = sma1 ** 3 / per1 ** 2
    m0 = binary_params['M0']
    m1 = mtot - m0

    # Convert binary epp to tau
    epp1 = binary_params['epp1']
    tau1 = tp_to_tau(epp1, tau_ref_epoch, per1)
    param_arr['tau1'] = tau1
    param_arr['m0'] = m0
    param_arr['plx'] = plx
    param_arr['inc1'] = binary_params['inc1']
    param_arr['pan1'] = binary_params['pan1']
    param_arr['aop1'] = binary_params['aop1']
    param_arr['ecc1'] = binary_params['ecc1']

    # Convert taupa to tau for planetary companions
    for body in range(len(planets)):
        body_num = body + 2

        # Convert coplanar params
        param_arr['inc' + str(body_num)] = binary_params['inc1']
        param_arr['pan' + str(body_num)] = binary_params['pan1']

        period = planets[body]['P']
        mass = (planets[body]['mass'] * u.Mearth).to(u.Msun).value
        sma = ((mass+m0)*period**2)**(1/3)
        param_arr['sma' + str(body_num)] = sma
        taupa = planets[body]['tau_pa']
        t_PA = tau_to_tp(taupa, tau_ref_epoch, period)
        inc = binary_params['inc1']
        pan = binary_params['pan1']
        ecc, aop = sesw_secw_to_eomega(planets[body]['sesw'], planets[body]['secw'])
        param_arr['ecc' + str(body_num)] = ecc
        param_arr['aop' + str(body_num)] = aop
        cx = (np.cos(inc) * np.sin(aop) * np.cos(pan)) + (np.cos(aop) * np.sin(pan))
        sx = (np.cos(inc) * np.cos(aop) * np.cos(pan)) - (np.sin(aop) * np.sin(pan))
        cy = (np.cos(aop) * np.cos(pan)) - (np.cos(inc) * np.sin(aop) * np.sin(pan))
        sy = -(np.sin(aop) * np.cos(pan)) - (np.cos(inc) * np.cos(aop) * np.sin(pan))
        cyp = cy * np.tan(PA)
        syp = sy * np.tan(PA)
        nu_PA = np.arctan((cyp - cx) / (sx - syp))
        pa_sign_match = np.sin(PA) if np.abs(np.sin(PA)) > np.abs(np.cos(PA)) else np.cos(PA)
        nu_sign_match = cx * np.cos(nu_PA) + sx * np.sin(nu_PA) if np.abs(np.sin(PA)) > np.abs(
            np.cos(PA)) else cy * np.cos(nu_PA) + sy * np.sin(nu_PA)
        if np.sign(pa_sign_match) != np.sign(nu_sign_match):
            nu_PA = (nu_PA + np.pi) % (2 * np.pi)
        eanom = 2 * np.arctan2(np.sqrt((1 - ecc) / (1 + ecc)) * np.sin(nu_PA / 2), np.cos(nu_PA / 2))
        manom = eanom - ecc * np.sin(eanom)
        tp = t_PA - (period * u.yr).to(u.day).value * manom / 2 / np.pi
        tau = tp_to_tau(tp, tau_ref_epoch, period)
        param_arr['tau' + str(body_num)] = tau
        param_arr['m' + str(body_num)] = mass

        # Subtract planetary companion mass from converted m1
        m1 -= mass

    # Set secondary mass in param_arr
    param_arr['m1'] = m1

    return param_arr


def compute_model_via_orbitize(t, pa_axis, tau_ref_epoch, params_arr, num_secondary_bodies):
    """
    Compute along-axis separation residuals using orbitize routines without
    instantiating orbitize classes (if possible).
    """
    epochs = t
    n_epochs = len(epochs)
    m0 = params_arr['m0']
    
    ra_kepler = np.zeros((n_epochs, num_secondary_bodies + 1, 1))
    dec_kepler = np.zeros((n_epochs, num_secondary_bodies + 1, 1))
    
    for body_num in np.arange(num_secondary_bodies)+1:
        sma = params_arr['sma{}'.format(body_num)]
        ecc = params_arr['ecc{}'.format(body_num)]
        inc = params_arr['inc{}'.format(body_num)]
        argp = params_arr['aop{}'.format(body_num)]
        lan = params_arr['pan{}'.format(body_num)]
        tau = params_arr['tau{}'.format(body_num)]
        plx = params_arr['plx']
        
        mass = params_arr['m{}'.format(body_num)]

        all_smas = np.array([params_arr['sma{}'.format(i)] for i in range(num_secondary_bodies+1)[1:]])
        within_orbit = np.where(all_smas <= sma)
        outside_orbit = np.where(all_smas > sma)
        all_pl_masses = np.array([params_arr['m{}'.format(i)] for i in range(num_secondary_bodies+1)[1:]])
        inside_masses = all_pl_masses[within_orbit]
        mtot = np.sum(inside_masses) + m0
        
        raoff, decoff, vz_i = calc_orbit(
            epochs, sma, ecc, inc, argp, lan, tau, plx, mtot,
            mass_for_Kamp=m0, tau_ref_epoch=tau_ref_epoch
        )
        ra_kepler[:, body_num, :] = np.reshape(raoff, (n_epochs, 1))
        dec_kepler[:, body_num, :] = np.reshape(decoff, (n_epochs, 1))
    
    sep, pa = radec2seppa(ra_kepler, dec_kepler)
    total_mass = 0
    model = np.zeros(n_epochs)
    for body_num in np.arange(num_secondary_bodies + 1)[2:]:
        m_planet = params_arr['m{}'.format(body_num)]
        total_mass += m_planet
        model += (sep[:, body_num, :] * m_planet * np.cos(((pa[:, body_num, :] - pa[:, 1,:]) * np.pi / 180))).reshape(model.shape)

    total_mass += m0
    model /= total_mass
    
    return model


def separation_model(t, planets, pa_axis, tau_ref_epoch, binary_params):
    """
    Thin wrapper so objectives can call a single function for the model.
    """
    params_arr = to_orbitize_params(planets, binary_params, tau_ref_epoch, pa_axis)
    return compute_model_via_orbitize(t, pa_axis, tau_ref_epoch, params_arr, len(planets)+1)


def model_fit_plot_dense(
    t, y, yerr, model_fn, planets, pa_axis, tau_ref_epoch, binary_params,
    npts=1000, title=None
):
    """
    Plot data with errors + a *continuous* model curve over the full time span.
    Also shows residuals at the *data* times.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    t = np.asarray(t)
    tmin, tmax = float(np.min(t)), float(np.max(t))
    t_dense = np.linspace(tmin, tmax, int(npts))

    # model at dense times (smooth curve) and at data points (for residuals)
    y_model_dense = model_fn(t_dense, planets, pa_axis, tau_ref_epoch, binary_params)
    y_model_points = model_fn(t,        planets, pa_axis, tau_ref_epoch, binary_params)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    axs[0].errorbar(t, y*1000, yerr=yerr*1000, fmt='.', alpha=0.85, label='data', color='red')
    axs[0].plot(t_dense, y_model_dense*1000, '-', lw=1.5, label='model', color='black')
    axs[0].set_ylabel('Δsep (μas)', size=13)
    axs[0].legend(loc='best')
    axs[0].set_title(title if title else 'Model vs Data')

    res = y - y_model_points
    axs[1].axhline(0, color='k', lw=0.8, alpha=0.6)
    axs[1].errorbar(t, res*1000, yerr=yerr*1000, fmt='.', alpha=0.85, color='red')
    axs[1].set_xlabel('Epoch (MJD)', size=13)
    axs[1].set_ylabel('Residuals', size=11)

    plt.tight_layout()
    plt.show()


def ls_periodogram_plot(periods, power, fap_levels=None, probs=None, title=None,
                        ax=None, save_path=None):
    """
    Astropy Lomb–Scargle power vs period (log-x) with optional FAP guide lines.
    Styled with the SHERA color palette.
 
    Parameters
    ----------
    periods, power : array-like
        Period grid (yr) and LS power, same length.
    fap_levels : array-like, optional
        Power thresholds corresponding to `probs` (e.g. from
        LombScargle.false_alarm_level).
    probs : list of float, optional
        FAP probabilities matching `fap_levels`. Defaults to [0.05, 0.01, 0.001].
    title : str, optional
        Plot title. Default: 'Lomb–Scargle Periodogram'.
    ax : matplotlib Axes, optional
        Draw into an existing axes (useful for multi-panel composites).
        If None, a new figure is created.
    save_path : str, optional
        If given, save the figure to this path (only when a new figure is made).
    """
    import matplotlib.pyplot as plt
 
    # palette
    SHERA_POWER   = '#d331a5'   # magenta — LS power trace
    SHERA_FAP     = ['#2e2e6b', '#4a4aa0', '#8a8ad6']   # dark → light purples
    SHERA_GRID    = '#cccccc'
    SHERA_TEXT    = '#2e2e6b'
 
    # Font sizes (kept large to match the existing composite-panel layout)
    TEXT_SCALE  = 2.32     # ← bump this to enlarge all text
    LABEL_SIZE  = 18 * TEXT_SCALE
    TICK_SIZE   = 18 * TEXT_SCALE
    LEGEND_SIZE = 10.5 * TEXT_SCALE
    TITLE_SIZE  = 16 * TEXT_SCALE
 
    periods = np.asarray(periods, dtype=float).ravel()
    power   = np.asarray(power,   dtype=float).ravel()
    if periods.size != power.size:
        raise ValueError(
            f"ls_periodogram_plot: length mismatch: "
            f"periods={periods.shape}, power={power.shape}"
        )
 
    # Set up axes
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 7))
    else:
        fig = ax.figure
 
    # LS power curve
    ax.plot(periods, power, lw=1.6, color=SHERA_POWER, label='LS Power', zorder=3)
 
    # FAP guide lines 
    if fap_levels is not None:
        fap_levels = np.asarray(fap_levels, dtype=float).ravel()
        if probs is None:
            probs = [0.05, 0.01, 0.001]
        probs = list(probs)
        n = min(len(fap_levels), len(probs))
        for i in range(n):
            lvl = float(fap_levels[i])
            p   = float(probs[i])
            ax.axhline(
                lvl, ls='--', lw=1.3,
                color=SHERA_FAP[i % len(SHERA_FAP)],
                alpha=0.95, zorder=2,
                label=f'FAP = {100 * p:.1f}%',
            )
 
    ax.set_xscale('log')
    ax.set_xlabel('Period (yr)', size=LABEL_SIZE, color=SHERA_TEXT)
    ax.set_ylabel('Periodogram Power', size=LABEL_SIZE, color=SHERA_TEXT)
    ax.set_xticks([1, 10])
    ax.set_xticklabels(['1', '10'])
 
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, colors=SHERA_TEXT)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_SIZE, colors=SHERA_TEXT)
    ax.grid(True, which='major', color=SHERA_GRID, alpha=0.5, lw=0.6, zorder=0)
    ax.grid(True, which='minor', color=SHERA_GRID, alpha=0.25, lw=0.4, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(SHERA_TEXT)
        spine.set_linewidth(1.0)
 
    ax.legend(loc='best', fontsize=LEGEND_SIZE, framealpha=0.9,
              edgecolor=SHERA_TEXT)
 
    if standalone:
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
 
    return ax


def fit_max_peak(freq, power, init_i, plot=False):
    """
    Refine a periodogram peak by fitting a parabola to 3 points
    around init_i and returning the vertex.
    """

    if init_i <= 0 or init_i >= len(freq) - 1:
        raise ValueError("init_i must not be at the edges")

    # walk left
    i_left = init_i
    while i_left > 0 and power[i_left] > power[init_i]/2:
        i_left -= 1

    # walk right
    i_right = init_i
    while i_right < len(power) - 1 and power[i_right] > power[init_i]/2:
        i_right += 1

    x = freq[i_left:i_right + 1]
    y = power[i_left:i_right + 1]

    # fit y = ax^2 + bx + c
    a, b, c = np.polyfit(x, y, 2)

    if a >= 0:
        raise RuntimeError("Parabola is not concave down")

    # vertex of parabola
    peak_x = -b / (2 * a)
    peak_y = a * peak_x**2 + b * peak_x + c

    if plot:
        plt.figure()
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = a * x_fit**2 + b * x_fit + c
        
        plt.plot(freq, power, "ko", label="data")
        plt.plot(x_fit, y_fit, "r-", label="parabola")
        plt.legend()
        plt.show()

    return peak_x, peak_y


def wrap_tau_pa(x):
    """
    Wrap tau_pa into [0,1).
    """
    tau = float(x) % 1.0
    return tau if tau >= 0 else tau + 1.0


def sesw_secw_to_eomega(sesw, secw):
    """
    Convert (sesw, secw) with sesw=√e sin ω, secw=√e cos ω  → (e, ω in [0,2π)).
    """
    e = sesw**2 + secw**2
    omega = np.arctan2(sesw, secw)
    if omega < 0:
        omega += 2.0 * np.pi
    return float(e), float(omega)


def eomega_to_sesw_secw(e, omega):
    """
    Convert (e, ω) → (sesw, secw) with sesw=√e sin ω, secw=√e cos ω.
    """
    e = float(np.clip(e, 0.0, 1.0 - 1e-12))
    sqrt_e = np.sqrt(e)
    sesw = sqrt_e * np.sin(omega)
    secw = sqrt_e * np.cos(omega)
    return float(sesw), float(secw)


def analytic_mass(y, y_unit, yerr):
    """
    Weighted least-squares scalar 'mass' in y ≈ mass * y_unit,
    using the SAME conventions as neg_loglike:
      - NO constant term in the returned NLL
      - NO clamping of yerr
      - Raises if yerr is non-positive/non-finite or if the problem is degenerate.
    """
    y = np.asarray(y)
    y_unit = np.asarray(y_unit)
    yerr = np.asarray(yerr)

    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(y_unit)) or not np.all(np.isfinite(yerr)):
        raise ValueError("y, y_unit, and yerr must be finite.")
    if np.any(yerr <= 0):
        raise ValueError("yerr must be strictly positive for all data points.")
    if y.shape != y_unit.shape or y.shape != yerr.shape:
        raise ValueError("y, y_unit, and yerr must have the same shape.")

    w = 1.0 / (yerr**2)
    denom = np.sum(w * (y_unit**2))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("Degenerate unit model under current weighting (denominator ≤ 0).")

    num = np.sum(w * y * y_unit)
    mhat = num / denom

    resid = (y - mhat * y_unit) / yerr
    nll = 0.5 * np.sum(resid**2)
    return float(mhat), float(nll)

    
def tp_to_tau(tp, ref_epoch, period):
    """
    Convert t_p to tau
    """
    tau = (tp - ref_epoch)/(period * u.year.to(u.day))
    tau %= 1
    return tau


def tau_to_tp(tau, ref_epoch, period, after_date=None):
    """
    Convert tau to t_p (first periastron on/after after_date if provided).
    """
    period_days = period * u.year.to(u.day)
    tp = tau * (period_days) + ref_epoch
    if after_date is not None:
        num_periods = (after_date - tp)/period_days
        num_periods = np.ceil(num_periods).astype(int)
        tp += num_periods * period_days
    return tp
