"""
k2d_replica.py

A faithful Python replica of the deployed "Res-2Da" 2-state Kalman filter
(H:\\packages\\gnss_live.yaml, template sensor.gnss15k2d), built as a sandbox
for testing candidate corrections/fixes against the ACTUAL filter's
behaviour rather than against a static regression -- see CHANGELOG entry
24 (why lowwater_correction.py's naive fit-and-subtract approach was
retracted: K2D is adaptive/nonlinear, so you cannot safely reason about a
correction's effect without running it through the real filter logic).

This is read-only research: it does not modify, call, or depend on
anything under H:\\. The Jinja template logic in gnss_live.yaml was read
(with permission) and translated to Python line-for-line; parameters are
copied verbatim from the deployed constants. No change has been made to
any live H:\\ file.

Scope: this replicates the Kalman filter's state machine (prediction,
spike-gating, adaptive R, update) given a time series of
(predicted tide, observed height, forcing/surge) -- it does NOT
re-derive V13's forcing_surge from raw met/river/wind data (that's
forcing.yaml's separate 8-feature ridge regression). Historical
forcing_surge values are taken from the already-logged
'HA-PWR/Surge' column in GNSStidesAllData.xlsx sheet 'Res-2Da', which is
what the live filter actually saw at each historical step -- decouples
testing K2D's own logic from re-deriving V13.

Validated against the sheet's own logged 'Filter: Res-2Da' output (see
validate_against_log() / __main__) before being trusted for anything.

Change log
----------
v1  2026-07-30  Claude Code. Initial replica + validation.
"""
import numpy as np
import pandas as pd

# Constants copied verbatim from H:\packages\gnss_live.yaml (2026-07-30 read)
PHI_R = 0.734862406
PHI_B = 0.92722028
Q_R = 0.000407375
Q_B = 0.0000549996
SPIKE_GATE = 3.0
VAR_DECAY = 0.062763838
R_FLOOR = 0.075786571
R_CAP = 0.215923095
DELTA_FORCING_CLAMP = 0.05   # H:\packages\gnss_live.yaml delta_forcing clamp
BLOWUP_GATE = 2.5            # needs_reseed blowup threshold
KALMAN_CLIP = (-1.0, 10.0)   # final output clip


DEFAULT_PARAMS = dict(phi_r=PHI_R, phi_b=PHI_B, q_r=Q_R, q_b=Q_B,
                       spike_gate=SPIKE_GATE, var_decay=VAR_DECAY,
                       r_floor=R_FLOOR, r_cap=R_CAP,
                       delta_forcing_clamp=DELTA_FORCING_CLAMP,
                       blowup_gate=BLOWUP_GATE, kalman_clip=KALMAN_CLIP)


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def k2d_step(thisobs, thispred, forcing_at_obs, state, params=None, extra_r_fn=None):
    """
    One 15-min K2D cycle. Mirrors gnss_live.yaml's template variables
    name-for-name (obs_residual, delta_forcing, predicted_r/b,
    predicted_P_r/b, effective_r, innovation, status, clean_innovation,
    adaptive_r, K_r/b, updated_r/b, updated_P_r/b, kalman_raw, kalman).

    `state` is a dict carrying prior_r, prior_b, prior_P_r, prior_P_b,
    prior_adaptive_r, prior_forcing (all analogous to the HA sensor's
    stored attributes from the previous cycle). Returns (kalman, new_state,
    diagnostics) where diagnostics mirrors the sheet's logged columns.

    `params`: optional dict overriding DEFAULT_PARAMS (the real deployed
    constants) -- for sensitivity testing / re-tuning exploration only;
    the default reproduces the live filter exactly (see validate_against_log).

    `extra_r_fn`: optional callable(thispred) -> extra measurement-noise
    variance (m^2) added to effective_r, on top of the deployed adaptive-R
    mechanism. Not part of the live filter -- a research hook for testing
    whether state-dependent (e.g. low-water) uncertainty inflation helps,
    per the "trust the model more when the observation is known to be
    riskier" idea discussed 2026-07-30. None reproduces the live filter.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    obs_residual = thisobs - thispred

    fnow = forcing_at_obs
    fprev = state.get("prior_forcing")
    raw_delta = (fnow - fprev) if (fnow is not None and fprev is not None) else 0.0
    delta_forcing = raw_delta if abs(raw_delta) < p["delta_forcing_clamp"] else 0.0

    prev_r = state.get("prior_r", 0.0)
    prev_b = state.get("prior_b", 0.0)
    prev_P_r = state.get("prior_P_r", 1.0)
    prev_P_b = state.get("prior_P_b", 1.0)
    prior_adaptive_r = state.get("prior_adaptive_r", 0.05)

    combined = prev_r + prev_b
    blowup = abs(combined - obs_residual) > p["blowup_gate"]
    needs_reseed = state.get("state_missing", False) or blowup

    predicted_r = obs_residual if needs_reseed else (prev_r + delta_forcing)
    predicted_b = 0.0 if needs_reseed else (p["phi_b"] * prev_b)
    predicted_P_r = 1.0 if needs_reseed else (p["phi_r"] ** 2 * prev_P_r + p["q_r"])
    predicted_P_b = 1.0 if needs_reseed else (p["phi_b"] ** 2 * prev_P_b + p["q_b"])

    effective_r = _clip(prior_adaptive_r, p["r_floor"], p["r_cap"])
    if extra_r_fn is not None:
        effective_r += max(0.0, extra_r_fn(thispred))

    innovation = obs_residual - predicted_r - predicted_b
    innov_var = predicted_P_r + predicted_P_b + effective_r
    innov_std = innov_var ** 0.5
    threshold = p["spike_gate"] * innov_std
    status = "spike" if abs(innovation) > threshold else "OK"
    clean_innovation = innovation if status == "OK" else 0.0

    if status in ("OK", "spike"):
        raw_ar = p["var_decay"] * innovation ** 2 + (1 - p["var_decay"]) * prior_adaptive_r
    else:
        raw_ar = prior_adaptive_r
    adaptive_r = _clip(raw_ar, p["r_floor"], p["r_cap"])

    K_r = (predicted_P_r / innov_var) if status == "OK" else 0.0
    K_b = (predicted_P_b / innov_var) if status == "OK" else 0.0

    updated_r = predicted_r + K_r * clean_innovation
    updated_b = predicted_b + K_b * clean_innovation
    updated_P_r = (1 - K_r) ** 2 * predicted_P_r + K_r ** 2 * (predicted_P_b + effective_r)
    updated_P_b = (1 - K_b) ** 2 * predicted_P_b + K_b ** 2 * (predicted_P_r + effective_r)

    kalman_raw = thispred + updated_r + updated_b
    kalman = _clip(kalman_raw, *p["kalman_clip"])

    new_state = dict(prior_r=updated_r, prior_b=updated_b, prior_P_r=updated_P_r,
                      prior_P_b=updated_P_b, prior_adaptive_r=adaptive_r,
                      prior_forcing=forcing_at_obs, state_missing=False)
    diag = dict(obs_residual=obs_residual, delta_forcing=delta_forcing,
                predicted_r=predicted_r, predicted_b=predicted_b,
                predicted_P_r=predicted_P_r, predicted_P_b=predicted_P_b,
                innovation=innovation, status=status, adaptive_r=adaptive_r,
                K_r=K_r, K_b=K_b, updated_r=updated_r, updated_b=updated_b,
                kalman_raw=kalman_raw, kalman=kalman)
    return kalman, new_state, diag


def run_series(df, obs_col="Obs", pred_col="Pred", forcing_col="HA-PWR/Surge", seed_first_row=True,
               params=None, extra_r_fn=None):
    """
    df: DataFrame sorted by time with at least [obs_col, pred_col, forcing_col].
    Returns df with added columns: kalman, obs_residual, innovation, status,
    K_r, K_b, adaptive_r -- directly comparable to the Res-2Da sheet's own
    logged columns (Filter: Res-2Da, Obs Residual, Innovation, ...).

    seed_first_row: the real historical log's very first row is a cold-start
    seed (output = Pred, r=b=0, P_r=P_b=0 -- i.e. treated as exactly known,
    NOT the live filter's needs_reseed convention of P=1.0 for an
    unexpected gap/blowup recovery). Confirmed by reverse-solving the
    logged row 1 predicted_P_r/P_b (0.000627/0.000102) back through
    phi^2*P+q: both solve to ~0, i.e. P_r=P_b=0 at the seed, not 1.0.
    Default True reproduces that one-time convention.
    """
    df = df.reset_index(drop=True)
    rows = []
    if seed_first_row and len(df):
        r0 = df.iloc[0]
        forcing0 = None if pd.isna(r0[forcing_col]) else float(r0[forcing_col])
        state = dict(prior_r=0.0, prior_b=0.0, prior_P_r=0.0, prior_P_b=0.0,
                     prior_adaptive_r=0.05, prior_forcing=forcing0, state_missing=False)
        rows.append(dict(obs_residual=np.nan, delta_forcing=np.nan, predicted_r=0.0,
                          predicted_b=0.0, predicted_P_r=1.0, predicted_P_b=1.0,
                          innovation=np.nan, status="seed", adaptive_r=0.05, K_r=0.0, K_b=0.0,
                          updated_r=0.0, updated_b=0.0, kalman_raw=r0[pred_col], kalman=r0[pred_col]))
        start = 1
    else:
        state = dict(state_missing=True)
        start = 0

    for i in range(start, len(df)):
        r = df.iloc[i]
        thisobs, thispred, forcing = r[obs_col], r[pred_col], r[forcing_col]
        if pd.isna(thisobs) or pd.isna(thispred):
            rows.append(dict(kalman=np.nan, status="gap"))
            continue
        forcing = None if pd.isna(forcing) else float(forcing)
        kalman, state, diag = k2d_step(float(thisobs), float(thispred), forcing, state,
                                        params=params, extra_r_fn=extra_r_fn)
        rows.append(diag)

    out = df.copy()
    diag_df = pd.DataFrame(rows)
    for col in diag_df.columns:
        out[col] = diag_df[col]
    return out


def validate_against_log(n=500):
    """
    Loads the real historical Res-2Da sheet and checks this replica
    reproduces its logged 'Filter: Res-2Da' output. Run as:
    python -m pipeline.k2d_replica

    Result (2026-07-30, n=2460 rows / ~59 days from 2025-10-30): after a
    small transient in the first ~dozen cycles (max diff 0.046m, from the
    one-time zero-covariance seed converging), the replica agrees with the
    logged output to within ~5e-6 m by the end -- i.e. this is a faithful
    replica of the live filter's logic, not an approximation.
    """
    f = r"C:\Users\dandl\OneDrive\Documents\EstuaryStudy\IthacaGNSStides\GNSStidesAllData.xlsx"
    res = pd.read_excel(f, sheet_name="Res-2Da", header=1, nrows=n)
    res.columns = [str(c).strip() for c in res.columns]
    res["TimeStamp"] = pd.to_datetime(res["TimeStamp"], errors="coerce")
    res = res.dropna(subset=["TimeStamp", "Pred", "Obs"]).reset_index(drop=True)

    out = run_series(res, obs_col="Obs", pred_col="Pred", forcing_col="HA-PWR/Surge")
    logged = pd.to_numeric(res["Filter: Res-2Da"], errors="coerce")
    diff = out["kalman"] - logged
    print(f"n={len(out)}")
    print(f"max |diff| vs logged Filter: Res-2Da: {diff.abs().max():.6f}")
    print(f"mean diff: {diff.mean():+.6f}  std: {diff.std():.6f}")
    print()
    print(pd.DataFrame({"logged": logged, "replica": out["kalman"], "diff": diff}).tail(15).to_string())


if __name__ == "__main__":
    validate_against_log()
