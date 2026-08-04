"""
geometry.py

Satellite position / azimuth-elevation geometry for APP1.

This is the orbit-propagation and az/el code proven correct in v26.ipynb
(the local RINEX-level notebook lineage) -- carried over essentially
unchanged, since the bugs found in that lineage were in the surge sign
convention and consensus logic (see arcfit.py), not in this geometry layer.

Change log
----------
v1  2026-07-29  Claude Code. Extracted and lightly cleaned from v26.ipynb
    for the from-scratch pipeline (briefing SS9 / plan `unified-growing-pony`).
"""
import numpy as np
import pandas as pd
import pymap3d as pm

MU = 3.986005e14
OMEGA_E = 7.2921151467e-5

SIG_PRIORITY = {
    "G": ["S1C", "S1", "C1", "S2W", "S2", "S2L", "S5Q", "S5"],
    "E": ["S1C", "S1", "S5Q", "S5", "S7Q", "S7", "S8Q", "S8"],
    "C": ["S2I", "S2", "S6I", "S6", "S7I", "S7", "S1I", "S1"],
    "R": ["S1C", "S1", "C1", "C1C"],
}

LAMBDA_MAP = {
    "S1C": 0.19029, "S1": 0.19029, "C1": 0.19029, "S1P": 0.18700,
    "S2W": 0.24421, "S2": 0.24421, "S2L": 0.24421, "P2": 0.24421,
    "S5Q": 0.25483, "S5": 0.25483,
    "S2I": 0.19200, "S6I": 0.23631,
    "S7Q": 0.24800, "S7I": 0.24800, "S8Q": 0.24800,
    "S7": 0.24800, "S8": 0.24800,
    "S1I": 0.19029,
}


def get_val(eph, names):
    """Return the first of `names` present in the ephemeris dataset."""
    for n in names:
        if n in eph:
            return float(eph[n])
    raise KeyError(str(names))


def propagate_glonass(state0, toe, tobs):
    """Linear extrapolation of GLONASS broadcast position/velocity -- fine for the few
    minutes between broadcast epochs used here, per the existing local implementation."""
    dt = (tobs - toe).total_seconds()
    x, y, z, vx, vy, vz = state0
    return np.array([x + vx * dt, y + vy * dt, z + vz * dt], float)


def get_sat_pos(eph_near, t, const):
    """Keplerian (GPS/Galileo/BeiDou) or linear (GLONASS) satellite ECEF position at time t."""
    try:
        if const == "R":
            X = get_val(eph_near, ["X", "x", "Xposition"])
            Y = get_val(eph_near, ["Y", "y", "Yposition"])
            Z = get_val(eph_near, ["Z", "z", "Zposition"])
            dX = get_val(eph_near, ["dX", "dx", "DX", "VelX"])
            dY = get_val(eph_near, ["dY", "dy", "DY", "VelY"])
            dZ = get_val(eph_near, ["dZ", "dz", "DZ", "VelZ"])
            state0 = np.array([X, Y, Z, dX, dY, dZ], float) * 1000.0
            toe = pd.Timestamp(eph_near["_epoch_time"])
            return propagate_glonass(state0, toe, t)

        ecc = get_val(eph_near, ["Eccentricity", "ecc", "e"])
        sqA = get_val(eph_near, ["sqrtA", "SqrtA"])
        if sqA <= 0:
            return None
        M0 = get_val(eph_near, ["M0"])
        w = get_val(eph_near, ["omega", "argumentofperigee"])
        Om0 = get_val(eph_near, ["Omega0", "OMG0", "OMEGA0"])
        OmDot = get_val(eph_near, ["OmegaDot", "OMGd", "OMEGADOT"])
        i0 = get_val(eph_near, ["Io", "i0", "I0"])

        toe = pd.Timestamp(eph_near["_epoch_time"]) - pd.Timestamp("1980-01-06")
        toe = toe.total_seconds() % 604800
        tsec = (t - pd.Timestamp("1980-01-06")).total_seconds() % 604800
        tk = tsec - toe
        if tk > 302400:
            tk -= 604800
        if tk < -302400:
            tk += 604800

        n0 = np.sqrt(MU / (sqA ** 6))
        M = M0 + n0 * tk
        E = M
        for _ in range(5):
            E = M + ecc * np.sin(E)

        v = np.arctan2(np.sqrt(1 - ecc ** 2) * np.sin(E), np.cos(E) - ecc)
        u = v + w
        r = (sqA ** 2) * (1 - ecc * np.cos(E))

        omega = Om0 + (OmDot - OMEGA_E) * tk - OMEGA_E * toe

        xp = r * np.cos(u)
        yp = r * np.sin(u)

        x = xp * np.cos(omega) - yp * np.cos(i0) * np.sin(omega)
        y = xp * np.sin(omega) + yp * np.cos(i0) * np.cos(omega)
        z = yp * np.sin(i0)
        return np.array([x, y, z], float)
    except Exception:
        return None


RADIO_REFRACTIVITY_SCALE = 1.15
"""
Follow-up scaling on the Bennett (1982) bending-angle formula below, applied
in az_el_series(). After the base correction fixed the dominant part of the
bias (surge dropped from a wildly-scattered 1.0-2.3m down to a tight
+0.40 to +0.54m cluster across 6 satellites/3 constellations), a residual
+0.25-0.35m remained -- too consistent across independent satellites to be
noise, and too small to be a further TOTALANTH problem (a TOTALANTH=30.676
test, per V25.ipynb's old recalibration, OVERSHOT to -0.19 to -0.33m when
combined with the base refraction fix -- confirming V25's old adjustment was
itself compensating for this same missing refraction correction, not a real
antenna-height error; TOTALANTH stays at 31.406).

Bennett's formula is derived from OPTICAL astronomical refraction. Radio
refractivity in the troposphere at L-band is commonly measured ~10-15%
larger than the optical value (both are dominated by the same dry/
hydrostatic term, but the wet-term and L-band-specific contributions differ)
-- this 1.15 scale factor is a literature-informed estimate consistent with
that figure, tested once against the real 2026-01-27 arcs and found to bring
the same 6 arcs to a much tighter +0.24 to +0.37m cluster, close to the
+0.10 to +0.27m NOC_Obs-implied range for that window.

This is NOT a properly calibrated value -- it's one data point, not a fit
against the slipway ground truth across many dates/conditions. Recalibrating
it (or replacing it with a real radio-refractivity model rather than a
scalar fudge on an optical formula) belongs in Phase 1 validation, not here.
"""


def refraction_angle_correction(el_deg, scale=RADIO_REFRACTIVITY_SCALE):
    """
    Bennett (1982) atmospheric refraction BENDING-ANGLE formula, in degrees,
    scaled by `scale` (see RADIO_REFRACTIVITY_SCALE above).

    This is a genuinely different physical quantity from get_tropo_delay() above:
    that function is a zenith path-DELAY mapping function (meters of excess range,
    used for pseudorange corrections). This one is the bending of the ray path
    itself near the horizon -- the angle by which the apparent (as-received)
    elevation of the satellite differs from the geometric (orbit-computed, vacuum
    straight-line) elevation. GNSS-IR's SNR interference model needs the APPARENT
    elevation (the real local angle of arrival of the ray hitting the antenna and
    the reflecting surface), not the geometric one, since that is what actually
    sets the direct/reflected path geometry.

    At the low elevations used here (sectors span el 0.5-8 deg) this bending is
    large: roughly 0.1-0.4 deg across that range (compare to the ~0.0002-0.0005 deg
    elevation error from omitted broadcast-ephemeris harmonic terms, which is
    negligible by comparison) -- large enough to be the dominant remaining source
    of the ~1-2m systematic reflector-height bias found in this debugging session
    (see CHANGELOG). Root-caused by: (1) fitting a synthetic SNR signal with a
    KNOWN injected surge through fit_arc() using the real (uncorrected) az/el/
    predlevel for a real arc -- fit_arc recovered the injected value to within
    0.002-0.015 m every time, proving the LSP fit itself is unbiased; (2)
    comparing orbit propagation with vs without the missing broadcast-ephemeris
    harmonic correction terms (Cuc/Cus/Crc/Crs/Cic/Cis/DeltaN/IDOT) and finding
    the resulting elevation-angle difference is only ~0.0002-0.0005 deg -- too
    small by ~2 orders of magnitude to explain the bias; (3) applying this
    refraction-angle correction to the real E03 arc and confirming the surge
    result moves from +1.19 m toward the expected small-residual range (see
    CHANGELOG for the exact before/after numbers). The existing get_tropo_delay()
    correction (already disabled -- see comment in fit_arc()) is NOT a
    substitute for this: it is dimensionally a range-delay mapping function
    misapplied as an elevation correction, not this angular bending effect.

    Reference: G.G. Bennett, "The Calculation of Astronomical Refraction in
    Marine Navigation", Journal of Navigation 35(2), 1982. Radio refraction in
    the troposphere at L-band is close in magnitude to optical refraction
    (both dominated by the same dry/hydrostatic term); this is the standard
    formula used for exactly this purpose in low-elevation GNSS-IR work
    (the "elevation angle correction" referenced in Williams & Nievinski 2017).
    """
    el = np.asarray(el_deg, float)
    R_arcmin = 1.02 / np.tan(np.radians(el + 10.3 / (el + 5.11)))
    return (R_arcmin / 60.0) * scale


def get_tropo_delay(el_deg):
    """Bennett (1982) tropospheric delay mapping, as used in the existing local code (V17 onward).

    NOT the more advanced MPF (Williams & Nievinski 2017) or NITE (Feng et al. 2023) models that
    gnssrefl also documents -- those are left as a follow-up (briefing SS9), not implemented here,
    to keep this pass scoped to the sign/arc-length fixes.
    """
    el = np.radians(el_deg)
    sinel = np.sin(el)
    zdry, zwet = 2.3, 0.1
    ad, bd, cd = 0.00127683, 0.00291536, 0.062610505
    mdry = 1 + ad / (1 + bd * sinel + cd)
    aw, bw, cw = 0.000580218, 0.00142752, 0.0434729
    mwet = 1 + aw / (1 + bw * sinel + cw)
    return zdry * mdry + zwet * mwet


def _extract_ephemeris_records(eph):
    """
    Pull each distinct broadcast ephemeris epoch out of the xarray slice into a
    plain Python dict once, so nearest-epoch lookup during the coarse-time loop
    below can be done in pandas/numpy instead of repeated xarray .sel() calls.

    Broadcast ephemerides update every ~1-2 hours (a handful of records per
    satellite per day) -- xarray's per-call indexing overhead (index building,
    broadcasting checks, etc.) turned out to dominate az_el_series' runtime
    (profiled: ~26s of a ~32s/61-satellite/2-hour run was xarray .sel()
    machinery, only ~4s was the actual Kepler-equation solving in get_sat_pos).
    Converting to plain dicts up front and doing nearest-time lookup with
    pandas.Index.get_indexer (fully vectorized) removes that overhead.
    """
    eph_times = pd.DatetimeIndex(eph.time.values)
    records = []
    for i in range(len(eph_times)):
        rec = eph.isel(time=i)
        d = {k: rec[k].values for k in rec.data_vars}
        d["_epoch_time"] = eph_times[i]
        records.append(d)
    return eph_times, records


def az_el_series(nav, sv, const, rx_lat, rx_lon, rx_alt, times, coarse_step_s=30,
                 apply_refraction=True):
    """
    Azimuth/elevation for satellite `sv` at every timestamp in `times`.

    Orbit propagation is only evaluated at a coarse cadence (coarse_step_s) and
    cubic-spline-interpolated to the full sample rate -- az/el vary smoothly over
    the tens of seconds between samples, so this avoids ~solving Kepler's equation
    at every 2-second SNR epoch (43200/day) for every visible satellite, which is
    the actual cost driver, not the LSP fit itself. Same efficiency idea as
    gnssrefl/Purnell's own elv_interp.py.

    `apply_refraction=False` returns the GEOMETRIC elevation with no bending
    correction -- used by pipeline/extraction_cache.py so cached extractions
    can have alternative refraction models applied downstream without
    re-parsing RINEX. Default True preserves the original behaviour.

    Returns (az_deg, el_deg) arrays aligned with `times`, or (None, None) if the
    satellite has no ephemeris covering this window.
    """
    eph = nav.sel(sv=sv).dropna(dim="time", how="all")
    if len(eph.time) == 0:
        return None, None

    eph_times, eph_records = _extract_ephemeris_records(eph)

    t0, t1 = times[0], times[-1]
    coarse_times = pd.date_range(t0, t1, freq=f"{coarse_step_s}s")
    if len(coarse_times) < 2:
        coarse_times = pd.DatetimeIndex([t0, t1])

    nearest_idx = eph_times.get_indexer(coarse_times, method="nearest")

    az_c, el_c, tc_ok = [], [], []
    for t, ei in zip(coarse_times, nearest_idx):
        if ei < 0:
            continue
        pos = get_sat_pos(eph_records[ei], t, const)
        if pos is None:
            continue
        az, el, _ = pm.ecef2aer(pos[0], pos[1], pos[2], rx_lat, rx_lon, rx_alt, deg=True)
        az_c.append(float(az))
        el_c.append(float(el))
        tc_ok.append(t)

    if len(tc_ok) < 2:
        return None, None

    tc_sec = np.array([(t - tc_ok[0]).total_seconds() for t in tc_ok])
    t_sec = np.array([(t - tc_ok[0]).total_seconds() for t in times])

    # unwrap azimuth before interpolating so it doesn't jump across the 0/360 seam
    az_unwrapped = np.degrees(np.unwrap(np.radians(az_c)))
    az_interp = np.interp(t_sec, tc_sec, az_unwrapped) % 360.0
    el_interp = np.interp(t_sec, tc_sec, el_c)

    # Apparent (as-received) elevation = geometric elevation + atmospheric
    # refraction bending. See refraction_angle_correction() docstring -- this
    # is the confirmed root cause of the ~1-2m systematic surge bias found in
    # this debugging session, not a speculative addition.
    if apply_refraction:
        el_interp = el_interp + refraction_angle_correction(el_interp)

    return az_interp, el_interp
