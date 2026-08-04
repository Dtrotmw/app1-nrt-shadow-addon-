"""
tropo_rh.py

Tropospheric reflector-height correction for GNSS-IR, following
Santamaria-Gomez & Watson (2017, GPS Solutions 21:451-459) as implemented
in Purnell's gnssir-rt (github.com/purnelldj/gnssir-rt, tropd.py) -- the
same correction Purnell et al. (2024, GRL) apply in their real-time
GNSS-IR water-level system. Meteorology comes from the GPT2w 1-degree
climatology grid (Boehm et al. 2015), via the gpt_1wA.pickle grid file
shipped with gnssir-rt (copied alongside this module) -- no live weather
feed needed.

Physics: the GNSS-IR reflector height is biased by (a) atmospheric
BENDING (the apparent elevation of the satellite is higher than the
geometric one, growing sharply below ~10 deg) and (b) the non-unit
REFRACTIVE INDEX along the extra reflected path. Santamaria-Gomez &
Watson derive a multiplicative correction factor rh_fac(elv_min,
elv_max, P, T, e) such that

    RH_corrected = RH * (1 + rh_fac)

per arc, where elv_min/elv_max are the arc's elevation limits and P/T/e
are pressure (hPa), temperature (C), water-vapour pressure (hPa) at the
antenna. The correction is designed to be applied to heights retrieved
using GEOMETRIC (unrefracted) elevation angles -- i.e., it REPLACES the
elevation-side Bennett x1.15 bending correction in pipeline/geometry.py,
it does not stack on top of it. (The x1.15 scale was always flagged as
"one data point, not calibrated" -- this module is the principled
replacement candidate, tested against it, not silently swapped in.)

At APP1's geometry (RH 24-31m, elevations 0.5-8 deg -- unusually low)
this correction is at its largest and most consequential: a fac of a few
tenths of a percent moves RH by several centimetres, comparable to the
residual low-water bias being chased.

Change log
----------
v1  2026-08-03  Claude Code (Fable). Ported from gnssir-rt tropd.py
    (bend_eqn / N_eqn / corr_rh_facs verbatim math, restructured; GPT2w
    reader used via refl_code/scratch/make_gpt_reference.py copy of
    make_gpt.py). Site file built for APP1 lat/lon.
"""
import os
import sys

import numpy as np
import pandas as pd
import pymap3d as pm

PIPE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPE_DIR)  # make_gpt_reference.py lives alongside this file
GPT_INIT_FILE = os.path.join(PIPE_DIR, "gpt_1wA.pickle")
GPT_SITE_FILE = os.path.join(PIPE_DIR, "gpt_app1.pickle")

RX_XYZ = (3997184.446, -293144.113, 4939523.568)
_lat, _lon, _alt = pm.ecef2geodetic(*RX_XYZ)
# The RINEX-header ECEF gives a geodetic height of -4217m -- it's a coarse
# APPROX POSITION, harmless for satellite az/el (which is all the pipeline
# ever used it for: a few km of position error barely moves the direction to
# a satellite 20,000km away) but catastrophic for GPT's height-based
# pressure/temperature reduction (caught 2026-08-03: it produced P=1709hPa,
# T=36C in a Devon winter). Use the physically-derived ellipsoidal height
# instead: antenna 31.406m above Chart Datum, CD ~4m below ODN at Appledore,
# geoid undulation +53.65m here (from GPT's own undu output) -> ~81m.
# Met sensitivity is tiny (pressure scale height 8.4km -> 10m height error
# = 0.1% pressure error), so the CD/ODN offset's exact value doesn't matter.
SITE_ELL_HEIGHT = 81.0
SITE_LLA = (float(_lat), float(_lon), SITE_ELL_HEIGHT)


# ---------------------------------------------------------------------------
# Santamaria-Gomez & Watson (2017) correction math (per gnssir-rt tropd.py)
# ---------------------------------------------------------------------------

def bend_eqn(p, t, elv):
    """Atmospheric bending (deg) at elevation elv (deg) for pressure p (hPa),
    temperature t (C). Bennett-form with real met inputs."""
    arc_min = 510 / (9 / 5 * t + 492) * p / 1010.16 / np.tan(np.deg2rad(elv + 7.31 / (elv + 4.4)))
    return arc_min / 60.0


def N_eqn(p, t, e):
    """Atmospheric refractivity (dimensionless, ~3e-4) at the antenna."""
    k1, k2, k3 = 77.604, 70.4, 373900.0
    pd_ = p - e
    tk = t + 273.15
    return (k1 * pd_ / tk + k2 * e / tk + k3 * e / tk ** 2) / 1e6


def corr_rh_facs(elv_min, elv_max, p, t, e):
    """Multiplicative RH correction factor(s) for arcs spanning geometric
    elevations [elv_min, elv_max] (deg). RH_corr = RH * (1 + fac)."""
    elv_min = np.asarray(elv_min, float)
    elv_max = np.asarray(elv_max, float)
    elv_min_corr = bend_eqn(p, t, elv_min)
    elv_max_corr = bend_eqn(p, t, elv_max)
    meanbendelv = (elv_min + elv_min_corr + elv_max + elv_max_corr) / 2.0

    elv_min_rad = np.deg2rad(elv_min)
    elv_max_rad = np.deg2rad(elv_max)
    elv_min_corr_rad = np.deg2rad(elv_min_corr)
    elv_max_corr_rad = np.deg2rad(elv_max_corr)
    meanbendcorrrad = (elv_min_corr_rad + elv_max_corr_rad) / 2.0
    meanrad = (elv_min_rad + elv_max_rad) / 2.0

    Nant = N_eqn(p, t, e)

    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(elv_max_rad != elv_min_rad,
                       (elv_max_corr_rad - elv_min_corr_rad) / (elv_max_rad - elv_min_rad),
                       0.0)

    der = xi
    e_ = meanrad
    edash = np.deg2rad(meanbendelv)
    de = meanbendcorrrad
    rh_fac1 = Nant / (np.sin(edash) ** 2) * (np.cos(edash) / np.cos(e_)) * (1 + der)
    rh_fac2 = -der + (np.sin(de) * np.tan(e_) + 1 - np.cos(de)) * (1 + der)
    return rh_fac1 + rh_fac2


# ---------------------------------------------------------------------------
# GPT2w meteorology at the site (climatology -- no live feed needed)
# ---------------------------------------------------------------------------

_gpt_ready = False


def _ensure_site_file():
    global _gpt_ready
    if _gpt_ready:
        return
    from make_gpt_reference import makegptfile  # ported gnssir-rt make_gpt.py
    if not os.path.exists(GPT_SITE_FILE):
        makegptfile(GPT_SITE_FILE, GPT_INIT_FILE, SITE_LLA[0], SITE_LLA[1])
    _gpt_ready = True


def site_met(when):
    """(pressure hPa, temperature C, water-vapour pressure hPa) at APP1 at
    datetime `when`, from the GPT2w climatology (annual+semiannual harmonics)."""
    _ensure_site_file()
    from make_gpt_reference import gpt2_1w
    ts = pd.Timestamp(when)
    dmjd = (ts - pd.Timestamp("1858-11-17")) / pd.Timedelta(days=1)  # modified Julian date
    # NOTE: gpt2_1w wants lat/lon in RADIANS (makegptfile wants DEGREES --
    # verified against the reference source; passing degrees here produced
    # physically impossible met values, caught by sanity-checking the output).
    p, t, _dt, _tm, e, *_ = gpt2_1w(GPT_SITE_FILE, dmjd,
                                     np.deg2rad(SITE_LLA[0]), np.deg2rad(SITE_LLA[1]),
                                     SITE_LLA[2], 0)
    return float(np.atleast_1d(p)[0]), float(np.atleast_1d(t)[0]), float(np.atleast_1d(e)[0])


def rh_correction_factor(when, elv_min, elv_max):
    """rh_fac for an arc at datetime `when` spanning geometric elevations
    [elv_min, elv_max] deg. RH_corr = RH * (1 + rh_fac)."""
    p, t, e = site_met(when)
    return float(corr_rh_facs(elv_min, elv_max, p, t, e))


if __name__ == "__main__":
    for label, when in [("winter", "2026-01-19 12:00"), ("summer", "2026-07-17 12:00")]:
        p, t, e = site_met(when)
        print(f"{label}: P={p:.1f} hPa  T={t:.1f} C  e={e:.1f} hPa")
        for lo, hi in [(0.5, 6.5), (1.0, 8.0), (1.5, 12.0)]:
            fac = rh_correction_factor(when, lo, hi)
            print(f"  elv {lo}-{hi}: rh_fac={fac:.5f}  -> at RH=31m: {fac*31*100:+.1f} cm; at RH=24m: {fac*24*100:+.1f} cm")
