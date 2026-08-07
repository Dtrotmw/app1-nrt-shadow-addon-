"""
masks.py

Candidate azimuth/elevation masks for the Phase 2 experiment (plan
`unified-growing-pony.md`): does expanding the mask toward the NW -- where
Simon Williams' SWOT imagery suggested better low-water reflections --
reduce the spring-low-tide bias, scored against the slipway survey marks?

Change log
----------
v1  2026-07-29  Claude Code. Initial version (Phase 2).
"""
from pipeline.arcfit import BASELINE_SECTORS

# (a) Control: the live, currently-deployed mask -- unchanged.
MASK_BASELINE = dict(BASELINE_SECTORS)

# (b) NW-expansion test: baseline + the North sector already defined in
# V91.ipynb/V101.ipynb/V103.ipynb's shared SECTORS dict (confirmed identical
# across all three -- see CHANGELOG "Added group-3 storm/mask cluster").
# Reusing already-defined geometry rather than deriving new bounds from the
# SWOT image -- that's candidate (c), not yet built (open question in the
# plan: only worth doing if (b) shows nothing and David wants to hand-derive
# azimuth/elevation bounds from the image directly).
MASK_NW_EXPANDED = dict(BASELINE_SECTORS)
MASK_NW_EXPANDED["North"] = dict(minaz=300, maxaz=33, minel=1.5, maxel=12.0, weight=0.5)
# Note: `weight` here is carried over from the old local code's convention but is
# NOT currently used anywhere in arcfit.py/consensus.py -- the consensus step weights
# by PNR only (Phase 1 simplification). Kept as documentation of intent, not live config.

# (c) Refined North sector, per pipeline/run_north_sector_sweep.py (2026-08-02):
# these boundaries (300-33, 1.5-12) were never actually tuned -- inherited
# wholesale from V91/V101/V103's shared SECTORS dict. A coordinate-descent
# sweep of azimuth/elevation boundaries (David's request, after asking whether
# we'd actually tried Altuntas & Tunalioglu 2023's systematic empirical
# mask-optimization method -- we hadn't; we'd only tested this ONE hand-picked
# candidate against baseline) found extending maxaz from 33 to 43 improves
# RMSE ~7% and bias ~18% against the slipway ground truth (10 dates, 296
# matched readings), and wins the leave-one-date-out stability check in 9/10
# folds. Kept as a separate candidate, not substituted into MASK_NW_EXPANDED
# above, since that's the specific definition already sent to Simon as Track
# A evidence. This refinement was subsequently sent too, in a combined note,
# 2026-08-03 (CHANGELOG entry 38).
MASK_NW_REFINED = dict(BASELINE_SECTORS)
MASK_NW_REFINED["North"] = dict(minaz=300, maxaz=43, minel=1.5, maxel=12.0, weight=0.5)

# (d) North sector ALONE (no Chivenor/Instow) -- CHANGELOG entry 59, 2026-08-07:
# properly settled (block-continuous, matching how the live add-on actually
# runs -- never resetting), North-only clearly beats the wide MASK_NW_REFINED
# combination on both raw/K2D accuracy and the decision-relevant false-negative
# rate, in every band tested against the full 41-reading slipway record. The
# earlier "wide mask beats North" reading (this same mask, entry 46) was a
# settling-time artefact of a differently-designed sliding-window test, not a
# real property of the sectors -- see entry 59 for the full correction.
# Deployed live in the shadow trial add-on from v0.2.2 onward (was
# MASK_NW_REFINED through v0.2.1).
MASK_NORTH_ONLY = {"North": dict(minaz=300, maxaz=43, minel=1.5, maxel=12.0, weight=0.5)}

ALL_MASKS = {
    "baseline": MASK_BASELINE,
    "nw_expanded": MASK_NW_EXPANDED,
    "nw_refined": MASK_NW_REFINED,
    "north_only": MASK_NORTH_ONLY,
}
