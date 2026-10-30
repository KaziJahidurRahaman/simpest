# fr_disease_model module

::: simpest.models.fr_disease_model

## Upper-leaf tissue pool (optional)

FraNchEstYN tracks one pool of host tissue for the whole canopy. Field
severity is often assessed on the upper leaves only, which emerge shortly
before heading. By then the single pool can already be largely infected or
senesced, so simulated severity responds to the weather before heading while
the observed severity responds to the weather after it.

Setting `upper_leaf_pool=True` on
[`FranchestynConfig`](franchestyn.md) adds a second pool for the upper leaves.
It is off by default, and with it off the model is unchanged.

- **Opening.** The pool starts empty and opens linearly between the crop cycle
  percentages `upper_leaf_cycle_start` and `upper_leaf_cycle_full`.
- **Infection.** Each day's infection pressure is the canopy's (primary
  inoculum plus secondary spread from the canopy's sporulating tissue) plus
  secondary spread from the pool's own sporulating tissue, at the same
  `PathogenSpread` rate and with the same rain detachment. New infection is
  that pressure times the emerged, unaffected fraction of the pool. It is not
  scaled by light interception and no tissue is lost to senescence.
- **Progression.** Cohorts move through latent, sporulating and dead with the
  same durations as in the canopy.
- **Outputs.** `UpperLeafEmerged` and `UpperLeafSeverity` (sporulating plus
  dead, relative to the emerged area) in the simulation records. With the
  switch on, the calibration objective compares the reference severity with
  `UpperLeafSeverity` instead of `DiseaseSeverity`.
- **Not changed.** The canopy epidemic, and therefore the damage to the crop
  and the yield loss, are the same with the switch on or off.

The pool adds no calibrated parameter. The default window (36 % to 52 % of the
crop cycle) is for the upper four leaves of soft red winter wheat cv. Monon at
West Lafayette, Indiana. In 1990, leaf F-2 was fully expanded 27 days, F-1
21 days and the flag leaf 14 days before spike emergence (Shaner and Buechley
1995, *Plant Disease* 79:928–938), which is 43 %, 47 % and 52 % of the
simulated crop cycle of that season; the start is extrapolated back to the
emergence of F-3. Set both values for another crop, cultivar or leaf layer.

The C# FraNchEstYN model has no such pool, so results with the switch on have
no C# counterpart.
