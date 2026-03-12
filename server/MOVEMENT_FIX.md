# Fix: People spending too much time at home

**File:** `Algorithms/server/patterns.py`
**Date:** 2026-03-02

## Problem

Simulated agents were spending far too much time at home. In many runs, ~50% of the population never left home on a given day, and those who did rarely made more than one trip.

## Root cause

The per-hour probability of a person leaving home (`base_move_prob`) was derived from a **normalized probability distribution**, which made it vanishingly small as the number of facilities grew.

The original code (line 258):

```python
place_probs = place_probs / place_probs.sum()   # normalize to sum=1
overall_busy = float(place_probs.max())          # BUG: max of a probability distribution
base_move_prob = min(0.35, 0.85 * overall_busy)
```

`place_probs` is the destination-selection distribution (sums to 1.0). Taking `.max()` of it returns the share of the single most popular facility — **not** a measure of overall activity. With 100 facilities, even the most popular might hold 3-5% of the distribution, yielding:

```
overall_busy ≈ 0.05
base_move_prob = min(0.35, 0.85 * 0.05) = 0.0425  →  ~4% chance per hour
P(never leave in 16 waking hours) = (1 - 0.0425)^16 ≈ 50%
```

Critically, adding more facilities made the problem **worse**: the normalized max shrinks as the distribution spreads across more places.

## Fix

Separate the two concerns that were previously conflated:

1. **Destination selection** (given you're going out, *where* do you go?) — uses the normalized `place_probs` distribution. This was correct and is unchanged.

2. **Movement probability** (should you go out *at all*?) — now uses **raw SafeGraph visitor counts** to measure absolute activity level for the current timeslot.

### Changes

#### 1. `load_patterns_csv` — preserve raw counts

Added `raw_hour_counts` (list of 24 ints) and `raw_day_counts` (dict of weekday -> int) to each facility's stats entry, alongside the existing normalized weights. These are the original SafeGraph visitor counts before normalization.

#### 2. New helper: `_compute_peak_busyness(stats)`

Pre-computes the maximum aggregate raw activity across all 7 days x 24 hours (limited to 6am-10pm). For each timeslot, sums `raw_hour_counts[hour] * raw_day_counts[weekday]` across all facilities. Returns the single highest value — representing the busiest hour of the busiest day.

#### 3. New helper: `_aggregate_busyness(stats, weekday, hour)`

Computes the same aggregate sum for a specific timeslot. Used in the hourly loop.

#### 4. `gen_patterns` — replaced busyness calculation

```python
# Before (broken):
overall_busy = float(place_probs.max())

# After (fixed):
peak_busyness = _compute_peak_busyness(stats)        # once, before the loop
overall_busy = _aggregate_busyness(...) / peak_busyness  # each hour
```

`overall_busy` is now a 0-1 ratio of current activity vs. peak activity. During the busiest hour of the week it reaches 1.0; during quiet early mornings it drops toward 0. The formula `min(0.35, 0.85 * overall_busy)` then produces meaningful probabilities:

- Peak hour: `min(0.35, 0.85 * 1.0) = 0.35` → 35% chance per person per hour
- Moderate hour: `min(0.35, 0.85 * 0.5) = 0.425` → capped to 35%
- Quiet morning: `min(0.35, 0.85 * 0.15) = 0.128` → 13% chance

## What stays the same

- `_overall_busy_factor()` and normalized `place_probs` — destination selection logic unchanged
- Dwell time sampling — unchanged
- 10pm curfew / 6am-10pm activity window — unchanged
- `min(0.35, 0.85 * ...)` formula — unchanged (but the 0.35 cap and 0.85 multiplier may benefit from tuning now that the input is meaningful)
- All intervention logic (lockdown, self-isolation, capacity) in `simulate.py` — unchanged

## Future tuning

The `0.35` cap and `0.85` multiplier on line 316 control the upper bound of movement. Now that `overall_busy` produces a proper 0-1 signal, these constants can be tuned based on empirical validation — e.g., comparing simulated time-away-from-home distributions against real-world mobility data.
