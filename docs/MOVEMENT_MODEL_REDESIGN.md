# Delineo Movement Model: POI Over‑Occupancy & the Open‑System Problem

**A diagnosis, the data behind it, the relevant literature, and a recommended design.**

*Written 2026‑06‑20. Companion to `MOVEMENT_MODEL_REVIEW.md`. All numbers below were measured firsthand on run 173 (Barnsdall, OK) and the source file `Algorithms/server/data/patterns/OK/2021-04-OK.parquet`; literature claims were adversarially fact‑checked (25 claims, 3‑vote verification).*

---

## 0. Executive summary (read this first)

**The symptom.** A dental office in our simulation ("Fooshee D Scott Dntst") fills up with **1,220 people at once in a 374 m² office** — 3.3 people per square metre, denser than a packed nightclub. Many POIs do this.

**The root cause (one sentence).** The model decides *where people go* using each POI's *self‑normalized* hourly popularity *shape* and throws away *how many people actually go there* — so a place with **2 recorded visits all month** looks just as attractive as a Walmart, and during the quiet 3 a.m. hour it becomes the single most attractive destination in the entire region.

**A second, deeper problem we uncovered.** The simulation models a small cluster of **18 Census Block Groups (CBGs)**, but the POIs inside that cluster draw **~85% of their real‑world visitors from outside it**. So the region is an *open system*: most foot traffic at our POIs comes from people we don't simulate.

**The recommended design (the choice).**

1. **Fix occupancy with the data already in hand.** Drive and cap each POI's demand from its **`popularity_by_hour`** field — which turns out to be a literal observed *occupancy curve* — instead of the self‑normalized shape. This kills the over‑occupancy and the "noise POI" problem in one move.
2. **Scale demand to our population** using the per‑POI catchment fraction **`f_j ≈ 0.13–0.15`** (the share of a POI's visitors who actually live in our zone). Without this you refill POIs to metro scale and recreate the bug.
3. **Handle the external 85% as a bounded, one‑way "external force of infection" term** at each POI — the literature‑standard way to couple a small region to the outside world — rather than ignoring it (indefensible at 85%) or simulating a million extra agents (defeats the purpose).
4. **Keep the seed‑grown zone.** It correctly bounds *our residents'* movement; it was never going to bound *inbound* visitors, and it doesn't need to once (3) is in place.

The rest of this document explains each point, simply and then in detail.

---

## 1. The symptom

Run 173 is a Barnsdall, OK simulation (18 CBGs, ~49,700 people, 1,843 POIs, Delta variant, 1 initial infection). Reading its results directly:

| POI | peak occupancy | floor area | density |
|---|---|---|---|
| Birch Lake Twin Coves Recreation Area | **3,622** (7.3% of the whole population) | 81 m² | 44.7 / m² |
| Owasso Little Free Library | 1,326 | 244 m² | 5.4 / m² |
| **Fooshee D Scott Dntst** | **1,220** | 374 m² | **3.3 / m²** |
| B Square Motel | 1,205 | 112 m² | 10.8 / m² |

22 POIs peak above 500 people; 317 above 100. These numbers are physically impossible and they distort the epidemic (a POI crammed with 1,200 people becomes a fake super‑spreading site).

**The tell:** in the *same* run, every dentist with *real* visit data behaves sanely — Aspen Dental peaks at 38, Heartland at 33, Bartlesville Family Dental at 147. Only "Fooshee D Scott Dntst" explodes. The blow‑ups are not random; they are exactly the POIs the mechanism below predicts.

---

## 2. Why it happens — the root cause

### In plain terms

Imagine you're handing out a town's daily errands to shops. Instead of asking *"how busy is each shop?"*, the current model asks *"what fraction of THIS shop's own activity happens right now?"* and sends people in proportion to that. A shop with two customers all month, both of whom happened to show up at 3 a.m., looks like it does **100% of its business at 3 a.m.** — so at 3 a.m. it looks like the hottest destination in town. A Walmart that's busy all day looks only mildly busy in any single hour. So the near‑empty shop wins.

### In detail

Each hour, the *old* model gave every POI a weight and sent movers in proportion to it (this function, `_overall_busy_factor`, was removed in §9's rewrite). The weight was:

```
weight(POI, hour, weekday) = hour_share[hour] × day_share[weekday]
```

and both `hour_share` and `day_share` are **normalized *within each POI* to sum to 1**. That normalization deletes absolute volume. Concretely, for our dentist:

- Real data: `popularity_by_hour = [0,0,0,1,0,1,0,…]` — i.e. **2 visits all month**, one logged around 3 a.m. and one around 5 a.m. `raw_visit_counts = 2`.
- After normalizing: the 3 a.m. slot becomes **0.5** (half of its entire "activity").
- A Walmart (`raw_visit_counts = 11,024`) has its visits spread smoothly across the day, so its 3 a.m. slot is `43 / 20,141 ≈ 0.002`.

So at 3 a.m. the 2‑visit dentist is weighted **~1,000× higher than the Walmart**, despite Walmart having **5,500× more real visits**. The normalization *inverts reality*.

When we measured this on the real POI set, at Wednesday 3 a.m. the dentist was **rank #1 of 603 active POIs and captured 28.5% of everyone who moved that hour**. Systematically, **in 86% of all weekday‑hour slots the single most‑attractive POI was a place with ≤10 visits/month** (median: 4 visits/month). The model is, most of the time, sending the largest share of movers to a near‑empty building.

### The four structural gaps

The normalization bug is the loudest, but it sits on top of three more:

1. **No absolute volume** — a dentist is treated like a mall (just described).
2. **No spatial structure** — destinations are drawn from one region‑wide list; there's no notion of "people go to places near them."
3. **No capacity** — nothing caps how many people can be in a building. The simulator *has* a capacity guard ([`runner.py`](Simulation/simulator/runner.py)), but it's dormant (capacity is always `-1` = unlimited) and the production engine bypasses it anyway. (See `project_poi_capacity_dormant`.)
4. **No open hours** — the 3 a.m. dentist visit is only possible because opening hours are ignored.

---

## 3. The data we actually have

The source files (`data/patterns/{STATE}/{YYYY-MM}-{STATE}.parquet`) have ~50 columns. The current model uses **four** and normalizes the useful magnitude out of them. The fields that matter, with coverage:

| Field | Coverage | What it is |
|---|---|---|
| `popularity_by_hour` | 78.5% | 24 numbers: **device‑hours present** at the POI by hour‑of‑day, summed over the month |
| `raw_visit_counts` / `raw_visitor_counts` | 78.5% | absolute monthly visits / unique visitors (a ~5% device **sample**) |
| `normalized_visits_by_state_scaling` | 78.5% | SafeGraph's panel→population projection (≈ `raw × 19.7` for OK) |
| `visitor_home_cbgs` | 66.7% | **`{home_cbg: visitor_count}`** — the directly‑observed origin of each POI's visitors |
| `wkt_area_sq_meters` | 92.8% | floor area (already emitted to papdata as `area`) |
| `median_dwell` / `bucketed_dwell_times` | 74–79% | how long visitors stay |
| `open_hours` | 41.2% | opening hours per weekday |

### The key insight about `popularity_by_hour`

**`popularity_by_hour` is not a visit count — it's an occupancy curve.** We checked: `sum(popularity_by_hour) / raw_visit_counts ≈ 1.56`, meaning it counts *device‑hours present*, not visit events. So:

```
average people present at POI j, hour h  ≈  popularity_by_hour_j[h] × 19.7 / 30
                                            └ sampled month total ┘  └panel┘ └days┘
```

- **`× 19.7`** converts the ~5% device sample to the full population (this factor is read from the data, near‑constant statewide).
- **`÷ 30`** converts a monthly total to a per‑day average (April = 30 days).

> **`popularity_by_hour` and `popularity_by_day` are different units — don't conflate them.** `popularity_by_day` sums to `raw_visit_counts` *exactly* (100% of POIs): it counts each visit once, on its day — a **visit-event count**. `popularity_by_hour` sums to ~1.57× that, because a visit is counted in every clock-hour it spans (~`1 + dwell/60`): it's **person-hours present = occupancy**. Use `popularity_by_hour` for occupancy/the cap; use `popularity_by_day` only as a *relative day-of-week multiplier* (its shape), never as an occupancy number. The current model's bug includes normalizing both into one selection weight, which silently mixes these units.

This single field gives us, per POI, a realistic occupancy ceiling — exactly what the model is missing. Sanity check:

| POI | implied true peak occupancy | current sim |
|---|---|---|
| Fooshee D Scott Dntst | **~0.7** | 1,220 |
| Birch Lake Rec Area | **~0.7** | 3,622 |
| Aspen Dental (real) | ~13 | (sane) |
| Walmart | ~1,269 (full catchment) | (big) |

The noise POIs cap at ~1; the real ones land at sane values.

---

## 4. The deeper problem: the region is "open"

### 4a. Most of our POIs' visitors don't live in our zone

For run 173's POIs, we measured **what fraction of their visitors actually live in our 18 CBGs**:

- **~15%** of the *listed* visitor flow, **~13%** once you account for visitors SafeGraph drops from its truncated lists.
- So **~85% of foot traffic at our POIs comes from people outside the simulated population.**

This is the **catchment fraction**, which we'll call **`f_j`** for POI *j* (≈ 0.13–0.15 on average, varying per POI).

### 4b. Why the demand looks "405% of the population"

If you drive the simulation by raw `popularity_by_hour` (full catchment), the POIs collectively "want" **201,355 people at peak — 405% of our population.** That's not a bug; it's the 85% externals showing up as demand. Strip them out (`× f_j`) and peak demand becomes **29,829 ≈ 60% of the population out at once**, with everyone else at home. Conserved and realistic.

**The lesson:** `popularity_by_hour` describes the POI's *full, real* occupancy (great as a *ceiling*); `popularity_by_hour × f_j` is the part *our* population is responsible for (the demand we should actually generate). The remaining `(1 − f_j)` is external traffic we must either ignore or model as background.

### 4c. "But the mobility prune keeps 70% inside!" — reconciling the two numbers

The convenience zone is grown around a **seed CBG** until ~70% of that seed's movement is "captured." That 70% and our 15% are **different measurements**, both correct:

| | What it measures | Value |
|---|---|---|
| Prune's "seed capture" | of the **seed CBG's outbound** trips, the share that land on POIs inside the cluster | **71.0%** |
| Our catchment | of the **POIs' inbound** visitors, the share who live inside the cluster | **15.0%** |

These are opposite directions. A mall can be *"where 71% of your town's trips go"* (outbound) **and** *"a place where your town is only 15% of the shoppers"* (inbound) at the same time. We verified it's not a column artifact either: redoing the inbound measurement with the prune's own data column still gives ~18%, nowhere near 70%. **Outbound capture and inbound locality are not symmetric — and the prune only controls the outbound side.**

### 4d. Why Chang et al. (the gold‑standard SafeGraph model) only drop 3–10%

The leading SafeGraph POI model — **Chang et al. 2021, *Nature*** — also treats its region as closed, but it only discards ~3–10% external traffic. How? **Scale.** Chang models an entire *metro area* (~5,700 CBGs per metro); we model 18. A metro is, by Census design, roughly self‑contained — so it ≈ the catchment of its own POIs. Watch our external fraction shrink as we grow the modeled region:

| Modeled region | external fraction |
|---|---|
| **18 czone CBGs (us)** | **85%** |
| + their 3 counties | 33% |
| + Tulsa/Bartlesville metro (8 counties) | **14%** |
| + all of Oklahoma | 8% |

**At metro scale — Chang's unit — leakage is ~8–14%, right in his ballpark.** The rule of thumb:

> **external fraction ≈ 1 − (modeled region ÷ POI catchment size).**

Chang's region ≈ catchment → small leakage. Our region ≪ catchment → 85% leakage. *We didn't do anything wrong; we chose a smaller, faster simulation unit, and the leakage is the price of that choice.*

---

## 5. What the literature says about the open‑system problem

(Adversarially fact‑checked; sources linked at the end.)

1. **The closest analog ignores it.** Chang et al. 2021 treats each metro as **closed**, down‑scaling each POI's visits to the in‑region total and dropping the external 3–10%. It seeds infection **once** (a single fitted "initial exposed" parameter) with **no ongoing importation term**. Aleta et al. 2020 (Boston) is likewise closed and explicitly names infected‑traveler reintroduction as an *unmodeled limitation*. **Copying their closed choice is far harder to defend at 85% than at 3–10%.**

2. **A background "force of infection" term is the textbook fix, not a hack.** The metapopulation literature has exactly two ways to couple a region to the outside world: *mechanistically* (simulate the travelers) or via an *effective force‑of‑infection coupling* (Colizza & Vespignani). Since we **don't** simulate the 85%, the effective‑coupling route — an **exogenous, one‑way background term** — is the right one. Sattenspiel–Dietz shows "closed" and "fully coupled" are just two ends of one dial, so a calibrated term is principled.

3. **Parameterize it by *time‑at‑risk*, not headcount (Citron et al. 2021, PNAS).** An external visitor who spends 1 hour at a POI should contribute *less* infection pressure than a resident exposed there all day. Weight by **dwell time**, which connects directly to our `median_dwell` / `bucketed_dwell_times` fields.

4. **Bound it, or it breaks (Citron et al.).** A naïve memoryless importation term *over‑seeds* and becomes mathematically uninterpretable under heavy traffic — exactly our regime. The term must be bounded (behave like a "return‑home trip," not unlimited diffusion).

5. **Seed it discretely (Colizza & Vespignani).** Use whole imported infections with an extinction probability, not a continuous fractional drip, or you'll spuriously guarantee an outbreak.

6. **GLEaM is the working template (Balcan et al. 2009).** It lets non‑resident visitors contribute to a location's local force of infection as a *stationary occupancy term* **without simulating them as agents** — precisely the mechanism we need.

7. **Don't double‑count.** If you add a background term, **reduce or zero the one‑time t=0 seed**, and treat the external pool as a one‑way source (don't let our simulated cases flow back and inflate the external prevalence we assumed).

---

## 6. The recommended design

The two problems (over‑occupancy and the open system) share one fix family, because both come from the same root: the model ignores *real observed volume*. The design has three layers, each independently shippable.

### Layer A — Fix occupancy from `popularity_by_hour` (no new data needed)

- **Replace** the self‑normalized selection weight with **absolute occupancy** from `popularity_by_hour`. A 2‑visit dentist now carries ~0 weight; it can never win the 3 a.m. lottery again.
- **Cap** each POI's occupancy at its observed ceiling `C_j ≈ max_h(popularity_by_hour_j[h] × 19.7 / 30)`, with an **area‑based floor** (`area / m²‑per‑person`) for the ~21% of POIs with no popularity data, so they don't cap at zero.
- **Gate by `open_hours`** where present (kills 3 a.m. visits); fall back to category‑typical hours otherwise.

This alone fixes pathologies #1 (noise), #2 (magnitude), #3‑partially, and #4 (capacity). **It is the highest‑value, lowest‑effort step and needs no spatial data.**

### Layer B — Scale demand to our population with the catchment fraction `f_j`

`popularity_by_hour` is full‑catchment occupancy. To generate the *right number of our own people*, multiply each POI's demand by **`f_j`** = its in‑cluster visitor share (computed once per run from `visitor_home_cbgs`). This is a *single number per POI*, far lighter than a full origin–destination matrix. Without it, demand is 405% of our population and POIs refill to metro scale.

### Layer C — Represent the external 85% as a background force of infection

For each POI *j* and hour *h*, add an exogenous transmission‑pressure term:

```
external_pressure_j(h)  ∝  P_ext(t)  ×  [ popularity_by_hour_j[h] × 19.7 / 30 ]  ×  (1 − f_j)
                            └ outside  ┘   └─ occupancy = person‑hours present ─┘  └ external ┘
                             prevalence       (Citron time‑at‑risk already baked in)  share
```

- **`P_ext(t)`** = prevalence outside the zone, from county/metro case data (refinable via the `visitor_home_cbgs` origin mix — we know *which* outside counties feed each POI).
- **`(1 − f_j)`** = the external share (the 85%) — the exact complement of Layer B.
- **No separate dwell term.** `popularity_by_hour` is *person‑hours present* (we verified `sum/raw_visit_counts ≈ 1.57 ≈ 1 + dwell/60`), so it **already encodes** Citron's time‑at‑risk. Multiplying by an extra dwell weight would double‑count exposure time. (The dwell fields are still used on the *resident* side, to decide how long simulated people stay — just not here.)
- Implemented as **discrete, stochastic** seeding, **one‑way** (no feedback into `P_ext`), and **reduce the t=0 seed** so it isn't double‑counted. In the Wells‑Riley kernel it enters as an added quanta source at the POI.

**Every quantity here is something we already have.** The open‑system fix is a natural extension of the same three fields, not a new data dependency.

### What each layer fixes

| Problem | Fixed by | How |
|---|---|---|
| Noise POI wins (#1) | A | absolute weight; a 2‑visit POI carries ~0 mass |
| Dentist == mall (#2) | A | weight ∝ real occupancy, not normalized shape |
| Over‑concentration (#3) | A + B | observed occupancy + catchment scaling spread people realistically |
| 1,220 in a dentist (#4) | A | per‑POI occupancy cap at generation time |
| 3 a.m. visits | A | open‑hours gate |
| 85% external traffic | C | bounded, one‑way, dwell‑weighted external FOI |

### How it all feeds Wells‑Riley (we keep the kernel)

**The redesign is entirely *upstream* of transmission — the Wells‑Riley kernel stays exactly as it is.** Layers A+B fix *how many of our residents* are in each room; Layer C adds the *outsiders'* expected infectious presence. Both do nothing but hand a better **infector count `n`** to the unchanged kernel:

```
                     one POI, one hour  →  fed into the Wells‑Riley kernel

  OUR residents (simulated)              EXTERNAL visitors — 85% (NOT simulated)
  occupancy = PBH × 19.7/30 × f_j        background = PBH × 19.7/30 × (1 − f_j) × P_ext(t)
  capped at C_j, open‑hours gated        one‑way  (no feedback into P_ext)
            │                                          │
            └────────────────────┬─────────────────────┘
                                 ▼
        infectors sharing the air:
            n = (our infectious present)  +  (expected external infectious)
                                 ▼
        Wells‑Riley kernel  —  UNCHANGED
            risk = 1 − exp( − n · q · p · t / Q )      Q = area‑aware ventilation
                                 ▼
            our susceptible residents get infected
   (PBH = popularity_by_hour)
```

`n` is the only thing that changes. `q` (quanta emission), `p` (breathing rate), `t` (exposure time), and `Q` (the already‑shipped area‑aware ventilation) are untouched. So the occupancy fix and the external‑FOI term make Wells‑Riley **more accurate**; they do not replace it. The O(n) aggregate kernel is the same math, just vectorized.

### Staged rollout (each behind a flag, mirroring `area_aware_ventilation`)

- **Stage 0** — surface the extra columns in the loader (no behavior change).
- **Stage 1** — Layer A: absolute weight + open‑hours gate. *Gate: the noise POIs (Fooshee, the rec area) drop to ~single digits; real POIs unchanged.*
- **Stage 2** — Layer A capacity cap. *Gate: no POI exceeds a sane persons‑per‑m².*
- **Stage 3** — Layer B catchment scaling. *Gate: total out‑of‑home rate is realistic (~30–60% at peak), conserved.*
- **Stage 4** — Layer C external‑FOI term. *Gate: the ablation below.*

### The decisive experiment

The theory (Colizza–Vespignani's invasion threshold; Rapaport–Mimouni showing coupling can flip outbreak/no‑outbreak) says the external term can change *whether* the zone sustains an outbreak, not just by how much. So run the **ablation: closed vs. background‑FOI vs. full‑catchment**, and see whether the term moves the zone across the outbreak threshold. That experiment decides how much the open‑system choice actually matters for our results.

---

## 7. Open questions / decisions to settle

1. **Region scale vs. coupling term.** Growing the zone to metro scale would shrink leakage to ~10% (Chang's regime) but means simulating ~20× the population and abandoning the fast convenience‑zone premise. The recommendation keeps the small zone + the term — but this is the fundamental fork and worth an explicit decision. *(Both are the same trade‑off from opposite ends: Chang pays with scale; we pay with a coupling term.)*
2. **Where `P_ext(t)` comes from** and how sensitive results are to it vs. our internal transmission parameter (which already overshoots ~4× in external validation — so movement realism may let us isolate and fix the transmission level).
3. **No published hard threshold** exists for "what % external invalidates a closed model." Chang tolerates 3–10%; our 85% is far beyond — strong qualitative, not quantitative, grounds.
4. **Calibration interaction** between the occupancy cap's density floor and the already‑shipped area‑aware ventilation.
5. **`movement_scale` is the single mixing knob.** The demand‑pull architecture (§9) makes the total out‑of‑home level a linear function of one factor (`movement_scale`, default `panel/30 ≈ 0.66`). The data‑raw default puts ~80% of people out at peak — too high — so this must be calibrated against real case counts (it is the parameter behind the ~4× validation overshoot). Per‑POI occupancy is plausible at *every* scale; only the level needs tuning.
6. **School under‑representation (FOLLOW‑UP — after the redesign).** Verified in the data: workplaces *are* in the POI set (~30% of total occupancy weight — offices, medical, retail, mfg). Schools *are* too (44 POIs) with realistic long dwell, but carry only **~1% of the weight** because SafeGraph barely sees students (children don't carry tracked phones). So the model under‑fills schools and **under‑represents school‑age daytime mixing**. Worth a dedicated pass once the movement redesign lands — e.g. boosting school POI occupancy toward enrollment rather than observed visits. (Also flagged: the airport appears as ~4 co‑located duplicate records, inflating its share — a POI‑dedup hygiene item.)

---

## 8. Key numbers & sources

**Measured (run 173 / `2021-04-OK.parquet`):** Fooshee dentist 1,220 people in 374 m² (raw_visit_counts = 2); captures 28.5% of 3 a.m. movers; 86% of time‑slots won by ≤10‑visit POIs; `popularity_by_hour` ≈ device‑hours (ratio 1.56); panel factor ≈ 19.7; catchment `f_j` ≈ 0.13–0.15; full demand 405% of pop, scaled 60%; seed capture 71.0% vs inbound 15%; scale ladder 85% → 33% → 14% → 8%.

**Literature (peer‑reviewed, fact‑checked):**
- Chang et al. 2021, *Nature* — SafeGraph CBG→POI ABM, closed system. https://www.nature.com/articles/s41586-020-2923-3
- Aleta et al. 2020, *Nature Human Behaviour* — closed Boston ABM. https://www.nature.com/articles/s41562-020-0931-9
- Citron et al. 2021, *PNAS* — time‑at‑risk coupling; over‑seeding warning. https://www.pnas.org/doi/10.1073/pnas.2007488118
- Balcan et al. 2009, *PNAS* — GLEaM stationary‑occupancy visitor coupling. https://www.pnas.org/doi/10.1073/pnas.0906910106
- Colizza & Vespignani — effective‑coupling vs mechanistic; invasion threshold. https://arxiv.org/pdf/0706.3647
- Sattenspiel & Dietz 1995, *Math. Biosciences* — closed↔coupled continuum. https://pubmed.ncbi.nlm.nih.gov/7606146/

*Related internal docs: `MOVEMENT_MODEL_REVIEW.md` (the original scientific review), `EXTERNAL_VALIDATION_PLAN.md`.*

---

## 9. What we actually built (updates since §6)

The implementation simplified §6's IPF‑first plan into a staged build, verified on the real Barnsdall (czone‑108) data. The legacy supply‑push model — and the `DELINEO_LEGACY_MOVEMENT` / `DELINEO_CATCHMENT` / `DELINEO_DEMAND_PULL` flags — was **removed entirely**; demand‑pull is the only model, and missing/empty patterns data now **raises `ValueError`** (`patterns.py`) rather than silently producing a movement‑free run. There is no rollback flag.

- **Stage 0** — surface the redesign columns + per‑field coverage report (plumbing only). *(committed)*
- **Stage 1** — destinations weighted by **absolute occupancy** (`popularity_by_hour`) + **open‑hours gate**, replacing the self‑normalized shape. Fooshee 1,210 → 0. *(committed)*
- **Stage 2** — **catchment scaling** `f_j` from `visitor_home_cbgs` (observed where present, flat median fallback for the ~23% low‑traffic POIs without it; gravity fallback was measured unnecessary — those POIs carry 0.06% of weight).
- **Stage 3 — demand‑pull (the key correction).** Catchment as a *weight multiplier* in the old supply‑push loop was measured to **redistribute** over‑occupancy, not remove it (the airport's lost share ballooned onto local POIs; a hotel hit 1.03 ppl/m²). Root cause: a global move‑rate pushed a fixed flood of people out, re‑normalized — so reducing one POI inflated others, and dwell double‑counted on long‑stay POIs. The fix is **demand‑pull**: each POI is filled to its absolute occupancy target `popularity_by_hour × day_factor × f_j × open_gate × movement_scale`, topped up from the home pool, with everyone not pulled staying home. Result on real data: **0 density violations at every scale**, airport 8,309 → 1,236, Fooshee 0.

**The occupancy cap (§6 Layer A) is therefore demoted to an optional backstop**, not a primary fix — demand‑pull bounds occupancy at the source, and the cap never binds (measured). The remaining open item is **`movement_scale` calibration** (§7.5) and the **external‑FOI term** (§6 Layer C), still to come.
