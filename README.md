# caseload

**Reinforcement learning environments for budgeted fraud investigation:** deciding where to point scarce human attention, and when, while the thing you are hunting moves.

[![CI](https://github.com/JayeshSuryavanshi/caseload/actions/workflows/ci.yml/badge.svg)](https://github.com/JayeshSuryavanshi/caseload/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/caseload.svg)](https://pypi.org/project/caseload/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](https://github.com/JayeshSuryavanshi/caseload/blob/main/LICENSE)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/JayeshSuryavanshi/caseload/main/docs/img/collapse-dark.png">
  <img src="https://raw.githubusercontent.com/JayeshSuryavanshi/caseload/main/docs/img/collapse-light.png" alt="Two panels. Left: per-step recall at a 2% budget for a detector frozen on Elliptic steps 1-34. It runs between 36% and 91% through step 42. After the dark-market shutdown at step 43 it finds 2 of 169 illicit transactions over steps 43-49, and five of the seven steps are at zero (step 46 reads 50% because it has only two illicit transactions). Right: recall on steps 45-49 by budget. A detector refitted on labels up to step 44 climbs to 50% at a 10% budget and 77% at 20%, while the frozen detector stays under 6% and random picks reach 27% at 20%.">
</picture>

A detector frozen before a regime break does not degrade gracefully. On the Elliptic Bitcoin transaction graph (Weber et al., 2019), a detector fitted on steps 1-34 finds 36% to 91% of each step's illicit transactions up to step 42. After the dark-market shutdown at step 43 it finds **2 of the 169** illicit transactions in steps 43-49. Given every label up to step 44, including the two steps after the break, the same detector refitted finds **49.6%** of the illicit transactions in steps 45-49 at a 10% budget, against **3.3%** for the frozen one. The information needed to recover exists, and getting it costs investigation budget.

Two environments, one shared evaluation harness. Both are small enough that the 30-seed baseline study below runs on a laptop in under 15 minutes. That matters because Patterson et al. (JMLR 2024) show how badly a handful of runs can mislead: in one of their examples, 10 runs of DQN put the estimated mean far from the true one, and they write that in almost all cases 5 runs is not enough for strong claims, while even 30 can fall short when outcomes are skewed. The learned-policy results further down rest on 6 training seeds, fewer than that, and are labelled as such.

Every result below is read from a file under `results/` that a script in this repository wrote, or is printed by a script from those files; [Reproducing](#reproducing) gives the command for each. Where a result is negative, or where my own first attempt was wrong, that is written down rather than removed.

The write-up of how this went, including the two results that reversed under scrutiny, is [**The simulator was the experiment**](https://www.jayeshsuryavanshi.com/blog/simulator-was-the-experiment.html).

---

## 1. Selective labels under a regime break (`caseload.envs.drift`, `caseload.envs.elliptic`)

Investigating a case reveals its label. Nothing else does. So the detector is only ever retrained on cases somebody chose to look at, and a policy that always spends its budget on the highest-scoring cases stops learning the moment the fraud population moves, and never finds out that it has.

This is the selective-labels setting (Lakkaraju et al., KDD 2017) as a sequential problem, and an instance of a **Monitored MDP** (Parisi et al., AAMAS 2024), where the reward exists whether or not you observe it and observing it competes for the same budget as exploiting it.

### The motivating measurement

On the Elliptic Bitcoin transaction graph, a gradient boosting detector fitted on time steps 1-34 and then frozen, at a 2% per-step budget (illicit transactions found / present):

| step | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42 | **43** | 44 | 45 | 46 | 47 | 48 | 49 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| found | 95/182 | 30/33 | 24/40 | 50/111 | 29/81 | 52/112 | 55/116 | 93/239 | **0/24** | 0/24 | 0/5 | 1/2 | 0/22 | 0/36 | 1/56 |
| recall | 52% | 91% | 60% | 45% | 36% | 46% | 47% | 39% | **0%** | 0% | 0% | 50% | 0% | 0% | 2% |

Step 43 is the dark-market shutdown documented by Weber et al. (2019), after which every model they tested, including a random forest retrained at every step, did poorly. Here that failure plays out at a budget: over steps 43-49 the frozen detector finds 2 of 169 illicit transactions, and five of the seven steps come back at zero. The 50% at step 46 is one transaction out of two.

The information needed to recover exists. The same detector refitted on every label up to step 44, including the two steps after the break, finds 49.6% of the illicit transactions in steps 45-49 at a 10% budget, against 3.3% frozen and 5.0% for random picks. In operation those post-break labels only arrive if somebody investigates, so closing that gap means spending budget on cases the broken detector ranks low.

| budget | frozen on t≤34 | refit on t≤44 | random |
|---|---|---|---|
| 2% | 1.7% | 5.0% | 1.7% |
| 5% | 1.7% | 18.2% | 5.8% |
| 10% | 3.3% | 49.6% | 5.0% |
| 20% | 5.8% | 76.9% | 27.3% |

Recall on steps 45-49 (121 illicit transactions). Scored on every labelled transaction after step 34, before and after the break pooled, the frozen detector's AUPRC is 0.8075 (AUROC 0.9374). A single held-out score looks healthy while the steps after the break collapse.

Reproduce with `python scripts/fetch_elliptic.py --convert <dir>` and then `python scripts/measure_collapse.py`, which writes `results/collapse.log` and `results/collapse.json` in about 25 s on a laptop and prints the archive's sha256 to check against the committed one. The numbers are sensitive to the last bits of the features: an archive parsed with a different float routine moves them slightly, which is why the converter parses deterministically and the sha256 is recorded.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/JayeshSuryavanshi/caseload/main/docs/img/tradeoff-dark.png">
  <img src="https://raw.githubusercontent.com/JayeshSuryavanshi/caseload/main/docs/img/tradeoff-light.png" alt="Scatter of recall before the break against recall after it, IQM over six training seeds on held-out drift episodes. top-k sits at 97% before and 4% after. More exploration walks left and slightly up, ending with eps-explore(0.6) at 69% before and 15% after. The yield-triggered rule sits at 95% and 11%. PPO sits at 97% before and 20% after, with a seed interval of 14% to 25% post-break. The top-right corner is empty.">
</picture>

### Baselines, 30 seeds, IQM with 95% bootstrap CIs

Procedural drift episodes, 10% budget, paired against top-k:

| policy | overall | pre-break | post-break |
|---|---|---|---|
| top-k (always the highest scores) | 0.598 | **0.958** | 0.054 |
| eps-explore(0.15) | 0.580 | 0.923 | 0.093 |
| eps-explore(0.35) | 0.546 | 0.838 | 0.138 |
| eps-explore(0.6) | 0.430 | 0.630 | **0.160** |
| yield-triggered | **0.600** | 0.930 | 0.116 |
| random | 0.096 | 0.095 | 0.101 |
| random-action | 0.363 | 0.527 | 0.105 |
| oracle ceiling | **0.988** | 0.987 | 0.996 |

Two things to take from this. Post-break, **even random beats top-k** (+0.043, CI [+0.008, +0.073]); always exploiting a broken detector is worse than not using it. And overall, the hand-written adaptive rule is **not** significantly better than top-k (+0.005, CI [-0.005, +0.020], P(better) = 0.53) because its post-break gain is paid for pre-break. No fixed rule wins, and the oracle ceiling (0.988 overall) sits far above the best of them (0.600). That gap is the point of the environment.

Reproduce with `python scripts/run_baselines.py --out results/drift_baselines.json > results/drift_baselines.log` (30 seeds is the default).

### Training on drift, evaluating on the real break

Elliptic has one regime break, so it is one trajectory and a policy fitted to it has memorised it. Policies train on `caseload.envs.drift`, which draws random break positions and fraud-mode geometry, and are evaluated zero-shot on the real break.

Building that simulator took one correction worth recording. My first version let top-k heal itself after the break, recovering a large share of post-break fraud with no exploration at all, which is nothing like Elliptic's near-total collapse. The new fraud mode landed somewhere the old detector still scored middling, so exploit picks stumbled into it and fed the refit. The break is now **adversarial**: candidate fraud modes are screened against a detector fitted on pre-break traffic and the lowest-scoring are chosen. That reproduces the real collapse, and it is also the more realistic story, since an adversary who adapts moves to where the current model is not looking. (I no longer quote a number for the non-adversarial version, because no committed script produces it.)

### What the learned policy actually did

PPO trained on 1,600 procedural drift episodes per seed, **n = 6 training seeds**, then evaluated on 20 held-out drift episodes and, without further training, on the real Elliptic break.

**In distribution it beats everything, including the hand-written rule, on overall recall.** IQM over each seed's 20 episodes, then over the six seeds (`results/fleet_aggregate.log`):

| policy | overall | post-break | vs top-k |
|---|---|---|---|
| **ppo** (n = 6 seeds) | **0.655** [0.635, 0.670] | **0.197** [0.138, 0.251] | **+0.076 [+0.056, +0.091]** |
| top-k | 0.579 | 0.042 | (reference) |
| eps-explore(0.15) | 0.590 | 0.088 | +0.012 |
| eps-explore(0.35) | 0.553 | 0.116 | -0.025 |
| eps-explore(0.6) | 0.455 | 0.152 | -0.124 |
| yield-triggered | 0.582 | 0.105 | +0.004 |
| oracle ceiling | 0.997 | | |

Paired against the hand-written adaptive rule on overall recall: **+0.0725 [+0.0526, +0.0877]**, so it clears it. Post-break it finds 0.197 against top-k's 0.042 and the rule's 0.105. The oracle row is the ceiling's overall recall with the same statistic as the other rows, and is the same for every seed; the fleet files do not split it into pre- and post-break.

Read those intervals correctly. The baselines are deterministic and do not depend on the training seed, so their intervals are degenerate; the interval on `ppo` is a bootstrap over its six training seeds, which answers "does the method beat the baseline across initialisations" rather than giving an episode-level interval. Six is a small sample for that question.

**Zero-shot on the real break, the advantage does not show up.** Elliptic, 15 rounds with the break at round 8, 1,083 illicit transactions (`results/fleet/eval_s*.log`). Baselines are the same in every seed's log; the `ppo` row is the range over the six training seeds, not an interval:

| policy | overall | pre-break | post-break |
|---|---|---|---|
| ppo (range over 6 seeds) | 0.705 to 0.759 | 0.819 to 0.854 | 0.051 to 0.286 |
| top-k | 0.730 | 0.851 | 0.077 |
| eps-explore(0.15) | 0.720 | 0.837 | 0.089 |
| eps-explore(0.35) | 0.713 | 0.802 | 0.233 |
| eps-explore(0.6) | 0.659 | 0.714 | 0.361 |
| yield-triggered | **0.754** | 0.818 | **0.408** |
| oracle ceiling | 1.000 | 1.000 | 1.000 |

On overall recall, the hand-written rule's 0.754 is ahead of five of the six seeds (0.705 to 0.732). The sixth reaches 0.759, level with it: its own spread over policy randomness is 0.729 to 0.785, and the rule's is 0.742 to 0.764. Three seeds are below top-k's 0.730 overall. On post-break recall all six seeds are below the hand-written rule's 0.408 and below eps-explore(0.6)'s 0.361, and three are below top-k's 0.077.

So the honest statement is narrower than either "RL works here" or "RL does not". On the simulator it trained on, the learned policy beats every baseline on overall recall (n = 6 seeds). On the real regime break that advantage is not visible: no seed is clearly ahead of the hand-written rule on either overall or post-break recall, and on post-break recall every seed is behind the heaviest fixed exploration rule. More training bought in-distribution skill, not transfer.

Holding Elliptic out is what makes that visible. Training on its single trajectory would have produced a number that looked like a win.

**How much of this is the archive.** These numbers come from an archive built with `scripts/fetch_elliptic.py --convert`; each log prints its sha256 (`c9bedd7e...`, the same as `results/collapse.log`). An earlier version of this table was computed on an archive from another parser whose features differ from it only in the last bits. With the same checkpoints and code, that archive gives yield-triggered 0.231 post-break instead of 0.408, eps-explore(0.6) 0.560 instead of 0.361, and seed 0 0.440 instead of 0.286; overall recall moves by up to 0.042 (`results/legacy_archive/eval_s*.log`). The in-distribution drift numbers are unaffected. On Elliptic, then, a difference smaller than about 0.2 in post-break recall, or about 0.05 overall, between two policies on this single trajectory is within what the last bits of the features can move, and should not be read as a difference between policies. That applies to the paragraph above as much as to the earlier version: on the older archive the rule was at 0.712 overall and five of the six seeds were ahead of it. What held on both archives is the post-break ordering, every seed below eps-explore(0.6) and most below the hand-written rule, though on the canonical archive by margins inside that range. Nothing on Elliptic supports a claim that the learned policy transfers, and nothing on it is precise enough to rank the fixed rules finely either.

### A correction to an earlier version of this README

An earlier commit reported that PPO "does not transfer" and, separately, that it never separated from the hand-written rule in distribution. The second half was wrong, and wrong for a reason worth recording: that run was **320 episodes on one seed**, which is not a serious training run. At 1,600 episodes across six seeds the in-distribution picture inverts and PPO clears every baseline. The files from that run are kept in `results/superseded_320ep/` so the correction can be checked; do not cite them.

The transfer finding survived the larger run, but as an absence of evidence for transfer rather than a measured loss: on one trajectory, the gaps between policies are mostly smaller than what the Elliptic archive alone can move (see above). If you are citing a number from this repository, take it from this section rather than from the commit history.

---

## 2. Capacity-constrained triage on FiFAR (`caseload.envs.fifar`, `caseload.triage`)

Feedzai's FiFAR (Alves et al., Scientific Data, 2025) records, for each of 30,622 **synthetic** bank-account-opening fraud alerts, what each of **50 synthetic analysts would have decided**. The alerts are the applications a LightGBM alert model flags in the Bank Account Fraud (BAF) base dataset (Jesus et al., NeurIPS 2022 Datasets and Benchmarks), which is itself privacy-preserving synthetic data. The analysts are generated by the paper's OpenL2D framework, and their error rates are generator parameters, not measurements of people. What makes the dataset worth building on is that routing to an analyst is an exact table lookup, not a model of a reviewer, and the analysts are fallible in parameterised, heterogeneous ways.

### The shipped testbed has no sequential structure

Each of the 25 test scenarios is a **single batch** of 4,457 alerts with review capacity for 90.9% of them. The 25 training scenarios split their alerts into 7 variable-size batches with capacity equal to **100%** of alerts. There is no horizon to ration across and the constraint is nearly slack, which is why the dataset's own baseline, DeCCaF (Alves et al., TMLR 2024), solves each batch to optimality with constraint programming and does well. `caseload.triage` therefore exposes capacity tightness as a swept parameter. (`python -c "from caseload.envs import fifar; print(fifar.describe())"` prints the batch counts and capacity ratios; the alert and analyst counts are asserted by the tests.)

### The cost regime, stated correctly

FiFAR's paper (Eq. 8) and DeCCaF (Sec. 4.1 and Eq. 21) both derive the cost ratio from the alert model's threshold: a false alarm costs **λ<sub>t</sub> = 0.057** of a missed fraud, about **17.5:1**. That is the regime the synthetic analysts were generated for: each was sampled to cost less than blocking every alert at λ<sub>t</sub>. DeCCaF also runs λ<sub>t</sub>/5 (about 88:1) and 5λ<sub>t</sub> (about 3.5:1) to test other cost structures, and says those "are not strictly comparable". They are stress cases, not the benchmark's regime.

**At the stated regime, sending alerts to the analysts saves money.** Sending 10% of alerts to the analysts, paced evenly and chosen nearest the model's decision threshold, cuts expected cost by **3.2% [2.2, 4.2]** in-sample and **7.6% [4.0, 11.2]** on held-out month-8 alerts. All six routing policies in the sweep save money in-sample at λ<sub>t</sub> (+1.7% to +3.9%, every interval above zero). On the shipped data, all 50 of 50 analysts beat blocking every alert at λ<sub>t</sub>.

Sweeping the cost ratio at a 10% review budget, with the automated threshold refit to be cost-optimal at every ratio (even pacing, near-threshold routing; in-sample = the 25 training scenarios over the same 26,165 alerts; held-out = threshold fitted on months 4-7 and applied to month 8, 5 teams; 95% intervals resample scenarios and alerts together):

| missed fraud : false alarm | analysts beating block-all | saving (95% CI) | saving per review (95% CI) | held-out saving (95% CI) |
|---|---|---|---|---|
| 87.7:1 (λ<sub>t</sub>/5, stress) | 6/50 | -13.27% [-17.92, -8.86] | -0.0134 [-0.0181, -0.0090] | -17.17% [-31.99, -4.76] |
| 50:1 | 10/50 | -3.54% [-6.20, -1.05] | -0.0063 [-0.0110, -0.0019] | -6.10% [-14.59, +1.05] |
| 33.3:1 | 24/50 | +0.75% [-1.04, +2.44] | +0.0020 [-0.0028, +0.0065] | -1.20% [-6.95, +3.65] |
| 25:1 | 40/50 | +2.70% [+1.29, +4.07] | +0.0095 [+0.0046, +0.0144] | +3.70% [-0.97, +8.11] |
| **17.5:1 (λ<sub>t</sub>, stated)** | **50/50** | **+3.20% [+2.22, +4.17]** | **+0.0160 [+0.0110, +0.0208]** | **+7.57% [+4.02, +11.23]** |
| 10:1 | 50/50 | +5.61% [+4.75, +6.49] | +0.0419 [+0.0354, +0.0488] | +4.56% [+2.20, +7.00] |
| 3.5:1 (5λ<sub>t</sub>, stress) | 50/50 | +4.53% [+3.21, +5.87] | +0.0467 [+0.0328, +0.0612] | +9.47% [+6.30, +12.71] |
| 2:1 | 50/50 | +0.32% [-1.43, +2.09] | +0.0035 [-0.0155, +0.0229] | +3.11% [-0.58, +6.56] |

**Where review stops paying.** In-sample, review is significantly negative at 50:1 and above, indistinguishable from zero at 33:1, and significantly positive from 25:1 down to 3.5:1. So the sign changes somewhere between about 50:1 and 25:1. It is a band, not a sharp threshold, and it tracks the share of analysts who beat blocking every alert (6, 10, 24, 40 and 50 of 50 across the first five rows). The held-out intervals are wider: only the 88:1 stress case is significantly negative there. The value of review is also not monotone in the ratio: at 2:1 the refit threshold moves to 0.42 and review's saving falls back to +0.32% [-1.43, +2.09]. With the threshold left at 0.051 instead, the same cell shows +8.78%, and that difference is threshold re-tuning, not review.

**The 88:1 stress case.** There the refit threshold (0.0510) sits at the bottom of the alert scores, so the model alone blocks every alert and "near-threshold" routing sends the least suspicious ones. Review raises cost by 13.27% [8.86, 17.92] in-sample, and every one of the six policies loses (-4.94% to -13.89%). Held out, near-threshold routing loses 17.17% [4.76, 31.99], while top-score (-0.15% [-4.78, +1.98]) and mixed (-2.40% [-9.89, +2.78]) routing cannot be told apart from zero. The mechanism is arithmetic. For a blocked alert sent to a random analyst, review saves P(legit) x (1 - FPR) x λ and loses P(fraud) x FNR, averaged over analysts; the sweep logs that as -0.0121 per review at 88:1 against a measured -0.0134 [-0.0181, -0.0090], and +0.0155 at λ<sub>t</sub> against a measured +0.0160 [+0.0110, +0.0208]. This is a property of how the synthetic analysts were calibrated, not a finding about human review.

It also does not explain anything odd in DeCCaF. In DeCCaF's own results at λ<sub>t</sub>/5 (their Table 3), every method that routes to humans, random routing included, has lower expected cost than rejecting every alert. And DeCCaF regenerates its expert team for each cost setting so that the experts beat full rejection in that setting, while FiFAR's 50 analysts are generated once, at λ<sub>t</sub>. The two are not the same experiment.

Reproduce with `python scripts/cost_ratio_sweep.py > results/cost_ratio_sweep.log`, which also writes `results/cost_ratio_sweep.json`.

### Capacity and pacing at the stated regime

At λ<sub>t</sub> with a 10% budget, proportional pacing with near-threshold routing removes 3.3% [3.1, 3.5] of the model-only cost, 0.0175 per review. Per review, front-loaded and proportional pacing differ by at most 0.003 at any capacity from 2% to 100%, and the sign of the difference changes with the budget. That is small in absolute terms but not always small relative to what a review is worth: at a 5% budget front-loading gets 0.0119 per review against 0.0148, about a fifth less (-0.0030 [-0.0034, -0.0025]), and even pacing is also worse there (-0.0020 [-0.0043, -0.0003]); at 10% front-loading is better (+0.0018 [+0.0014, +0.0022]). Routing matters more than pacing. Under even pacing, near-threshold routing stays within 0.002 per review of the reference at every capacity, while top-score routing gives up 0.0045 to 0.0096. These intervals cover batch order and analyst team only, since the 25 training scenarios share their alerts.

Reproduce with `python scripts/capacity_sweep.py > results/capacity_sweep.log`.

### Three corrections to my own experiments

Recorded because each would have produced a confident wrong headline, and the third one killed a finding I had already written down.

1. **A straw-man baseline.** My first sweep showed front-loaded pacing beating "even" pacing at every capacity level including 100%, which is impossible if the comparison is fair. Even pacing divides capacity by *remaining batch count* while FiFAR's batches vary a lot in size, so it under-reviews the big ones. The reference is now proportional-to-size allocation.
2. **A spend confound.** Policies were not spending the same budget, and the saving ranking was *exactly* the spend ranking. Unused slots were being wasted by grid quantisation, so the winning strategy was just "burn budget fast". Capacity now carries forward and the headline metric is `saving_per_review`. (Spend still differs at full capacity: front-loaded uses 100%, proportional 93.9%, even 85.3%.) I then wrote that front-loading was "clearly worse" per review. That came from code that counted reviews no analyst made (see the next section); with that fixed, front-loaded and proportional pacing differ by at most 0.003 per review, and the sign changes with the budget.
3. **A wrong threshold, which invalidated a headline.** I ran with a 0.5 decision threshold on alerts that are already everything scoring above 0.051. The model therefore flagged almost none of them, missed almost all the fraud, and human review looked enormously valuable: I had measured routing to near-threshold cases as **10x** better per review than routing to the most suspicious, and had written that down as the finding. At a cost-optimal threshold it evaporates: at λ<sub>t</sub> near-threshold routing saves +0.0160 per review against +0.0085 for top-score, and at 3.5:1 top-score is slightly ahead. The lesson is the one the number now carries: any claim about human review has to be made against a cost-optimal automated threshold, or it is measuring threshold re-tuning. The cost-ratio section above is where I learned it a second time.

---

## Corrections found before release

Before making this repository public I had every claim re-checked against the code, the committed results and the cited papers. That found eight more things I had wrong. All are fixed in the text above; this is the list.

1. **I misattributed the cost regime.** I called 88:1 "FiFAR's stated regime" and "DeCCaF's realistic setting". It is DeCCaF's λ<sub>t</sub>/5 stress case; the stated regime is λ<sub>t</sub> = 0.057, about 17.5:1, where review saves money. My headline was that routing to a human raises cost; at the stated regime it does the opposite. I also claimed this "independently explains a published oddity" in DeCCaF's results, which misread their table, and that the realistic cell of the benchmark is degenerate. Both claims are withdrawn.
2. **The cost-ratio table came from an ad hoc run.** It was one scenario at a fixed 0.051 threshold, and no committed script produced it, although the README pointed readers at one that did not vary the cost at all. It is now `scripts/cost_ratio_sweep.py`: all 25 training scenarios plus a held-out month, the threshold refit per ratio, and bootstrap intervals. Most of the value it had shown for review at low ratios was threshold re-tuning, and the sharp "about 35x" crossover became a band between about 50:1 and 25:1.
3. **The Elliptic headline spliced two configurations, and the figures were typed by hand.** The pre-break range came from the script's default (300 boosting iterations) and "exactly zero for seven consecutive steps" from a 30-iteration run; the recovery numbers matched neither. The figure script hard-coded its values while its docstring said it never did, and the tradeoff figure mixed statistics, plotting PPO's mean against hard-coded IQMs for the baselines' post-break recall. `scripts/measure_collapse.py` now writes `results/collapse.{log,json}` at the default, the figures read only committed result files, and the post-break result is "2 of 169", not "exactly zero".
4. **I called FiFAR real.** Its alerts and its analysts are both synthetic, and the analysts' error rates are generator parameters, so "measured" fallibility and "real fraud alerts" were wrong.
5. **Phantom reviews.** With capacity carry-over on, the triage environment counted picks that found no analyst with room as reviews, which inflated the per-review figures and drove the "front-loading is clearly worse" claim. It now counts only alerts an analyst decided, unused slots roll forward without borrowing from batches that have not arrived, and a regression test checks that every counted review is an analyst decision. `results/capacity_sweep.{log,json}` were stale against the code and have been regenerated.
6. **The "mixed" routing band was not what it said.** Its "top-score" half picked the lowest row indices rather than the highest scores, because `np.setdiff1d` sorts its output. Every earlier mixed-band number was affected.
7. **A stale rule contradicted the corrected PPO result.** The reporting rules still said PPO had not cleared the hand-written rule after the six-seed run showed it does in distribution.
8. **The zero-shot table came from an archive nobody else could build.** It was computed on my own older Elliptic archive, and I had written that a freshly converted one "may not match to the digit". On the converted archive the post-break numbers moved by up to 0.2, more than most of the gaps I was interpreting, and the one seed I had called ahead of every baseline overall is level with the hand-written rule. The table is now regenerated on the converted archive, and the section says how much the archive alone moves it.

---

## Reporting rules

Several of the results above were wrong before they were right, and each correction is written up where it happened. These rules are in the README because they are easy to skip when a run finishes and the number looks good.

- **Never report a mean without an interval.** `caseload.evaluation` gives interquartile mean with a stratified bootstrap, and **paired** differences, because episode difficulty varies far more than the gap between policies.
- **Always report against the oracle ceiling.** Recall under a budget means nothing without knowing what was achievable with that budget.
- **The opponent is `yield-triggered`, not `random`.** Beating random proves nothing here. The hand-written adaptive rule is the bar: PPO clears it in distribution on overall recall, and zero-shot on the real Elliptic break no seed is clearly ahead of it on overall or post-break recall.
- **Elliptic is one trajectory.** Its numbers are a spread over policy randomness, never a confidence interval over episodes.
- **Never train on Elliptic.** One regime break is one episode, so a policy fitted to it has memorised it. Training seeds live in a range disjoint from the evaluation seeds.
- **Make claims about human review against a cost-optimal threshold, at the cost ratio the data was built for.** Anything else measures threshold re-tuning or a stress case.

## Install

```bash
pip install "caseload[agents,fifar]"
```

or the latest code from GitHub, `pip install "caseload[agents,fifar] @ git+https://github.com/JayeshSuryavanshi/caseload"`, or from a clone:

```bash
pip install -e '.[agents,fifar]'
```

The core needs only numpy, gymnasium and scikit-learn, and the drift simulator needs no download. `agents` adds torch for the learned policies; `fifar` adds pandas and pyarrow for the FiFAR loader.

## Reproducing

### Environment

```bash
git clone https://github.com/JayeshSuryavanshi/caseload && cd caseload
uv venv -p 3.13
uv pip install -e ".[agents,fifar,dev]" -c requirements-lock.txt
export PYTHON=.venv/bin/python        # the scripts/*.sh runners use $PYTHON, else python on PATH
```

With plain pip: `python3.13 -m venv .venv && .venv/bin/pip install -e ".[agents,fifar,dev]" -c requirements-lock.txt`.

`requirements-lock.txt` pins the exact package versions behind the committed results. It was locked on CPython 3.13.9 on macOS arm64 (Apple M1 Pro); use it as a constraints file (`-c`). The version ranges in `pyproject.toml` are what CI tests on Python 3.10, 3.12 and 3.13. The lock does not include matplotlib, which `scripts/make_figures.py` needs (`pip install matplotlib`).

`pytest tests -q` runs the test suite. Tests that need FiFAR or Elliptic skip themselves when the data is absent, as it is in CI.

### Getting the data

```bash
python scripts/fetch_elliptic.py                  # says where to get Elliptic
python scripts/fetch_elliptic.py --convert <dir>  # builds ~/.cache/caseload/elliptic_parsed.npz from the three Kaggle CSVs
python scripts/fetch_fifar.py                     # downloads FiFAR, checks size and md5, unpacks to ~/.cache/caseload/fifar/FiFAR
```

### Commands behind each committed artifact

Run from the repository root.

| Artifact | Command |
|---|---|
| `results/collapse.{log,json}` | `python scripts/measure_collapse.py` (needs Elliptic, about 25 s) |
| `results/drift_baselines.{json,log}` | `python scripts/run_baselines.py --out results/drift_baselines.json > results/drift_baselines.log` (no data needed, 886 s in the committed log) |
| `results/fleet/ppo_s{0..5}.pt`, `ppo_s{0..5}_log.json`, `train_s{0..5}.log`, `results/fleet_run.log` | `./scripts/run_fleet.sh 6 200 > results/fleet_run.log`. Each worker runs `python scripts/train.py --seed S --out results/fleet/ppo_sS.pt`, whose defaults are the fleet settings: 200 updates x 8 episodes = 1,600 episodes, 18 rounds, 10% budget. |
| `results/fleet/eval_s{0..5}.json`, `eval_s{0..5}.log`, `results/fleet_eval.log` | `./scripts/evaluate_fleet.sh > results/fleet_eval.log`. Each worker runs `python scripts/evaluate.py --model results/fleet/ppo_sS.pt`: 20 held-out drift seeds and 3 Elliptic policy seeds, writing `eval_sS.json` beside the model. Needs Elliptic for the zero-shot part, converted with `scripts/fetch_elliptic.py --convert`; each log prints the archive's sha256. |
| `results/legacy_archive/eval_s{0..5}.{json,log}` | `python scripts/evaluate.py --model results/fleet/ppo_sS.pt --elliptic <archive> --out results/legacy_archive/eval_sS.json > results/legacy_archive/eval_sS.log` for S = 0..5, with `<archive>` the older archive written by the graphspot library's Elliptic parser (sha256 in each log). This repository cannot rebuild that archive; the files are kept only to measure how far the archive alone moves the zero-shot numbers. |
| `results/fleet_aggregate.log` (the in-distribution PPO table) | `python scripts/aggregate_fleet.py results/fleet > results/fleet_aggregate.log` |
| `results/cost_ratio_sweep.{log,json}` | `python scripts/cost_ratio_sweep.py > results/cost_ratio_sweep.log` (needs FiFAR) |
| `results/capacity_sweep.{log,json}` | `python scripts/capacity_sweep.py > results/capacity_sweep.log` (needs FiFAR; 25 training scenarios, fp cost λ<sub>t</sub> = 0.057, threshold refit) |
| `docs/img/*.png`, `results/tradeoff_points.json` | `python scripts/make_figures.py` (reads `results/collapse.json` and `results/fleet/eval_s*.json`, and writes the tradeoff figure's plotted points to `results/tradeoff_points.json`; needs matplotlib) |
| `results/train_long.log`, `results/ppo_s0_log.json` | `./scripts/train_long.sh > results/train_long.log`, stopped once seed 0 had finished. Its checkpoint is not tracked. |
| `results/superseded_320ep/` | The retracted 320-episode single-seed run (see its README). Kept for the record; do not cite it. |

The committed `eval_sS.json` files were regenerated from the committed checkpoints with the command above, on a converted archive, and their drift results are identical to the previous run's. Retraining has not been checked for bit-identical weights.

Checkpoints are `{"state_dict": ..., "config": {...}}` with plain str, int, float and bool config values, and load with `torch.load(path, weights_only=True)`, so no pickled code runs.

## Running many seeds

The environment refits a scikit-learn ensemble every round, so training is CPU-bound and scales with cores rather than with a GPU. `run_fleet.sh` runs one single-threaded process per seed, because refits in one process using every core oversubscribe threads and fight each other. In the committed logs, seed 0 trained alone with every thread available took 6,392 s for 200 updates (`results/train_long.log`), while six seeds trained side by side, one thread each, on 8 cores took 3,861 to 3,925 s apiece (`results/fleet/train_s*.log`).

```bash
./scripts/run_fleet.sh 6 200           # 6 seeds in parallel, 200 updates each
./scripts/evaluate_fleet.sh
python scripts/aggregate_fleet.py results/fleet > results/fleet_aggregate.log
```

## Design notes

**Device.** CPU, everywhere, and there is no flag. The policies are small and each update is a small batch, a size at which Apple's MPS backend was slower than CPU when I tried it. That comparison is not scripted in this repository, so treat it as a note, not a result.

**Ceilings, not just baselines.** Recall under a budget is meaningless without knowing what was achievable, so `oracle_ceiling` reports what a perfect ranker would have caught with the same budget and every policy is read as a fraction of it.

**Intervals, always.** `caseload.evaluation` provides interquartile mean and stratified bootstrap intervals following `rliable` (Agarwal et al., NeurIPS 2021), plus **paired** differences, because episode difficulty varies far more than the gap between policies.

## Related work

- **Write-up.** [The simulator was the experiment](https://www.jayeshsuryavanshi.com/blog/simulator-was-the-experiment.html) is the narrative version: why the collapse motivates a sequential formulation, how the drift simulator had to be made adversarial before exploration mattered, and why a policy that dominates that simulator across six seeds still loses to a one-line heuristic on the real break.
- **Selective labels and partial feedback.** Lakkaraju et al. (KDD 2017) define the selective-labels problem, where the outcome is only observed for the cases a decision lets through. Kilbertus et al. (AISTATS 2020) show that in this setting a predictor learned only from the labelled data is suboptimal, and argue for learning decision policies, with an explicit link to the explore/exploit trade-off in RL. In fraud specifically, Dal Pozzolo et al. (TNNLS 2018) formalise the operating conditions of a real fraud-detection system, including verification latency, where only a small set of transactions is checked by investigators in time to supply labels. caseload is an RL benchmark instance of these problems: the investigation budget is the only source of labels, and the break makes it costly to spend that budget only where the detector is confident.
- **Monitored MDPs** (Parisi et al., AAMAS 2024) formalise environments where the reward exists whether or not the agent observes it. The drift environment is one, with observation paid for from the same budget as exploitation.
- **Tong et al.** (AAAI 2020) learn alert-prioritisation policies with adversarial RL, against attackers who adapt to the defender's policy, with case studies in fraud and intrusion detection. The adversarial break in `caseload.envs.drift` is a much simpler version of the same idea: the new fraud mode goes where the current detector is not looking.
- **Deng** (arXiv 2608.08577, 2026) treats fraud-operations automation as an authorisation decision under selective, delayed labels and shared review capacity, with evaluations that include Elliptic++. It is a decision-support framework built on randomised audits, where caseload is an environment for learning investigation policies.
- **Wu, Kher and Smyth** (arXiv 2605.27999, 2026) learn to assign prediction tasks to human or AI agents that can each handle only a fraction of tasks, with sequential explore-exploit policies. The triage environment is a capacity-constrained assignment problem of this kind, over FiFAR's synthetic analysts.
- **DeCCaF** (Alves et al., TMLR 2024) is FiFAR's own baseline and the source of cost-sensitive, capacity-constrained deferral to multiple experts on this data: supervised expert-loss models plus constraint programming, solved to optimality per batch. caseload does not improve on it; its FiFAR results are about when review pays at all and whether the shipped testbed leaves anything for a sequential policy.
- **FALCON** (Zhang et al., arXiv 2604.00904, 2026, [code](https://github.com/zhengzhang37/FALCON)) formulates learning to defer as a CMDP whose state includes cumulative human workload, so that human accuracy degrades with fatigue, trained with PPO-Lagrangian. It is the closest sequential, RL-trained deferral work I know of. It models a changing human, where caseload's drift environment models a changing fraud population and its triage environment a fixed pool of fallible analysts under a shared capacity. caseload does not train a policy on FiFAR.

## Data

Neither dataset is redistributed. The drift simulator needs no download, so the package works without either.

- **Elliptic**: from Weber et al. (2019), distributed on Kaggle under CC BY-NC-ND 4.0 (non-commercial use, no redistribution of modified versions). Download it yourself. `scripts/fetch_elliptic.py --convert` builds a local archive; that archive is a derivative of the dataset, so do not share it. The archive keeps 165 features per transaction.
- **FiFAR**: on figshare under CC BY 4.0 ([10.6084/m9.figshare.28351172](https://doi.org/10.6084/m9.figshare.28351172)). `scripts/fetch_fifar.py` downloads it, checks the size and md5 figshare publishes, and unpacks it to `~/.cache/caseload/fifar/FiFAR`. Cite Alves et al. (2025) if you use it.

Datasets are read from `~/.cache/caseload`. The package's former cache directories are read as fallbacks, as documented in the loaders.

Only 22.9% of Elliptic transactions carry a label, and unlabelled cases are left in, because investigating one costs budget and reveals nothing, which is what the operational problem looks like. `load_episode(drop_unlabelled=True)` removes them so that cost can be measured rather than assumed.

## References

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., and Bellemare, M. G. Deep Reinforcement Learning at the Edge of the Statistical Precipice. *Advances in Neural Information Processing Systems 34* (NeurIPS 2021), pp. 29304-29320. [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)
- Alves, J. V., Leitão, D., Jesus, S., Sampaio, M. O. P., Liébana, J., Saleiro, P., Figueiredo, M. A. T., and Bizarro, P. A benchmarking framework and dataset for learning to defer in human-AI decision-making. *Scientific Data* 12, 506 (2025). [doi:10.1038/s41597-025-04664-y](https://doi.org/10.1038/s41597-025-04664-y). Dataset: Financial Fraud Alert Review Dataset, figshare, [doi:10.6084/m9.figshare.28351172](https://doi.org/10.6084/m9.figshare.28351172)
- Alves, J. V., Leitão, D., Jesus, S., Sampaio, M. O. P., Liébana, J., Saleiro, P., Figueiredo, M. A. T., and Bizarro, P. Cost-Sensitive Learning to Defer to Multiple Experts with Workload Constraints. *Transactions on Machine Learning Research*, 2024. [arXiv:2403.06906](https://arxiv.org/abs/2403.06906)
- Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., and Bontempi, G. Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy. *IEEE Transactions on Neural Networks and Learning Systems* 29(8), pp. 3784-3797 (2018). [doi:10.1109/TNNLS.2017.2736643](https://doi.org/10.1109/TNNLS.2017.2736643)
- Deng, J. When Can Fraud Operations Authorize Automation? A Decision-Support Framework for Fresh Audit Evidence and Review Workload. Preprint, 2026. [arXiv:2608.08577](https://arxiv.org/abs/2608.08577)
- Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R. P., Gama, J., and Bizarro, P. Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation. *Advances in Neural Information Processing Systems 35* (NeurIPS 2022), Datasets and Benchmarks Track, pp. 33563-33575. [arXiv:2211.13358](https://arxiv.org/abs/2211.13358)
- Kilbertus, N., Gomez-Rodriguez, M., Schölkopf, B., Muandet, K., and Valera, I. Fair Decisions Despite Imperfect Predictions. *International Conference on Artificial Intelligence and Statistics* (AISTATS 2020), PMLR 108, pp. 277-287. [arXiv:1902.02979](https://arxiv.org/abs/1902.02979)
- Lakkaraju, H., Kleinberg, J., Leskovec, J., Ludwig, J., and Mullainathan, S. The Selective Labels Problem. *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (KDD 2017), pp. 275-284. [doi:10.1145/3097983.3098066](https://doi.org/10.1145/3097983.3098066)
- Parisi, S., Mohammedalamen, M., Kazemipour, A., Taylor, M. E., and Bowling, M. Monitored Markov Decision Processes. *Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems* (AAMAS 2024), pp. 1549-1557. [arXiv:2402.06819](https://arxiv.org/abs/2402.06819)
- Patterson, A., Neumann, S., White, M., and White, A. Empirical Design in Reinforcement Learning. *Journal of Machine Learning Research* 25(318), pp. 1-63 (2024). [arXiv:2304.01315](https://arxiv.org/abs/2304.01315)
- Tong, L., Laszka, A., Yan, C., Zhang, N., and Vorobeychik, Y. Finding Needles in a Moving Haystack: Prioritizing Alerts with Adversarial Reinforcement Learning. *Proceedings of the AAAI Conference on Artificial Intelligence* 34(01), pp. 946-953 (2020). [doi:10.1609/aaai.v34i01.5442](https://doi.org/10.1609/aaai.v34i01.5442)
- Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., and Leiserson, C. E. Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. KDD '19 Workshop on Anomaly Detection in Finance, 2019. [arXiv:1908.02591](https://arxiv.org/abs/1908.02591)
- Wu, S., Kher, S., and Smyth, P. Learning to Assign Prediction Tasks to Agents with Capacity Constraints. Preprint, 2026. [arXiv:2605.27999](https://arxiv.org/abs/2605.27999)
- Zhang, Z., Nguyen, C. C., Rosewarne, D., Wells, K., and Carneiro, G. Fatigue-Aware Learning to Defer via Constrained Optimisation. Preprint, 2026. [arXiv:2604.00904](https://arxiv.org/abs/2604.00904)

## License

BSD 3-Clause. The datasets keep their own licences (see [Data](#data)).
