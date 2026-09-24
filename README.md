# caseload

Reinforcement learning environments for **budgeted fraud investigation**: deciding where to point scarce human attention, and when, while the thing you are hunting moves.

Two environments, one shared evaluation harness. Both are small enough that a 30-seed study runs on a laptop, which is deliberate: Patterson et al. (JMLR 2024) show 10 runs is not enough to estimate a mean reliably, so a cheap environment measured 30 times is a stronger claim than an expensive one measured 3 times.

Every number below is produced by a script in this repository. Where a result is negative, or where my own first attempt was wrong, that is written down rather than removed.

---

## 1. Selective labels under a regime break (`caseload.envs.drift`, `caseload.envs.elliptic`)

Investigating a case reveals its label. Nothing else does. So the detector is only ever retrained on cases somebody chose to look at, and a policy that always spends its budget on the highest-scoring cases stops learning the moment the fraud population moves, and never finds out that it has.

This is the selective-labels setting (Lakkaraju et al., KDD 2017) as a sequential problem, and an instance of a **Monitored MDP** (Parisi et al., AAMAS 2024), where the reward exists whether or not you observe it and observing it competes for the same budget as exploiting it. Published Monitored MDP experiments are gridworlds.

### The motivating measurement

On the Elliptic Bitcoin transaction graph, a gradient boosting detector fitted on time steps 1-34 and frozen, at a 2% per-step budget:

| step | 35 | 38 | 41 | 42 | **43** | 44 | 45 | 46 | 47 | 48 | 49 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recall | 59% | 46% | 54% | 38% | **0%** | 0% | 0% | 0% | 0% | 0% | 0% |

Step 43 is a documented regime break. The detector does not degrade; it finds **exactly zero** illicit transactions for seven consecutive steps. And the information needed to recover exists: the same detector given labels from steps 43-44 reaches **56.2%** recall on 45-49 at a 10% budget, against **1.7%** frozen. Closing that gap requires spending budget on cases the broken detector ranks low.

Reproduce with `python scripts/measure_collapse.py`.

### Baselines, 30 seeds, IQM with 95% bootstrap CIs

Procedural drift episodes, 10% budget, paired against top-k:

| policy | overall | pre-break | post-break |
|---|---|---|---|
| top-k (the industry default) | 0.598 | **0.958** | 0.054 |
| eps-explore(0.15) | 0.580 | 0.923 | 0.093 |
| eps-explore(0.35) | 0.546 | 0.838 | 0.138 |
| eps-explore(0.6) | 0.430 | 0.630 | **0.160** |
| yield-triggered | **0.600** | 0.930 | 0.116 |
| random | 0.096 | 0.095 | 0.101 |
| random-action | 0.363 | 0.527 | 0.105 |
| oracle ceiling | **0.988** | 0.987 | 0.996 |

Two things to take from this. Post-break, **even random beats top-k** (+0.043, CI [+0.008, +0.073]); always exploiting a broken detector is worse than not using it. And overall, the hand-written adaptive rule is **not** significantly better than top-k (+0.005, CI [-0.005, +0.020], P(better) = 0.53) because its post-break gain is paid for pre-break. No fixed rule wins, and the oracle sits 39 points above the best of them. That gap is the point of the environment.

Reproduce with `python scripts/run_baselines.py --seeds 30`.

### Training on drift, evaluating on the real break

Elliptic has one regime break, so it is one trajectory and a policy fitted to it has memorised it. Policies train on `caseload.envs.drift`, which draws random break positions and fraud-mode geometry, and are evaluated zero-shot on the real break.

Building that simulator took one correction worth recording. My first version let top-k self-heal to 27.7% post-break recall, nothing like Elliptic's seven zeros, because the new fraud mode landed somewhere the old detector still scored middling, so exploit picks stumbled into it and fed the refit. The break is now **adversarial**: candidate fraud modes are screened against a detector fitted on pre-break traffic and the lowest-scoring are chosen. That reproduces the real collapse, and it is also the more realistic story, since an adversary who adapts moves to where the current model is not looking.

### What the learned policy actually did

PPO trained on 320 procedural drift episodes, then evaluated on 20 held-out drift seeds and, without
any further training, on the real Elliptic break.

On drift it learned the behaviour the environment was built to reward: it keeps almost all of top-k's
pre-break recall (0.941 against 0.953) while nearly doubling post-break recall (0.134 against 0.071).
Paired against each opponent over the same 20 seeds:

| comparison | delta (IQM, 95% CI) | P(better) | verdict |
|---|---|---|---|
| overall vs top-k | +0.010 [+0.001, +0.025] | 0.59 | significant, and small |
| overall vs eps-explore(0.15) | +0.021 [+0.006, +0.033] | 0.58 | significant |
| overall vs **yield-triggered** | +0.004 [-0.011, +0.041] | 0.59 | **not significant** |
| post-break vs top-k | +0.042 [+0.024, +0.069] | 0.74 | significant |
| post-break vs **eps-explore(0.6)** | **-0.043** [-0.066, -0.013] | 0.36 | **significantly worse** |

So in its own training distribution the learned policy beats the industry default by about one point
on a zero-to-one scale, never separates from a hand-written rule anyone could write in an afternoon,
and is beaten post-break by the crudest possible policy, which simply always spends 60% of its budget
on exploration.

**And it does not transfer.** Zero-shot on Elliptic's real regime break, 15 rounds, break at round 8:

| policy | overall | pre-break | post-break |
|---|---|---|---|
| ppo | 0.718 | 0.843 | **0.043** |
| top-k | **0.730** | 0.851 | 0.077 |
| eps-explore(0.15) | 0.720 | 0.837 | 0.089 |
| eps-explore(0.35) | 0.713 | 0.803 | 0.223 |
| eps-explore(0.6) | 0.690 | 0.714 | **0.560** |
| yield-triggered | 0.712 | 0.801 | 0.231 |
| oracle ceiling | 1.000 | 1.000 | 1.000 |

On the real break the learned policy is worse than plain top-k overall, and its post-break recall of
0.043 is **thirteen times worse** than the fixed 60%-exploration rule's 0.560. Whatever it learned to
recognise in the simulator, the real collapse does not present it. Holding Elliptic out was therefore
the load-bearing design decision in this repository: training on it would have produced a number that
looked like a win.

The honest summary is that this environment is not solved, the oracle sits 27 points above the best
policy even on Elliptic, and the current state of the art on it is a one-line heuristic.

---

## 2. Capacity-constrained triage on FiFAR (`caseload.envs.fifar`, `caseload.triage`)

Feedzai's FiFAR (Nature Scientific Data, April 2025) records, for each of 30,622 real bank-account-fraud alerts, what each of **50 analysts would have decided**. Routing to a human is an exact table lookup, not a model of a human, and the human is fallible in measured, heterogeneous ways: analyst false-negative rates run 0.015 to 0.312, false-positive rates 0.015 to 0.759.

### What I found, and it is negative

**The shipped testbed has no sequential structure.** Each of the 25 test scenarios is a **single batch** of 4,457 alerts carrying 4,052 review slots, which is 90.9% of the alerts. The 25 training scenarios have 7 variable-size batches with capacity equal to **100%** of alerts. There is no horizon to ration across and the constraint is nearly slack, which is why the dataset's own baseline solves each batch to optimality with constraint programming and does well.

**At FiFAR's own cost regime, sending an alert to a human raises expected cost.** DeCCaF's realistic setting puts a missed fraud at roughly 88x a false alarm. Under that ratio the cost-optimal automated policy is to block every alert: the fitted threshold lands on FiFAR's own alert threshold of 0.051 with **zero** false negatives. There is no missed fraud left for a reviewer to find, so review can only rescue false positives, and it is not good enough at it. Measured at a 10% review budget, review moves total cost from 265.9 to 296.8, a **11.6% increase**.

The arithmetic is simple enough to state exactly, and it matches the measurement to three decimals. Per review:

```
lose = P(fraud) x analyst_FNR x fn_cost      = 0.121 x 0.157 x 1.0 = 0.0190
gain = P(legit) x (1 - analyst_FPR) x fp_cost = 0.879 x 0.689 x fp_cost
```

At 88:1 that predicts -0.0121 per review; the environment measures -0.0118. A 15.7% analyst miss rate against a 12.1% fraud base rate costs more than the false alarms those analysts correctly clear.

**So the useful question is not "which policy wins" but "when is review worth doing at all".** Sweeping the cost ratio at a 10% budget:

| missed fraud : false alarm | saving per review | total cost saving |
|---|---|---|
| **88:1** (FiFAR's stated regime) | -0.0118 | **-11.6%** |
| 33:1 | +0.0007 | +0.25% |
| 17:1 | +0.0209 | +3.9% |
| 10:1 | +0.0477 | +5.4% |
| 2:1 | +0.3167 | +7.2% |

Human review on this data becomes worthwhile only once a missed fraud costs **less than about 35x** a false alarm. Above that, the right policy is to automate and not route anything, which independently explains a published oddity: DeCCaF scores 0.79 +/- 0.04 against Random's 0.80 +/- 0.08 in exactly this regime. The realistic cell of this benchmark is degenerate for human-in-the-loop methods, so no amount of reinforcement learning will win it.

Reproduce with `python scripts/capacity_sweep.py`.

### Three corrections to my own experiments

Recorded because each would have produced a confident wrong headline, and the third one killed a finding I had already written down.

1. **A straw-man baseline.** My first sweep showed front-loaded pacing beating "even" pacing at every capacity level including 100%, which is impossible if the comparison is fair. Even pacing divides capacity by *remaining batch count* while FiFAR's batches run 5,000 down to 400, so it under-reviews the big ones. The reference is now proportional-to-size allocation.
2. **A spend confound.** Policies were not spending the same budget: front-loaded used 100% of capacity, proportional 93.9%, even 85.4%, and the saving ranking was *exactly* the spend ranking. Unused slots were being wasted by grid quantisation, so the winning strategy was just "burn budget fast". Capacity now carries forward and the headline metric is `saving_per_review`. With both fixed, front-loading is clearly **worse**, the opposite of what the buggy version said.
3. **A wrong threshold, which invalidated a headline.** I ran with a 0.5 decision threshold on alerts that are already everything scoring above 0.051. The model therefore flagged only 1.9% of them, missed almost all the fraud, and human review looked enormously valuable: I had measured routing to near-threshold cases as **10x** better per review than routing to the most suspicious, and had written that down as the finding. At the correct threshold it evaporates. Every routing band is net-negative, and the ranking between them is noise against a much larger effect. The lesson is the one the number now carries: any claim about human review has to be made against a cost-optimal automated threshold, or it is measuring threshold re-tuning.

---

## Reporting rules

Four results in this repository were wrong before they were right, and each was caught by
one of these. They are in the README because they are easy to skip when a run finishes and
the number looks good.

- **Never report a mean without an interval.** `caseload.evaluation` gives interquartile
  mean with a stratified bootstrap, and **paired** differences, because episode difficulty
  varies far more than the gap between policies.
- **Always report against the oracle ceiling.** Recall under a budget means nothing without
  knowing what was achievable with that budget.
- **The opponent is `yield-triggered`, not `random`.** Beating random proves nothing here.
  The hand-written adaptive rule is the bar, and PPO has not cleared it.
- **Elliptic is one trajectory.** Its numbers are a spread over policy randomness, never a
  confidence interval over episodes.
- **Never train on Elliptic.** One regime break is one episode, so a policy fitted to it has
  memorised it. Training seeds live in a range disjoint from the evaluation seeds.

## Running many seeds

The environment refits a scikit-learn ensemble every round, so this is CPU-bound and
scales with cores rather than with a GPU. One single-threaded process per seed is
**2.3x faster per update** than one process using every core (13 s against 30 s measured on
an M1 Pro), because the refits otherwise oversubscribe threads and fight each other.

```bash
./scripts/run_fleet.sh 6 200           # 6 seeds in parallel, 200 updates each
./scripts/evaluate_fleet.sh
./.venv/bin/python scripts/aggregate_fleet.py results/fleet
```

## Install

```bash
pip install caseload                # environments and baselines
pip install "caseload[agents]"      # adds torch for the learned policies
pip install "caseload[fifar]"       # adds pandas/pyarrow for the FiFAR loader
```

## Reproducing

```bash
python scripts/fetch_elliptic.py     # prints where to get the data
python scripts/fetch_fifar.py        # downloads FiFAR (209 MB, CC BY) and verifies md5
python scripts/measure_collapse.py   # the Elliptic collapse table
python scripts/run_baselines.py      # drift baselines, 30 seeds, IQM + CIs
python scripts/capacity_sweep.py     # the FiFAR capacity and routing sweep
python scripts/train.py              # train PPO on drift
python scripts/evaluate.py           # zero-shot transfer to Elliptic
```

## Design notes

**Device.** CPU, everywhere, and there is no flag. Measured on an M1 Pro, CPU beats MPS by 4.7x at batch 64 and 2.7x at batch 256; MPS only wins above roughly 1,000 rows per update, which these policies never reach. Single-observation CPU forward is 21 µs, about 47,000 actions/sec.

**Ceilings, not just baselines.** Recall under a budget is meaningless without knowing what was achievable, so `oracle_ceiling` reports what a perfect ranker would have caught with the same budget and every policy is read as a fraction of it.

**Intervals, always.** `caseload.evaluation` provides interquartile mean and stratified bootstrap intervals following `rliable` (Agarwal et al., NeurIPS 2021), plus **paired** differences, because episode difficulty varies far more than the gap between policies.

## Related work

- **FALCON**, "Fatigue-Aware Learning to Defer via Constrained Optimisation" ([arXiv 2604.00904](https://arxiv.org/abs/2604.00904), [code](https://github.com/zhengzhang37/FALCON)) formulates learning-to-defer as a CMDP whose state includes cumulative human workload, trained with PPO-Lagrangian. It is single-expert with a binary defer action, on vision and medical data. The difference here is multi-expert per-analyst routing, asymmetric fraud cost, and real fraud alerts.
- **DeCCaF** (TMLR) is FiFAR's own baseline: supervised expert-loss models plus constraint programming, solved to optimality per batch.
- **Monitored MDPs** (Parisi et al., AAMAS 2024) and **selective labels** (Lakkaraju et al., KDD 2017) are the two formalisms the drift environment sits between.

## Data

Neither dataset is redistributed. Elliptic is on Kaggle; FiFAR is on figshare under CC BY ([10.6084/m9.figshare.28351172](https://doi.org/10.6084/m9.figshare.28351172)). The drift simulator needs no download, so the package works without either.

Only 22.9% of Elliptic transactions carry a label, and unlabelled cases are left in, because investigating one costs budget and reveals nothing, which is what the operational problem looks like. `load_episode(drop_unlabelled=True)` removes them so that cost can be measured rather than assumed.

## License

BSD 3-Clause.
