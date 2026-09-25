"""Capacity-constrained triage: where should scarce human attention go, and when.

A deployed model scores every alert. A small team of analysts can review some of
them. Reviewed alerts take the analyst's decision, which is recorded in FiFAR for
every analyst and every alert, and which is sometimes wrong: on the shipped data,
the synthetic analysts' false-negative rates run from 0.015 to 0.312 and
false-positive rates from 0.015 to 0.759. Unreviewed alerts are decided by the
model at a fixed threshold.

Review only helps when the analyst is better than the model *on that alert*, so the
decision is not "review the most suspicious" but "review where the model is least
trustworthy", and, because capacity is shared across a horizon of unevenly sized
batches, also "review now or bank the slot for a batch that has not arrived".

That second question is the only place a sequential policy can beat a one-shot
assignment, and it is empty in FiFAR as shipped, where capacity is 100% of alerts in
training scenarios and 90.9% in test ones. Capacity tightness is therefore a swept
parameter here, and the headline measurement is the ratio below which spending
sequentially starts to pay.

Cost is asymmetric and the ratio is explicit, because the conclusion depends on it:
a missed fraud costs ``fn_cost`` and a false alarm costs ``fp_cost``. The default is
the regime FiFAR and DeCCaF state, ``fp_cost = LAMBDA_T = 0.057`` (about 17.5:1).
DeCCaF also runs lambda_t/5 and 5*lambda_t, which it calls "not strictly
comparable"; those are ``STRESS_FP_COSTS`` here and should be reported as stress
cases, not as the benchmark's regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# how much of the remaining capacity to commit to this batch, relative to spending
# it evenly over the batches that remain. index 2 is exactly even spending
PACE_GRID: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.5)
# which alerts to send to a human: by raw suspicion, or by how close the model's
# score sits to its own decision threshold, which is where it is least reliable
BAND_GRID: tuple[str, ...] = ("top-score", "near-threshold", "mixed", "bottom-score")

OBS_NAMES: tuple[str, ...] = (
    "batch_frac",
    "capacity_frac_left",
    "even_pace_available",
    "batch_size_rel",
    "score_mean",
    "score_p90",
    "score_near_thresh_frac",
    "alert_rate_rel",
    "cost_rate_last",
    "cost_rate_ma3",
    "review_value_last",
    "review_value_ma3",
    "batches_left_frac",
)
OBS_DIM = len(OBS_NAMES)


# FiFAR alerts are already everything the deployed model scored above this, so a
# second threshold above it declines cases the bank had decided to look at
ALERT_THRESHOLD = 0.051

# false-alarm cost as a fraction of a missed fraud, derived from the alert threshold t
# as lambda_t = t / (1 - t): FiFAR (Alves et al., Sci Data 2025, Eq. 8, used to
# generate its analysts) and DeCCaF (Alves et al., TMLR 2024, Sec. 4.1 and Eq. 21)
LAMBDA_T = 0.057
# DeCCaF's other two cost structures, run to probe sensitivity and described there as
# "not strictly comparable" to lambda_t, because the alert model was not tuned for them
STRESS_FP_COSTS: dict[str, float] = {
    "lambda_t/5": LAMBDA_T / 5,
    "5*lambda_t": LAMBDA_T * 5,
}


def cost_optimal_threshold(
    score: np.ndarray, y: np.ndarray, fn_cost: float = 1.0, fp_cost: float = LAMBDA_T
) -> float:
    """The threshold a competent operator would already be using.

    Reporting any value for human review without this is measuring threshold
    re-tuning, not review. On FiFAR a fixed threshold of 0.5 flags almost none of
    the alerts and misses almost all the fraud, and the choice of threshold then
    dominates anything a routing policy does.
    """
    cand = np.unique(np.quantile(score, np.linspace(0.0, 1.0, 201)))
    best, best_c = float(cand[0]), float("inf")
    for t in cand:
        call = score > t
        c = fn_cost * float((~call & (y == 1)).sum()) + fp_cost * float((call & (y == 0)).sum())
        if c < best_c:
            best, best_c = float(t), c
    return best


@dataclass
class TriageConfig:
    threshold: float = ALERT_THRESHOLD
    fn_cost: float = 1.0
    fp_cost: float = LAMBDA_T
    assign: str = "best-fnr"  # which analyst takes a routed alert
    carry_capacity: bool = True  # unused review slots roll forward


@dataclass
class BatchRecord:
    batch_index: int
    size: int
    capacity_before: int
    reviewed: int
    pace: float
    band: str
    cost: float
    cost_model_only: float
    fn: int
    fp: int
    reviewer_fixed: int
    reviewer_broke: int


@dataclass
class TriageResult:
    cost: float
    cost_model_only: float
    reviewed: int
    capacity: int
    alerts: int
    records: list[BatchRecord] = field(default_factory=list)

    @property
    def saving(self) -> float:
        """Fraction of the model-only cost that review removed. Higher is better."""
        if self.cost_model_only <= 0:
            return 0.0
        return (self.cost_model_only - self.cost) / self.cost_model_only

    @property
    def cost_per_alert(self) -> float:
        return self.cost / max(self.alerts, 1)

    @property
    def saving_per_review(self) -> float:
        """Cost avoided per review spent. The spend-normalised metric.

        Total saving alone ranks policies by how much of their budget they burn:
        measured on FiFAR, review rates across pacing rules ranged from 85% to 100%
        of available capacity and the saving ranking was exactly the spend ranking.
        Any claim about *allocation* has to be made per unit spent.
        """
        return (self.cost_model_only - self.cost) / max(self.reviewed, 1)

    @property
    def capacity_used(self) -> float:
        return self.reviewed / max(self.capacity, 1)

    @property
    def review_rate(self) -> float:
        return self.reviewed / max(self.alerts, 1)


def _pick(scores: np.ndarray, k: int, band: str, thresh: float, rng) -> np.ndarray:
    if k <= 0 or len(scores) == 0:
        return np.empty(0, dtype=int)
    k = min(k, len(scores))
    if band == "top-score":
        return np.argsort(-scores)[:k]
    if band == "bottom-score":
        return np.argsort(scores)[:k]
    if band == "near-threshold":
        return np.argsort(np.abs(scores - thresh))[:k]
    if band == "mixed":
        half = k // 2
        a = np.argsort(np.abs(scores - thresh))[:half]
        # np.setdiff1d would sort by row index and lose the score order
        by_score = np.argsort(-scores)
        rest = by_score[~np.isin(by_score, a)]
        return np.concatenate([a, rest[: k - len(a)]]).astype(int)
    raise ValueError(f"unknown band {band!r}")


class TriageMDP:
    """One pass over a scenario's batches, spending a shared review budget."""

    def __init__(self, data, scenario, cfg: TriageConfig | None = None) -> None:
        self.d = data
        self.sc = scenario
        self.cfg = cfg or TriageConfig()
        self.total_capacity = int(scenario.capacity.sum())
        # analysts ranked by how often they miss fraud; the routing rule uses this
        er = data.expert_error_rates()
        self._order = np.argsort(er[:, 0])
        self.carry_capacity = self.cfg.carry_capacity
        self._alert_rate = float(data.y.mean())

    def reset(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self.t = 0
        self.cap_left = self.total_capacity
        self.cost = 0.0
        self.cost_model = 0.0
        self.reviewed = 0
        self._carried = np.zeros(self.sc.capacity.shape[1], dtype=int)
        # per alert row: the decision that stood (-1 until its batch is seen) and
        # whether an analyst made it, so costs can be resampled alert by alert
        self.final_decision = np.full(self.d.n, -1, dtype=int)
        self.routed_mask = np.zeros(self.d.n, dtype=bool)
        self.records: list[BatchRecord] = []
        self._cost_rates: list[float] = []
        self._values: list[float] = []
        return self._observe()

    @property
    def done(self) -> bool:
        return self.t >= self.sc.n_batches

    def _observe(self) -> np.ndarray:
        if self.done:
            return np.zeros(OBS_DIM, dtype=np.float32)
        idx = self.sc.batches[self.t]
        s = self.d.score[idx]
        left = self.sc.n_batches - self.t
        even = self.cap_left / max(left, 1)
        mean_size = self.sc.total_alerts / max(self.sc.n_batches, 1)
        th = self.cfg.threshold
        obs = np.array(
            [
                self.t / max(self.sc.n_batches, 1),
                self.cap_left / max(self.total_capacity, 1),
                min(even / max(len(idx), 1), 1.0),
                min(len(idx) / max(mean_size, 1.0), 3.0) / 3.0,
                float(s.mean()),
                float(np.quantile(s, 0.9)),
                float((np.abs(s - th) < 0.1).mean()),
                float((s > th).mean()),
                self._cost_rates[-1] if self._cost_rates else 0.0,
                float(np.mean(self._cost_rates[-3:])) if self._cost_rates else 0.0,
                self._values[-1] if self._values else 0.0,
                float(np.mean(self._values[-3:])) if self._values else 0.0,
                left / max(self.sc.n_batches, 1),
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _decide(
        self, idx: np.ndarray, review_local: np.ndarray, room: np.ndarray
    ) -> tuple[float, float, int, int, int, int, np.ndarray]:
        """Apply this batch's decisions and return the cost and the alerts routed.

        ``room`` is each analyst's review slots for this batch and is spent in place.
        A picked alert that finds no analyst with room left keeps the model's call and
        is not a review, so callers must count ``routed``, not ``review_local``.
        """
        c = self.cfg
        y = self.d.y[idx]
        model_call = (self.d.score[idx] > c.threshold).astype(int)
        final = model_call.copy()

        # route reviewed alerts to active analysts, lowest-miss-rate first, honouring
        # per-analyst capacity for this batch
        routed = []
        if len(review_local):
            order = [j for j in self._order if room[j] > 0]
            cursor = 0
            for local in review_local:
                while cursor < len(order) and room[order[cursor]] <= 0:
                    cursor += 1
                if cursor >= len(order):
                    break
                j = order[cursor]
                room[j] -= 1
                final[local] = self.d.expert[idx[local], j]
                routed.append(local)
        routed = np.asarray(routed, dtype=int)
        self.final_decision[idx] = final
        self.routed_mask[idx[routed]] = True

        def cost_of(dec: np.ndarray) -> tuple[float, int, int]:
            fn = int(((dec == 0) & (y == 1)).sum())
            fp = int(((dec == 1) & (y == 0)).sum())
            return c.fn_cost * fn + c.fp_cost * fp, fn, fp

        got, fn, fp = cost_of(final)
        base, _, _ = cost_of(model_call)
        # how the human changed the outcome on the alerts they saw, both directions,
        # because a reviewer who overturns a correct model call is a real cost
        if len(routed):
            model_right = model_call[routed] == y[routed]
            human_right = final[routed] == y[routed]
            fixed = int((~model_right & human_right).sum())
            broke = int((model_right & ~human_right).sum())
        else:
            fixed = broke = 0
        return got, base, fn, fp, fixed, broke, routed

    def step(self, pace_idx: int, band_idx: int):
        if self.done:
            raise RuntimeError("scenario is over; call reset()")
        pace = PACE_GRID[int(pace_idx)]
        band = BAND_GRID[int(band_idx)]
        idx = self.sc.batches[self.t]
        left = self.sc.n_batches - self.t
        even = self.cap_left / max(left, 1)
        # as FiFAR ships it, each analyst's slots are per batch and expire unused
        room = self.sc.capacity[self.t].astype(int).copy()
        if self.carry_capacity:
            # an analyst's unused slots roll forward to their later batches, so a slow
            # batch banks capacity for a large one. slots cannot be borrowed from
            # batches that have not arrived
            room += self._carried
        want = round(pace * even)
        ceiling = min(len(idx), self.cap_left, int(room.sum()))
        n_review = int(max(min(want, ceiling), 0))

        local = _pick(self.d.score[idx], n_review, band, self.cfg.threshold, self._rng)
        got, base, fn, fp, fixed, broke, routed = self._decide(idx, local, room)
        n_routed = len(routed)
        if self.carry_capacity:
            self._carried = room

        cap_before = self.cap_left
        self.cap_left -= n_routed
        self.reviewed += n_routed
        self.cost += got
        self.cost_model += base
        self._cost_rates.append(got / max(len(idx), 1))
        self._values.append((base - got) / n_routed if n_routed else 0.0)

        rec = BatchRecord(
            batch_index=self.t,
            size=len(idx),
            capacity_before=cap_before,
            reviewed=n_routed,
            pace=pace,
            band=band,
            cost=got,
            cost_model_only=base,
            fn=fn,
            fp=fp,
            reviewer_fixed=fixed,
            reviewer_broke=broke,
        )
        self.records.append(rec)
        self.t += 1
        # reward is cost avoided, so higher is better and zero means review was pointless
        reward = float(base - got)
        return self._observe(), reward, self.done, rec

    def result(self) -> TriageResult:
        return TriageResult(
            cost=self.cost,
            cost_model_only=self.cost_model,
            reviewed=self.reviewed,
            capacity=self.total_capacity,
            alerts=self.sc.total_alerts,
            records=list(self.records),
        )
