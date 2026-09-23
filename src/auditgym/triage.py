"""Capacity-constrained triage: where should scarce human attention go, and when.

A deployed model scores every alert. A small team of analysts can review some of
them. Reviewed alerts take the analyst's decision, which is recorded in FiFAR for
every analyst and every alert, and which is sometimes wrong: measured on the
shipped data, analyst false-negative rates run from 0.015 to 0.312 and
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
a missed fraud costs ``fn_cost`` and a false alarm costs ``fp_cost``.
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


def cost_optimal_threshold(
    score: np.ndarray, y: np.ndarray, fn_cost: float = 1.0, fp_cost: float = 0.0114
) -> float:
    """The threshold a competent operator would already be using.

    Reporting any value for human review without this is measuring threshold
    re-tuning, not review. Measured on FiFAR at fp_cost=0.05, a threshold of 0.5
    costs 3,427 against 1,347 at the alert threshold, so the choice dominates
    anything a routing policy does.
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
    fp_cost: float = 0.0114
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
        rest = np.setdiff1d(np.argsort(-scores), a, assume_unique=False)
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
        self._active = np.where(scenario.capacity.sum(axis=0) > 0)[0]
        self.carry_capacity = self.cfg.carry_capacity
        self._alert_rate = float(data.y.mean())

    def reset(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self.t = 0
        self.cap_left = self.total_capacity
        self.cost = 0.0
        self.cost_model = 0.0
        self.reviewed = 0
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
        self, idx: np.ndarray, review_local: np.ndarray
    ) -> tuple[float, float, int, int, int, int]:
        c = self.cfg
        y = self.d.y[idx]
        model_call = (self.d.score[idx] > c.threshold).astype(int)
        final = model_call.copy()

        # route reviewed alerts to active analysts, lowest-miss-rate first, honouring
        # per-analyst capacity for this batch
        if len(review_local):
            order = [j for j in self._order if j in set(self._active.tolist())]
            room = {j: int(self.sc.capacity[self.t, j]) for j in order}
            cursor = 0
            for local in review_local:
                while cursor < len(order) and room[order[cursor]] <= 0:
                    cursor += 1
                if cursor >= len(order):
                    break
                j = order[cursor]
                room[j] -= 1
                final[local] = self.d.expert[idx[local], j]

        def cost_of(dec: np.ndarray) -> tuple[float, int, int]:
            fn = int(((dec == 0) & (y == 1)).sum())
            fp = int(((dec == 1) & (y == 0)).sum())
            return c.fn_cost * fn + c.fp_cost * fp, fn, fp

        got, fn, fp = cost_of(final)
        base, _, _ = cost_of(model_call)
        # how the human changed the outcome on the alerts they saw, both directions,
        # because a reviewer who overturns a correct model call is a real cost
        if len(review_local):
            r = review_local
            model_right = model_call[r] == y[r]
            human_right = final[r] == y[r]
            fixed = int((~model_right & human_right).sum())
            broke = int((model_right & ~human_right).sum())
        else:
            fixed = broke = 0
        return got, base, fn, fp, fixed, broke

    def step(self, pace_idx: int, band_idx: int):
        if self.done:
            raise RuntimeError("scenario is over; call reset()")
        pace = PACE_GRID[int(pace_idx)]
        band = BAND_GRID[int(band_idx)]
        idx = self.sc.batches[self.t]
        left = self.sc.n_batches - self.t
        even = self.cap_left / max(left, 1)
        batch_cap = int(self.sc.capacity[self.t].sum())
        want = round(pace * even)
        if self.carry_capacity:
            # unused slots roll forward into a shared pool, so the only ceilings are
            # the batch itself and what is left. this is what makes allocation a real
            # decision rather than an accounting artefact
            ceiling = min(len(idx), self.cap_left)
        else:
            # FiFAR-faithful: capacity is per batch and expires unused
            ceiling = min(len(idx), self.cap_left, max(batch_cap, 0))
        n_review = int(max(min(want, ceiling), 0))

        local = _pick(self.d.score[idx], n_review, band, self.cfg.threshold, self._rng)
        got, base, fn, fp, fixed, broke = self._decide(idx, local)

        cap_before = self.cap_left
        self.cap_left -= len(local)
        self.reviewed += len(local)
        self.cost += got
        self.cost_model += base
        self._cost_rates.append(got / max(len(idx), 1))
        self._values.append((base - got) / max(len(local), 1) if len(local) else 0.0)

        rec = BatchRecord(
            batch_index=self.t,
            size=len(idx),
            capacity_before=cap_before,
            reviewed=len(local),
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
