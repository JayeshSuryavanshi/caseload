from __future__ import annotations

import numpy as np
import pytest

from caseload import (
    EXPLORE_GRID,
    OBS_DIM,
    PACE_GRID,
    Episode,
    GradientBoostScorer,
    InvestigationMDP,
    LogisticScorer,
    RandomGrid,
    Round,
    TopK,
    YieldTriggered,
    oracle_ceiling,
    rollout,
)
from caseload.envs import DriftConfig, make_episode


def tiny_episode(seed: int = 0, n_rounds: int = 6, n: int = 120) -> Episode:
    rng = np.random.default_rng(seed)

    def draw(k: int) -> Round:
        x = rng.normal(size=(k, 6))
        y = (rng.random(k) < 0.15).astype(int)
        x[y == 1] += 2.0  # make positives learnable
        return Round(x=x, y=y)

    warm = draw(400)
    return Episode(rounds=[draw(n) for _ in range(n_rounds)], warm_x=warm.x, warm_y=warm.y)


def test_round_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        Round(x=np.zeros((3, 2)), y=np.zeros(4, dtype=int))


def test_budget_is_never_exceeded():
    ep = tiny_episode()
    for bf in (0.02, 0.1, 0.5, 1.0):
        mdp = InvestigationMDP(ep, LogisticScorer(), budget_frac=bf)
        r = rollout(mdp, RandomGrid(seed=1))
        assert r.spent <= r.budget, f"spent {r.spent} over budget {r.budget}"


def test_aggressive_pace_cannot_overspend():
    """The fastest pace on every round must still respect the total budget."""
    ep = tiny_episode()
    mdp = InvestigationMDP(ep, LogisticScorer(), budget_frac=0.3)
    mdp.reset(seed=0)
    top = len(PACE_GRID) - 1
    while not mdp.done:
        mdp.step(top, 0)
    assert mdp.result().spent <= mdp.total_budget
    assert mdp.budget_left >= 0


def test_spend_never_exceeds_pool():
    ep = tiny_episode(n_rounds=4, n=20)
    mdp = InvestigationMDP(ep, LogisticScorer(), budget_frac=1.0)
    mdp.reset(seed=0)
    while not mdp.done:
        _, _, _, rec = mdp.step(len(PACE_GRID) - 1, 0)
        assert rec.spend <= rec.pool_size


def test_observation_shape_and_finiteness():
    ep = tiny_episode()
    mdp = InvestigationMDP(ep, LogisticScorer())
    obs = mdp.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    while not mdp.done:
        obs, reward, done, _ = mdp.step(2, 1)
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs)), "observation must stay finite"
        assert np.isfinite(reward)


def test_only_investigated_labels_enter_the_pool():
    """The selective-labels rule. Violating it would silently make the task trivial."""
    ep = tiny_episode()
    mdp = InvestigationMDP(ep, LogisticScorer(), budget_frac=0.1)
    mdp.reset(seed=0)
    start = len(np.concatenate(mdp.pool_y))
    total_spend = 0
    while not mdp.done:
        _, _, _, rec = mdp.step(2, 0)
        total_spend += rec.spend
    end = len(np.concatenate(mdp.pool_y))
    assert end - start == total_spend, "pool grew by something other than what was investigated"


def test_found_never_exceeds_available_or_spend():
    ep = tiny_episode()
    mdp = InvestigationMDP(ep, GradientBoostScorer(max_iter=8), budget_frac=0.2)
    mdp.reset(seed=0)
    while not mdp.done:
        _, _, _, rec = mdp.step(2, 2)
        assert rec.found <= rec.available
        assert rec.found <= rec.spend
        assert rec.found == rec.found_by_exploit + rec.found_by_explore
        assert rec.spend == rec.exploit_spend + rec.explore_spend


def test_zero_pace_spends_nothing():
    ep = tiny_episode()
    mdp = InvestigationMDP(ep, LogisticScorer())
    mdp.reset(seed=0)
    _, reward, _, rec = mdp.step(PACE_GRID.index(0.0), 0)
    assert rec.spend == 0
    assert rec.found == 0
    assert reward == 0.0


def test_uniform_pace_matches_even_spending():
    """Pace 1.0 must reproduce a uniform spender, since that is the baseline's claim."""
    ep = tiny_episode(n_rounds=5, n=100)
    mdp = InvestigationMDP(ep, LogisticScorer(), budget_frac=0.10)
    r = rollout(mdp, TopK())
    per = [s.spend for s in r.steps]
    assert max(per) - min(per) <= 1, f"uniform pace gave uneven spends {per}"
    assert r.spent <= r.budget


def test_recall_and_precision_bounds():
    ep = tiny_episode()
    mdp = InvestigationMDP(ep, GradientBoostScorer(max_iter=8))
    r = rollout(mdp, TopK())
    assert 0.0 <= r.recall <= 1.0
    assert 0.0 <= r.precision <= 1.0


def test_oracle_is_an_upper_bound():
    ep = tiny_episode()
    for bf in (0.05, 0.2):
        mdp = InvestigationMDP(ep, GradientBoostScorer(max_iter=8), budget_frac=bf)
        learned = rollout(mdp, TopK())
        ceiling = oracle_ceiling(mdp)
        assert ceiling.recall >= learned.recall - 1e-9, "a policy beat the oracle ceiling"


def test_step_after_done_raises():
    ep = tiny_episode(n_rounds=2)
    mdp = InvestigationMDP(ep, LogisticScorer())
    mdp.reset(seed=0)
    mdp.step(2, 0)
    mdp.step(2, 0)
    assert mdp.done
    with pytest.raises(RuntimeError):
        mdp.step(2, 0)


def test_rollout_is_deterministic_given_seed():
    ep = tiny_episode()
    a = rollout(InvestigationMDP(ep, GradientBoostScorer(max_iter=8)), TopK(), seed=3)
    b = rollout(InvestigationMDP(ep, GradientBoostScorer(max_iter=8)), TopK(), seed=3)
    assert a.found == b.found and a.spent == b.spent


def test_invalid_budget_rejected():
    ep = tiny_episode()
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            InvestigationMDP(ep, LogisticScorer(), budget_frac=bad)


def test_scorer_handles_single_class_pool():
    """Early on, an investigated pool can be all-negative. That must not crash."""
    sc = GradientBoostScorer(max_iter=5)
    sc.fit(np.random.default_rng(0).normal(size=(20, 4)), np.zeros(20, dtype=int))
    s = sc.score(np.random.default_rng(1).normal(size=(7, 4)))
    assert s.shape == (7,)
    assert np.all(np.isfinite(s))


def test_drift_episode_has_a_break_and_positives():
    ep, breaks = make_episode(seed=5, cfg=DriftConfig(n_rounds=12))
    assert ep.n_rounds == 12
    assert ep.n_positives > 0
    assert len(breaks) == 1
    assert 0 < breaks[0] < ep.n_rounds, f"break at {breaks[0]} is outside the episode"


def test_drift_is_reproducible():
    a, ba = make_episode(seed=9)
    b, bb = make_episode(seed=9)
    assert ba == bb
    assert np.allclose(a.rounds[0].x, b.rounds[0].x)
    assert a.n_positives == b.n_positives


def test_drift_seeds_differ():
    a, _ = make_episode(seed=1)
    b, _ = make_episode(seed=2)
    same_shape = a.rounds[0].x.shape == b.rounds[0].x.shape
    # different seeds should differ in the draw, and usually also in pool sizes
    assert (not same_shape) or not np.allclose(a.rounds[0].x, b.rounds[0].x)


def test_yield_triggered_reacts_to_collapse():
    """It must switch to exploration when yield craters, or it is not adaptive."""
    pol = YieldTriggered(drop=0.5, explore=0.6, patience=1)
    pol.reset()
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[8] = obs[9] = 0.4  # healthy yield
    for _ in range(3):
        assert pol.act(obs)[1] == EXPLORE_GRID.index(0.0)
    obs[8] = 0.0  # collapse
    assert pol.act(obs)[1] != EXPLORE_GRID.index(0.0)
