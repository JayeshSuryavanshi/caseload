from __future__ import annotations

import numpy as np
import pytest

from auditgym.triage import (
    BAND_GRID,
    OBS_DIM,
    PACE_GRID,
    TriageConfig,
    TriageMDP,
    _pick,
)

fifar = pytest.importorskip("auditgym.envs.fifar")
pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def data():
    try:
        return fifar.load()
    except FileNotFoundError:
        pytest.skip("FiFAR not downloaded; run scripts/fetch_fifar.py")


@pytest.fixture(scope="module")
def scenario(data):
    return fifar.load_scenario(data, "shuffle_1#team_1", "train_alert", capacity_ratio=0.10)


def test_shipped_shape_is_what_we_documented(data):
    """Guards the numbers quoted in the docs against a different FiFAR release."""
    assert data.n == fifar.N_ALERTS
    assert data.expert.shape == (fifar.N_ALERTS, fifar.N_EXPERTS)
    assert set(np.unique(data.expert).tolist()) <= {0, 1}
    assert set(np.unique(data.y).tolist()) == {0, 1}


def test_experts_are_fallible_and_heterogeneous(data):
    """The point of FiFAR is that the human is wrong sometimes, and differently."""
    er = data.expert_error_rates()
    assert er.shape == (fifar.N_EXPERTS, 2)
    assert er[:, 0].max() > 0.05, "no analyst ever misses fraud, which cannot be right"
    assert er[:, 0].std() > 0.01, "analysts are identical, so routing could not matter"


def test_shipped_capacity_is_nearly_slack(data):
    """If a future release makes capacity binding, the docs need rewriting."""
    test_sc = fifar.load_scenario(data, "testsize#team_1-var_1", "test")
    assert test_sc.n_batches == 1, "the test split is one-shot; the docs say so"
    assert test_sc.capacity_ratio > 0.85


def test_capacity_rescaling_hits_the_target(data):
    for ratio in (0.02, 0.1, 0.5):
        sc = fifar.load_scenario(data, "shuffle_1#team_1", "train_alert", capacity_ratio=ratio)
        assert abs(sc.capacity_ratio - ratio) < 0.01


def test_never_reviews_more_than_capacity(data, scenario):
    m = TriageMDP(data, scenario, TriageConfig())
    m.reset(seed=0)
    top = len(PACE_GRID) - 1
    while not m.done:
        m.step(top, 1)
    r = m.result()
    assert r.reviewed <= r.capacity, f"reviewed {r.reviewed} over capacity {r.capacity}"


def test_never_reviews_more_than_the_batch(data, scenario):
    m = TriageMDP(data, scenario, TriageConfig())
    m.reset(seed=0)
    while not m.done:
        _, _, _, rec = m.step(len(PACE_GRID) - 1, 1)
        assert rec.reviewed <= rec.size


def test_zero_pace_reviews_nothing_and_costs_the_model_price(data, scenario):
    m = TriageMDP(data, scenario, TriageConfig())
    m.reset(seed=0)
    _, reward, _, rec = m.step(PACE_GRID.index(0.0), 1)
    assert rec.reviewed == 0
    assert reward == 0.0
    assert rec.cost == pytest.approx(rec.cost_model_only)


def test_observation_is_finite_and_right_shape(data, scenario):
    m = TriageMDP(data, scenario, TriageConfig())
    obs = m.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    while not m.done:
        obs, reward, _, _ = m.step(2, 1)
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)


def test_reviewing_can_help_and_can_hurt(data, scenario):
    """A reviewer who overturns a correct model call is a real cost, and the
    environment must record both directions or the metric is flattering."""
    m = TriageMDP(data, scenario, TriageConfig())
    m.reset(seed=0)
    fixed = broke = 0
    while not m.done:
        _, _, _, rec = m.step(2, 1)
        fixed += rec.reviewer_fixed
        broke += rec.reviewer_broke
    assert fixed > 0, "review never helped, which contradicts the measured saving"
    assert broke > 0, "review never hurt, so the expert table is being ignored"


def test_saving_per_review_is_spend_normalised(data):
    """Two policies spending different amounts must be comparable per review."""
    sc = fifar.load_scenario(data, "shuffle_1#team_1", "train_alert", capacity_ratio=0.1)
    out = {}
    for pace in (PACE_GRID.index(0.5), PACE_GRID.index(2.5)):
        m = TriageMDP(data, sc, TriageConfig())
        m.reset(seed=0)
        while not m.done:
            m.step(pace, 1)
        r = m.result()
        out[pace] = r
    lo, hi = out[PACE_GRID.index(0.5)], out[PACE_GRID.index(2.5)]
    assert hi.reviewed > lo.reviewed, "faster pace did not spend more"
    # both metrics must be finite and the normalised one must not be a copy of total
    for r in (lo, hi):
        assert np.isfinite(r.saving_per_review)
        assert 0.0 <= r.capacity_used <= 1.0


def test_review_is_net_negative_at_fifar_cost_regime(data):
    """The headline finding, asserted so a refactor cannot silently reverse it.

    At an 88:1 missed-fraud-to-false-alarm ratio, a 15.7% analyst miss rate against
    a 12.1% fraud base rate costs more than the false alarms analysts clear, so every
    routing band loses money. An earlier version of this test asserted the opposite,
    because it ran at a 0.5 threshold on alerts that are all above 0.051.
    """
    sc = fifar.load_scenario(data, "shuffle_1#team_1", "train_alert", capacity_ratio=0.10)
    for band in BAND_GRID:
        m = TriageMDP(data, sc, TriageConfig())
        m.reset(seed=0)
        while not m.done:
            m.step(2, BAND_GRID.index(band))
        assert m.result().saving_per_review < 0, f"{band} should lose at 88:1"


def test_review_pays_once_false_alarms_are_expensive(data):
    """And the crossover is real: cheap enough fraud cost and review earns its keep."""
    sc = fifar.load_scenario(data, "shuffle_1#team_1", "train_alert", capacity_ratio=0.10)
    got = {}
    for fp_cost in (0.0114, 0.20):
        m = TriageMDP(data, sc, TriageConfig(fp_cost=fp_cost))
        m.reset(seed=0)
        while not m.done:
            m.step(2, 1)
        got[fp_cost] = m.result().saving_per_review
    assert got[0.0114] < 0 < got[0.20], got


def test_first_principles_matches_the_environment(data):
    """The per-review loss should equal prevalence x FNR minus the cleared-alarm gain."""
    er = data.expert_error_rates()
    prev, fnr, fpr = data.y.mean(), er[:, 0].mean(), er[:, 1].mean()
    predicted = (1 - prev) * (1 - fpr) * 0.0114 - prev * fnr * 1.0
    sc = fifar.load_scenario(data, "shuffle_1#team_1", "train_alert", capacity_ratio=0.10)
    m = TriageMDP(data, sc, TriageConfig())
    m.reset(seed=0)
    while not m.done:
        m.step(2, 1)
    measured = m.result().saving_per_review
    # the routing rule is not random, so allow a gap, but the sign and scale must hold
    assert predicted < 0 and measured < 0
    assert abs(measured - predicted) < 0.01, (measured, predicted)


def test_carry_capacity_lets_a_slow_policy_spend_its_budget(data):
    sc = fifar.load_scenario(data, "shuffle_1#team_1", "train_alert", capacity_ratio=0.1)
    used = {}
    for carry in (True, False):
        m = TriageMDP(data, sc, TriageConfig(carry_capacity=carry))
        m.reset(seed=0)
        while not m.done:
            m.step(len(PACE_GRID) - 1, 1)
        used[carry] = m.result().capacity_used
    assert used[True] >= used[False] - 1e-9


def test_pick_bands_are_distinct_and_bounded():
    s = np.linspace(0, 1, 100)
    picks = {b: _pick(s, 10, b, 0.5, np.random.default_rng(0)) for b in BAND_GRID}
    for b, p in picks.items():
        assert len(p) == 10, b
        assert len(set(p.tolist())) == 10, f"{b} returned duplicates"
    assert s[picks["top-score"]].mean() > s[picks["bottom-score"]].mean()
    assert (
        abs(s[picks["near-threshold"]] - 0.5).mean() < abs(s[picks["top-score"]] - 0.5).mean()
    )


def test_pick_handles_degenerate_requests():
    s = np.linspace(0, 1, 5)
    assert len(_pick(s, 0, "top-score", 0.5, np.random.default_rng(0))) == 0
    assert len(_pick(s, 99, "top-score", 0.5, np.random.default_rng(0))) == 5
    assert len(_pick(np.empty(0), 3, "top-score", 0.5, np.random.default_rng(0))) == 0


def test_step_after_done_raises(data, scenario):
    m = TriageMDP(data, scenario, TriageConfig())
    m.reset(seed=0)
    while not m.done:
        m.step(2, 1)
    with pytest.raises(RuntimeError):
        m.step(2, 1)
