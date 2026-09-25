from __future__ import annotations

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")

from caseload.agents.ppo import ActorCritic  # noqa: E402
from caseload.envs import DriftConfig  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKPOINTS = sorted((ROOT / "results").rglob("*.pt"))
PLAIN = (str, int, float, bool)


def load_train_script():
    spec = importlib.util.spec_from_file_location("train", ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fleet_checkpoints_are_committed() -> None:
    assert any(p.parent.name == "fleet" for p in CHECKPOINTS)


@pytest.mark.parametrize("path", CHECKPOINTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_committed_checkpoint_loads_without_pickle(path: pathlib.Path) -> None:
    # weights_only=True refuses arbitrary objects, so this also fails on anything that
    # pickles a class, such as a pathlib path that only exists on one Python version
    blob = torch.load(path, map_location="cpu", weights_only=True)
    assert set(blob) == {"state_dict", "config"}
    assert all(isinstance(v, PLAIN) for v in blob["config"].values())
    ActorCritic().load_state_dict(blob["state_dict"])


def test_train_writes_a_weights_only_checkpoint(tmp_path: pathlib.Path) -> None:
    train = load_train_script()
    net = ActorCritic()
    args = {"seed": 0, "budget": 0.1, "out": tmp_path / "ppo.pt"}
    torch.save(train.checkpoint(net.state_dict(), args, DriftConfig()), args["out"])
    blob = torch.load(args["out"], map_location="cpu", weights_only=True)
    assert blob["config"]["out"] == str(args["out"])
    assert blob["config"]["drift.adversarial_break"] is True
    for k, v in net.state_dict().items():
        assert torch.equal(v, blob["state_dict"][k])
