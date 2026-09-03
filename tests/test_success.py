import importlib.util
from pathlib import Path

# Loading the pure aggregation module directly keeps these CPU tests independent of
# the optional simulator dependency imported by ``simpler_env``.
_SUCCESS_PATH = Path(__file__).parents[1] / "simpler_env" / "evaluation" / "success.py"
_SPEC = importlib.util.spec_from_file_location("simpler_env_success", _SUCCESS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
PlacementSuccessTracker = _MODULE.PlacementSuccessTracker


def test_placement_requires_release_and_stability_window():
    tracker = PlacementSuccessTracker(required_stable_steps=3)

    assert not tracker.update(True, {"src_on_target": True, "is_src_obj_grasped": True})
    assert not tracker.update(True, {"src_on_target": True, "is_src_obj_grasped": False})
    assert not tracker.update(True, {"src_on_target": True, "is_src_obj_grasped": False})
    assert tracker.update(True, {"src_on_target": True, "is_src_obj_grasped": False})


def test_placement_instability_resets_window():
    tracker = PlacementSuccessTracker(required_stable_steps=2)

    assert not tracker.update(True, {"src_on_target": True})
    assert not tracker.update(False, {"src_on_target": False})
    assert not tracker.update(True, {"src_on_target": True})
    assert tracker.update(True, {"src_on_target": True})


def test_non_placement_tasks_keep_native_success():
    tracker = PlacementSuccessTracker()

    assert tracker.update(True, {"distance": 0.0})
    assert not tracker.update(False, {"distance": 1.0})


def test_stability_window_must_be_positive():
    try:
        PlacementSuccessTracker(required_stable_steps=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected a positive stability window")
