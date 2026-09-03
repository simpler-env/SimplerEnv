"""Success aggregation for manipulation rollouts."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class PlacementSuccessTracker:
    """Require a released object to remain on target for several control steps.

    Placement environments expose ``src_on_target`` and ``is_src_obj_grasped`` in
    ``info``.  A geometric target check alone can fire while the gripper is still
    holding the object, or before it has settled.  This tracker keeps the original
    environment result for tasks without those fields and applies a short stability
    window only to placement tasks.

    :param required_stable_steps: Number of consecutive released/on-target frames
        required before reporting success.
    """

    required_stable_steps: int = 3
    stable_steps: int = 0

    def __post_init__(self) -> None:
        if self.required_stable_steps < 1:
            raise ValueError("required_stable_steps must be positive")

    def update(self, env_success: bool, info: Mapping[str, Any]) -> bool:
        """Aggregate one environment step into a stable success indicator."""
        if "src_on_target" not in info:
            # Non-placement tasks retain the environment's native success semantics.
            self.stable_steps = 0
            return bool(env_success)

        released = not bool(info.get("is_src_obj_grasped", False))
        candidate = bool(env_success) and bool(info["src_on_target"]) and released
        if candidate:
            self.stable_steps += 1
        else:
            self.stable_steps = 0
        return self.stable_steps >= self.required_stable_steps
