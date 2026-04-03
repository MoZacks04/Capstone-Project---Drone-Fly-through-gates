import time
from dataclasses import dataclass, field
from typing import List, Optional, Set


# ========================= DISCRETE MOVE PARAMETERS =========================

MOVE_STEP_CM = 20
FORWARD_STEP_CM = 20
SEARCH_YAW_DEG = 10

# Minimum time before another command can be sent
MOVE_COOLDOWN_S = 2.0
SEARCH_COOLDOWN_S = 2.5
FORWARD_COOLDOWN_S = 3.0

# Additional settle time after sending a command
SETTLE_TIME_S = 1.5

# Require the same corner pattern for a few frames before acting
STABLE_FRAMES_REQUIRED = 3


@dataclass
class VisionPacket:
    gate_detected: bool
    gate_name: str
    N_corners: int
    u_c: float
    v_c: float
    A_avg: float
    theta: float = 0.0
    theta_valid: bool = False
    corners_seen: List[str] = field(default_factory=list)
    marker_ids: List[int] = field(default_factory=list)


class Controller:
    """
    Safer discrete movement controller for Tello.

    Improvements:
    - prevents command spam
    - waits for settle time after each move
    - requires stable vision before acting
    - uses different cooldowns for search vs move vs forward
    """

    def __init__(self):
        self.last_move_time = 0.0
        self.busy_until = 0.0
        self.state = "SEARCH"

        self.last_corner_pattern: Optional[Set[str]] = None
        self.stable_count = 0

    def _now(self) -> float:
        return time.time()

    def _is_busy(self) -> bool:
        return self._now() < self.busy_until

    def _can_move(self, cooldown: float) -> bool:
        return (self._now() - self.last_move_time) >= cooldown and not self._is_busy()

    def _mark_move(self, cooldown: float):
        now = self._now()
        self.last_move_time = now
        self.busy_until = now + max(cooldown, SETTLE_TIME_S)

    def _update_stability(self, corners: Set[str]):
        if corners == self.last_corner_pattern:
            self.stable_count += 1
        else:
            self.last_corner_pattern = set(corners)
            self.stable_count = 1

    def _stable_enough(self) -> bool:
        return self.stable_count >= STABLE_FRAMES_REQUIRED

    def _do_command(self, tello, func_name: str, value: int, cooldown: float, note: str = "") -> str:
        try:
            getattr(tello, func_name)(value)
            self._mark_move(cooldown)
            msg = f"{func_name}({value})"
            if note:
                msg += f" {note}"
            return msg
        except Exception as e:
            return f"{func_name.upper()}_ERROR: {e}"

    def update(self, tello, vp: VisionPacket) -> str:
        corners = set(vp.corners_seen or [])

        if self._is_busy():
            return f"WAIT_SETTLE state={self.state}"

        # ---------------- Search mode ----------------
        if not vp.gate_detected or vp.N_corners == 0:
            self.state = "SEARCH"
            self.last_corner_pattern = None
            self.stable_count = 0

            if not self._can_move(SEARCH_COOLDOWN_S):
                return "SEARCH_COOLDOWN"

            return self._do_command(
                tello,
                "rotate_clockwise",
                SEARCH_YAW_DEG,
                SEARCH_COOLDOWN_S,
            )

        # ---------------- Align mode ----------------
        self.state = "ALIGN"

        # Require stable corner pattern before reacting
        self._update_stability(corners)
        if not self._stable_enough():
            return f"WAIT_STABLE corners={sorted(corners)} count={self.stable_count}"

        # ---------------- All four visible ----------------
        if corners == {"TL", "TR", "BL", "BR"}:
            if not self._can_move(FORWARD_COOLDOWN_S):
                return "FORWARD_COOLDOWN"
            return self._do_command(
                tello,
                "move_forward",
                FORWARD_STEP_CM,
                FORWARD_COOLDOWN_S,
            )

        # All other alignment moves use MOVE_COOLDOWN_S
        if not self._can_move(MOVE_COOLDOWN_S):
            return "ALIGN_COOLDOWN"

        # ---------------- Single-marker cases ----------------
        if corners == {"BR"}:
            return self._do_command(tello, "move_left", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        if corners == {"BL"}:
            return self._do_command(tello, "move_right", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        if corners == {"TR"}:
            return self._do_command(tello, "move_left", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        if corners == {"TL"}:
            return self._do_command(tello, "move_right", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        # ---------------- Two-marker cases ----------------
        if corners == {"BL", "BR"}:
            return self._do_command(tello, "move_up", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        if corners == {"TL", "TR"}:
            return self._do_command(tello, "move_down", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        if corners == {"TL", "BL"}:
            return self._do_command(tello, "move_right", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        if corners == {"TR", "BR"}:
            return self._do_command(tello, "move_left", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        # ---------------- Three-marker cases ----------------
        if corners == {"TR", "BL", "BR"}:
            return self._do_command(tello, "move_left", MOVE_STEP_CM, MOVE_COOLDOWN_S, "[missing TL]")

        if corners == {"TL", "BL", "BR"}:
            return self._do_command(tello, "move_right", MOVE_STEP_CM, MOVE_COOLDOWN_S, "[missing TR]")

        if corners == {"TL", "TR", "BR"}:
            return self._do_command(tello, "move_up", MOVE_STEP_CM, MOVE_COOLDOWN_S, "[missing BL]")

        if corners == {"TL", "TR", "BL"}:
            return self._do_command(tello, "move_up", MOVE_STEP_CM, MOVE_COOLDOWN_S, "[missing BR]")

        return f"NO_RULE corners={sorted(corners)}"