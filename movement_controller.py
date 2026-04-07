import time
from dataclasses import dataclass, field
from typing import List, Optional, Set


MOVE_STEP_CM = 20
VERTICAL_STEP_CM = 20
FORWARD_STEP_CM = 60
SEARCH_YAW_DEG = 20

MOVE_COOLDOWN_S = 2.0
VERTICAL_COOLDOWN_S = 3.0
SEARCH_COOLDOWN_S = 2.5
FORWARD_COOLDOWN_S = 4.0
SETTLE_TIME_S = 1.5

STABLE_FRAMES_REQUIRED = 2

# Deadbands: do nothing if error is within these ranges
X_DEADBAND_CM = 8.0
Y_DEADBAND_CM = 12.0

# Ignore absurd pose spikes
MAX_REASONABLE_Y_ERR_CM = 80.0


@dataclass
class VisionPacket:
    gate_detected: bool
    gate_name: str
    N_corners: int
    u_c: float
    v_c: float
    A_avg: float
    x_err: float = 0.0
    y_err: float = 0.0
    z_err: float = 0.0
    theta: float = 0.0
    theta_valid: bool = False
    corners_seen: List[str] = field(default_factory=list)
    marker_ids: List[int] = field(default_factory=list)


class Controller:
    def __init__(self):
        self.last_move_time = 0.0
        self.busy_until = 0.0
        self.state = "SEARCH"
        self.last_corner_pattern: Optional[Set[str]] = None
        self.stable_count = 0
        self.forward_committed = False

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

    def _reset_stability(self):
        self.last_corner_pattern = None
        self.stable_count = 0

    def _stable_enough(self) -> bool:
        return self.stable_count >= STABLE_FRAMES_REQUIRED

    def _do_command(self, tello, func_name: str, value: int, cooldown: float, note: str = "") -> str:
        if func_name in {
            "move_left", "move_right", "move_up", "move_down",
            "move_forward", "move_back"
        } and value < 20:
            value = 20

        try:
            getattr(tello, func_name)(value)
            self._mark_move(cooldown)
            msg = f"{func_name}({value})"
            if note:
                msg += f" {note}"
            return msg
        except Exception as e:
            return f"COMMAND_ERROR {func_name}({value}): {e}"

    def update(self, tello, vp: VisionPacket) -> str:
        corners = set(vp.corners_seen or [])

        if self._is_busy():
            return f"WAIT_SETTLE state={self.state}"

        if self.forward_committed:
            self.state = "COMMIT_FORWARD"
            self.forward_committed = False
            self._reset_stability()

            if not self._can_move(FORWARD_COOLDOWN_S):
                return "FORWARD_COOLDOWN"

            return self._do_command(
                tello,
                "move_forward",
                FORWARD_STEP_CM,
                FORWARD_COOLDOWN_S,
                "[committed after 4-marker lock]"
            )

        if len(corners) == 0 or vp.N_corners == 0:
            self.state = "SEARCH"
            self._reset_stability()

            if not self._can_move(SEARCH_COOLDOWN_S):
                return "SEARCH_COOLDOWN"

            return self._do_command(
                tello,
                "rotate_clockwise",
                SEARCH_YAW_DEG,
                SEARCH_COOLDOWN_S,
            )

        self.state = "ALIGN"
        self._update_stability(corners)

        if not self._stable_enough():
            return f"WAIT_STABLE corners={sorted(corners)} count={self.stable_count}"

        if corners == {"TL", "TR", "BL", "BR"}:
            self.forward_committed = True
            return "FORWARD_LOCKED"

        # -------- Vertical control from y_err, not corner combos --------
        if abs(vp.y_err) <= MAX_REASONABLE_Y_ERR_CM and abs(vp.y_err) > Y_DEADBAND_CM:
            if not self._can_move(VERTICAL_COOLDOWN_S):
                return f"VERTICAL_COOLDOWN y_err={vp.y_err:.1f}"

            # Tune sign if needed based on your camera convention:
            # From your logs, negative y_err often corresponded to sending "up".
            if vp.y_err < -Y_DEADBAND_CM:
                return self._do_command(
                    tello, "move_up", VERTICAL_STEP_CM, VERTICAL_COOLDOWN_S,
                    f"[y_err={vp.y_err:.1f}]"
                )

            if vp.y_err > Y_DEADBAND_CM:
                return self._do_command(
                    tello, "move_down", VERTICAL_STEP_CM, VERTICAL_COOLDOWN_S,
                    f"[y_err={vp.y_err:.1f}]"
                )

        # -------- Horizontal fallback --------
        if not self._can_move(MOVE_COOLDOWN_S):
            return "ALIGN_COOLDOWN"

        if corners == {"BR"}:
            return self._do_command(tello, "move_left", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        if corners == {"BL"}:
            return self._do_command(tello, "move_right", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        if corners == {"TR"}:
            return self._do_command(tello, "move_left", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        if corners == {"TL"}:
            return self._do_command(tello, "move_right", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        if corners == {"TL", "BL"}:
            return self._do_command(tello, "move_right", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        if corners == {"TR", "BR"}:
            return self._do_command(tello, "move_left", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        if corners == {"TR", "BL", "BR"}:
            return self._do_command(tello, "move_left", MOVE_STEP_CM, MOVE_COOLDOWN_S, "[missing TL]")

        if corners == {"TL", "BL", "BR"}:
            return self._do_command(tello, "move_right", MOVE_STEP_CM, MOVE_COOLDOWN_S, "[missing TR]")

        return f"HOLD y_err={vp.y_err:.1f} corners={sorted(corners)}"