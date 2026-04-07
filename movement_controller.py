import time
from dataclasses import dataclass, field
from typing import List, Optional, Set


MOVE_STEP_CM = 20
VERTICAL_STEP_CM = 20
SEARCH_YAW_DEG = 20

MOVE_COOLDOWN_S = 2.0
VERTICAL_COOLDOWN_S = 3.0
SEARCH_COOLDOWN_S = 2.5
FORWARD_COOLDOWN_S = 4.5
SETTLE_TIME_S = 1.5

STABLE_FRAMES_REQUIRED = 1

# Deadbands
X_DEADBAND_CM = 20.0
Y_DEADBAND_CM = 20.0

# Ignore absurd pose spikes
MAX_REASONABLE_X_ERR_CM = 50.0
MAX_REASONABLE_Y_ERR_CM = 50.0
MAX_REASONABLE_Z_ERR_CM = 400.0

# Dynamic forward distance
APPROACH_OFFSET_CM = 40
PASS_THROUGH_MARGIN_CM = 60

MIN_FORWARD_CM = 100
MAX_FORWARD_CM = 250


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
        self.locked_forward_cm = MIN_FORWARD_CM

    def _now(self) -> float:
        return time.time()

    def _is_busy(self) -> bool:
        return self._now() < self.busy_until

    def _can_move(self, cooldown: float) -> bool:
        return (time.time() - self.last_move_time) >= cooldown and not self._is_busy()

    def _mark_move(self, cooldown: float):
        now = time.time()
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

    def _compute_forward_distance(self, vp: VisionPacket) -> int:
        if 0 < vp.z_err <= MAX_REASONABLE_Z_ERR_CM:
            forward_cm = int(vp.z_err - APPROACH_OFFSET_CM + PASS_THROUGH_MARGIN_CM)
        else:
            forward_cm = MAX_FORWARD_CM

        return max(MIN_FORWARD_CM, min(MAX_FORWARD_CM, forward_cm))

    def update(self, tello, vp: VisionPacket) -> str:
        corners = set(vp.corners_seen or [])

        if self._is_busy():
            return f"WAIT_SETTLE state={self.state}"

        # committed forward pass
        if self.forward_committed:
            self.state = "COMMIT_FORWARD"
            self.forward_committed = False
            self._reset_stability()

            if not self._can_move(FORWARD_COOLDOWN_S):
                return "FORWARD_COOLDOWN"

            return self._do_command(
                tello,
                "move_forward",
                self.locked_forward_cm,
                FORWARD_COOLDOWN_S,
                f"[dynamic pass z={vp.z_err:.1f}]"
            )

        # search
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

        # =================================================
        # HIGHEST PRIORITY: all 4 markers visible
        # Do NOT do vertical correction here.
        # =================================================
        if corners == {"TL", "TR", "BL", "BR"}:
            # Only allow small horizontal correction if clearly off
            if abs(vp.x_err) <= MAX_REASONABLE_X_ERR_CM and abs(vp.x_err) > X_DEADBAND_CM:
                if not self._can_move(MOVE_COOLDOWN_S):
                    return f"FOUR_MARKER_HORIZONTAL_COOLDOWN x_err={vp.x_err:.1f}"

                # Flip these if left/right direction is wrong on your setup
                if vp.x_err > X_DEADBAND_CM:
                    return self._do_command(
                        tello,
                        "move_right",
                        MOVE_STEP_CM,
                        MOVE_COOLDOWN_S,
                        f"[4-marker x_err={vp.x_err:.1f}]"
                    )

                if vp.x_err < -X_DEADBAND_CM:
                    return self._do_command(
                        tello,
                        "move_left",
                        MOVE_STEP_CM,
                        MOVE_COOLDOWN_S,
                        f"[4-marker x_err={vp.x_err:.1f}]"
                    )

            # If all 4 visible, commit forward directly
            self.locked_forward_cm = self._compute_forward_distance(vp)
            self.forward_committed = True
            return (
                f"FOUR_MARKER_LOCKED "
                f"x={vp.x_err:.1f} y={vp.y_err:.1f} z={vp.z_err:.1f} "
                f"forward={self.locked_forward_cm}"
            )

        # =================================================
        # Bottom pair logic
        # Vertical correction allowed here
        # =================================================
        if "BL" in corners and "BR" in corners:
            # Horizontal correction
            if abs(vp.x_err) <= MAX_REASONABLE_X_ERR_CM and abs(vp.x_err) > X_DEADBAND_CM:
                if not self._can_move(MOVE_COOLDOWN_S):
                    return f"HORIZONTAL_COOLDOWN x_err={vp.x_err:.1f}"

                if vp.x_err > X_DEADBAND_CM:
                    return self._do_command(
                        tello,
                        "move_right",
                        MOVE_STEP_CM,
                        MOVE_COOLDOWN_S,
                        f"[x_err={vp.x_err:.1f}]"
                    )

                if vp.x_err < -X_DEADBAND_CM:
                    return self._do_command(
                        tello,
                        "move_left",
                        MOVE_STEP_CM,
                        MOVE_COOLDOWN_S,
                        f"[x_err={vp.x_err:.1f}]"
                    )

            # Vertical correction only for partial view / bottom-pair mode
            if abs(vp.y_err) <= MAX_REASONABLE_Y_ERR_CM and abs(vp.y_err) > Y_DEADBAND_CM:
                if not self._can_move(VERTICAL_COOLDOWN_S):
                    return f"VERTICAL_COOLDOWN y_err={vp.y_err:.1f}"

                if vp.y_err < -Y_DEADBAND_CM:
                    return self._do_command(
                        tello,
                        "move_up",
                        VERTICAL_STEP_CM,
                        VERTICAL_COOLDOWN_S,
                        f"[y_err={vp.y_err:.1f}]"
                    )

                if vp.y_err > Y_DEADBAND_CM:
                    return self._do_command(
                        tello,
                        "move_down",
                        VERTICAL_STEP_CM,
                        VERTICAL_COOLDOWN_S,
                        f"[y_err={vp.y_err:.1f}]"
                    )

            self.locked_forward_cm = self._compute_forward_distance(vp)
            self.forward_committed = True
            return (
                f"BOTTOM_PAIR_LOCKED "
                f"x={vp.x_err:.1f} y={vp.y_err:.1f} z={vp.z_err:.1f} "
                f"forward={self.locked_forward_cm}"
            )

        # fallbacks
        if not self._can_move(MOVE_COOLDOWN_S):
            return "ALIGN_COOLDOWN"

        if corners == {"BR"}:
            return self._do_command(
                tello,
                "move_left",
                MOVE_STEP_CM,
                MOVE_COOLDOWN_S,
                "[only BR visible]"
            )

        if corners == {"BL"}:
            return self._do_command(
                tello,
                "move_right",
                MOVE_STEP_CM,
                MOVE_COOLDOWN_S,
                "[only BL visible]"
            )

        if corners == {"TL", "BL"}:
            return self._do_command(tello, "move_right", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        if corners == {"TR", "BR"}:
            return self._do_command(tello, "move_left", MOVE_STEP_CM, MOVE_COOLDOWN_S)

        if corners == {"TR", "BL", "BR"}:
            return self._do_command(tello, "move_left", MOVE_STEP_CM, MOVE_COOLDOWN_S, "[missing TL]")

        if corners == {"TL", "BL", "BR"}:
            return self._do_command(tello, "move_right", MOVE_STEP_CM, MOVE_COOLDOWN_S, "[missing TR]")

        return f"HOLD x={vp.x_err:.1f} y={vp.y_err:.1f} z={vp.z_err:.1f} corners={sorted(corners)}"