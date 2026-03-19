import time
from dataclasses import dataclass, field
from typing import List


# ========================= DISCRETE MOVE PARAMETERS =========================

MOVE_STEP_CM = 20
FORWARD_STEP_CM = 30
MOVE_COOLDOWN_S = 1.0
SEARCH_YAW_DEG = 20


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
    Discrete movement controller.

    Instead of returning rc values every frame, this controller decides on one
    movement at a time and calls Tello commands like:
      - move_left(x)
      - move_right(x)
      - move_up(x)
      - move_down(x)
      - move_forward(x)
      - rotate_clockwise(x)

    Usage in your main loop:
        action = ctrl.update(tello, vp)

    The return value is a string for debugging/overlay.
    """

    def __init__(self):
        self.last_move_time = 0.0
        self.state = "SEARCH"

    def _can_move(self) -> bool:
        return (time.time() - self.last_move_time) >= MOVE_COOLDOWN_S

    def _mark_move(self):
        self.last_move_time = time.time()

    def update(self, tello, vp: VisionPacket) -> str:
        corners = set(vp.corners_seen or [])

        if not vp.gate_detected or vp.N_corners == 0:
            self.state = "SEARCH"
            if not self._can_move():
                return "SEARCH_COOLDOWN"

            try:
                tello.rotate_clockwise(SEARCH_YAW_DEG)
                self._mark_move()
                return f"SEARCH: rotate_clockwise({SEARCH_YAW_DEG})"
            except Exception as e:
                return f"SEARCH_ERROR: {e}"

        self.state = "ALIGN"

        if not self._can_move():
            return "ALIGN_COOLDOWN"

        try:
            # ---------------- All four visible ----------------
            if corners == {"TL", "TR", "BL", "BR"}:
                tello.move_forward(FORWARD_STEP_CM)
                self._mark_move()
                return f"move_forward({FORWARD_STEP_CM})"

            # ---------------- Single-marker cases ----------------
            if corners == {"BR"}:
                tello.move_left(MOVE_STEP_CM)
                self._mark_move()
                return f"move_left({MOVE_STEP_CM})"

            if corners == {"BL"}:
                tello.move_right(MOVE_STEP_CM)
                self._mark_move()
                return f"move_right({MOVE_STEP_CM})"

            if corners == {"TR"}:
                tello.move_left(MOVE_STEP_CM)
                self._mark_move()
                return f"move_left({MOVE_STEP_CM})"

            if corners == {"TL"}:
                tello.move_right(MOVE_STEP_CM)
                self._mark_move()
                return f"move_right({MOVE_STEP_CM})"

            # ---------------- Two-marker cases ----------------
            if corners == {"BL", "BR"}:
                tello.move_up(MOVE_STEP_CM)
                self._mark_move()
                return f"move_up({MOVE_STEP_CM})"

            if corners == {"TL", "TR"}:
                tello.move_down(MOVE_STEP_CM)
                self._mark_move()
                return f"move_down({MOVE_STEP_CM})"

            if corners == {"TL", "BL"}:
                tello.move_right(MOVE_STEP_CM)
                self._mark_move()
                return f"move_right({MOVE_STEP_CM})"

            if corners == {"TR", "BR"}:
                tello.move_left(MOVE_STEP_CM)
                self._mark_move()
                return f"move_left({MOVE_STEP_CM})"

            # ---------------- Three-marker cases ----------------
            # Missing TL -> gate is biased top-left, so move up + left correction choice
            if corners == {"TR", "BL", "BR"}:
                tello.move_left(MOVE_STEP_CM)
                self._mark_move()
                return f"move_left({MOVE_STEP_CM}) [missing TL]"

            # Missing TR
            if corners == {"TL", "BL", "BR"}:
                tello.move_right(MOVE_STEP_CM)
                self._mark_move()
                return f"move_right({MOVE_STEP_CM}) [missing TR]"

            # Missing BL
            if corners == {"TL", "TR", "BR"}:
                tello.move_up(MOVE_STEP_CM)
                self._mark_move()
                return f"move_up({MOVE_STEP_CM}) [missing BL]"

            # Missing BR
            if corners == {"TL", "TR", "BL"}:
                tello.move_up(MOVE_STEP_CM)
                self._mark_move()
                return f"move_up({MOVE_STEP_CM}) [missing BR]"

            # ---------------- Fallback ----------------
            return f"NO_RULE corners={sorted(corners)}"

        except Exception as e:
            return f"MOVE_ERROR: {e}"