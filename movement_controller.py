import time
from dataclasses import dataclass, field
from typing import List, Optional, Set


# ---------- RC tuning ----------
RC_MAX = 35                 # max rc speed command magnitude
RC_MIN_EFFECTIVE = 12       # minimum useful rc command once outside deadband
SEARCH_YAW_RC = 20          # yaw command while searching
FORWARD_RC = 18             # forward speed when centered and committed

# deadbands
X_DEADBAND_CM = 8.0
Y_DEADBAND_CM = 12.0
YAW_DEADBAND_DEG = 8.0

# ignore absurd spikes
MAX_REASONABLE_Y_ERR_CM = 80.0

# stability logic
STABLE_FRAMES_REQUIRED = 2
CENTERED_FRAMES_REQUIRED = 3

# control loop pacing
COMMAND_PERIOD_S = 0.08     # ~12.5 Hz rc updates


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
        self.state = "SEARCH"
        self.last_corner_pattern: Optional[Set[str]] = None
        self.stable_count = 0
        self.centered_count = 0
        self.last_cmd_time = 0.0

    def _now(self) -> float:
        return time.time()

    def _update_stability(self, corners: Set[str]):
        if corners == self.last_corner_pattern:
            self.stable_count += 1
        else:
            self.last_corner_pattern = set(corners)
            self.stable_count = 1

    def _reset_stability(self):
        self.last_corner_pattern = None
        self.stable_count = 0
        self.centered_count = 0

    def _stable_enough(self) -> bool:
        return self.stable_count >= STABLE_FRAMES_REQUIRED

    def _rate_limit_ok(self) -> bool:
        return (self._now() - self.last_cmd_time) >= COMMAND_PERIOD_S

    def _mark_command(self):
        self.last_cmd_time = self._now()

    def _clamp(self, value: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, value))

    def _scale_error_to_rc(self, err: float, deadband: float, max_err: float = 40.0) -> int:
        """
        Convert error in cm into an RC command.
        Returns 0 inside deadband.
        Outside deadband, scales up to RC_MAX.
        """
        if abs(err) <= deadband:
            return 0

        mag = min(abs(err), max_err) / max_err
        rc = int(RC_MIN_EFFECTIVE + mag * (RC_MAX - RC_MIN_EFFECTIVE))
        return rc if err > 0 else -rc

    def _send_rc(self, tello, lr: int, fb: int, ud: int, yaw: int) -> str:
        lr = self._clamp(lr, -100, 100)
        fb = self._clamp(fb, -100, 100)
        ud = self._clamp(ud, -100, 100)
        yaw = self._clamp(yaw, -100, 100)

        if not self._rate_limit_ok():
            return f"RATE_LIMIT lr={lr} fb={fb} ud={ud} yaw={yaw}"

        try:
            tello.send_rc_control(lr, fb, ud, yaw)
            self._mark_command()
            return f"RC lr={lr} fb={fb} ud={ud} yaw={yaw} state={self.state}"
        except Exception as e:
            return f"COMMAND_ERROR send_rc_control({lr},{fb},{ud},{yaw}): {e}"

    def stop(self, tello) -> str:
        return self._send_rc(tello, 0, 0, 0, 0)

    def update(self, tello, vp: VisionPacket) -> str:
        corners = set(vp.corners_seen or [])

        # -------------------------------------------------
        # 1) SEARCH MODE: no gate visible -> rotate slowly
        # -------------------------------------------------
        if len(corners) == 0 or vp.N_corners == 0 or not vp.gate_detected:
            self.state = "SEARCH"
            self._reset_stability()
            return self._send_rc(tello, 0, 0, 0, SEARCH_YAW_RC)

        # -------------------------------------------------
        # 2) ALIGN MODE: gate visible -> use x/y errors
        # -------------------------------------------------
        self.state = "ALIGN"
        self._update_stability(corners)

        if not self._stable_enough():
            return self._send_rc(tello, 0, 0, 0, 0)

        # Horizontal:
        # assume:
        #   x_err > 0  => gate center is to the right  -> move right
        #   x_err < 0  => gate center is to the left   -> move left
        lr = self._scale_error_to_rc(vp.x_err, X_DEADBAND_CM)

        # Vertical:
        # based on your earlier code:
        #   y_err < 0  => move up
        #   y_err > 0  => move down
        ud = 0
        if abs(vp.y_err) <= MAX_REASONABLE_Y_ERR_CM:
            ud = -self._scale_error_to_rc(vp.y_err, Y_DEADBAND_CM)
            # negative y_err -> positive ud (up)
            # positive y_err -> negative ud (down)

        # Optional yaw alignment
        yaw = 0
        if vp.theta_valid and abs(vp.theta) > YAW_DEADBAND_DEG:
            # tune sign if needed after testing
            yaw = -self._scale_error_to_rc(vp.theta, YAW_DEADBAND_DEG, max_err=25.0)

        # -------------------------------------------------
        # 3) FORWARD COMMIT:
        # only when all 4 corners are visible AND centered
        # -------------------------------------------------
        all_four_seen = corners == {"TL", "TR", "BL", "BR"}
        centered_xy = abs(vp.x_err) <= X_DEADBAND_CM and abs(vp.y_err) <= Y_DEADBAND_CM
        centered_yaw = (not vp.theta_valid) or (abs(vp.theta) <= YAW_DEADBAND_DEG)

        if all_four_seen and centered_xy and centered_yaw:
            self.centered_count += 1
        else:
            self.centered_count = 0

        if all_four_seen and self.centered_count >= CENTERED_FRAMES_REQUIRED:
            self.state = "FORWARD"
            return self._send_rc(tello, 0, FORWARD_RC, 0, 0)

        # -------------------------------------------------
        # 4) Otherwise: move toward center using one RC cmd
        # -------------------------------------------------
        return self._send_rc(tello, lr, 0, ud, yaw)
