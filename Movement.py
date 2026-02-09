"""
Tello gate-centering + pass-through controller (vision -> control -> RC commands)

Dependencies:
  pip install djitellopy opencv-python numpy
"""

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from djitellopy import Tello


# =============================================================================
# ============================ TUNABLE SETTINGS ===============================
# =============================================================================

# Gate marker IDs (set these to match your printed ArUco markers)
GATE_IDS = {10, 11, 12, 13}

# Smoothing / stability
EMA_ALPHA = 0.85
DEADBAND_XY = 0.04
DEADBAND_Z = 0.05

# RC limits (Tello uses roughly -100..100)
MAX_LR = 35
MAX_UD = 30
MAX_FB = 35
MAX_YAW = 35

# Gains (how aggressive corrections are)
KX = 40.0
KY = 35.0
KZ = 30.0
KTH = 80.0   # radians -> RC

# Marker quality thresholds
MIN_AREA = 250.0
MAX_SKEW = 0.75

# State machine timing / thresholds
LOST_TIMEOUT_S = 0.35
STABLE_FRAMES = 8
PASS_HOLD_FRAMES = 10
PASS_DURATION_S = 1.2

# Depth setpoint: measure A_avg at the “good pass-through distance”
A_TARGET = 5500.0
PASS_SPEED_FB = 35


# =============================================================================
# =============================== UTILITIES ===================================
# =============================================================================

def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def deadband(x: float, db: float) -> float:
    return 0.0 if abs(x) < db else x

def ema(prev: float, cur: float, alpha: float) -> float:
    return alpha * prev + (1 - alpha) * cur


# =============================================================================
# ====================== !!! VISION OUTPUT VARIABLES !!! ======================
# =============================================================================
# Your image recognition must produce THESE per frame (the controller needs them).
# In this script, we compute them from ArUco detections, but conceptually your
# "vision module" outputs the same values.
#
# !!! REQUIRED (minimum) !!!
#
# 1) gate_detected : bool
#    - True if we trust the gate measurement for this frame.
#
# 2) N_markers : int
#    - How many valid gate markers were used (1..4).
#
# 3) u_c, v_c : float (pixels)
#    - Estimated gate center location in the image (pixel coordinates).
#    - Used to compute pixel error e_x, e_y.
#
# 4) A_avg : float (pixels^2)
#    - Average marker area (depth proxy).
#    - Used to compute depth error e_z = A_TARGET - A_avg.
#
# !!! OPTIONAL (highly recommended) !!!
#
# 5) theta : float (radians) + theta_valid : bool
#    - Estimated yaw misalignment of the gate (how tilted/rotated it appears).
#    - Used for v_yaw command. Only valid when geometry is reliable.
#
# 6) marker_ids : set/list of ints
#    - Which marker IDs were detected; helps with multi-gate logic.
#
# =============================================================================


@dataclass
class VisionPacket:
    # ==================== !!! INPUT VARIABLES !!! ====================
    gate_detected: bool              # (bool) trusted detection this frame
    N_markers: int                   # (int) number of markers used
    u_c: float                       # (px) estimated gate center u-coordinate
    v_c: float                       # (px) estimated gate center v-coordinate
    A_avg: float                     # (px^2) average marker area (depth proxy)
    theta: float                     # (rad) yaw misalignment (optional)
    theta_valid: bool                # (bool) whether theta is valid
    marker_ids: List[int]            # IDs seen this frame
    # ================================================================


# =============================================================================
# ============================= VISION MODULE =================================
# =============================================================================

class GateVision:
    def __init__(self):
        # Use a standard dictionary (you can change this if your markers differ)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)

    @staticmethod
    def _poly_area(corners: np.ndarray) -> float:
        """
        corners: shape (4,2) as float32
        returns polygon area in pixel^2
        """
        x = corners[:, 0]
        y = corners[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @staticmethod
    def _skew_metric(corners: np.ndarray) -> float:
        """
        Rough skew metric based on side length differences.
        0 is "square-ish", larger is more skewed.
        """
        d01 = np.linalg.norm(corners[1] - corners[0])
        d12 = np.linalg.norm(corners[2] - corners[1])
        d23 = np.linalg.norm(corners[3] - corners[2])
        d30 = np.linalg.norm(corners[0] - corners[3])
        sides = np.array([d01, d12, d23, d30]) + 1e-6
        return float(np.std(sides) / np.mean(sides))

    def process(self, frame_bgr: np.ndarray) -> VisionPacket:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return VisionPacket(
                gate_detected=False,
                N_markers=0,
                u_c=0.0, v_c=0.0,
                A_avg=0.0,
                theta=0.0,
                theta_valid=False,
                marker_ids=[]
            )

        ids = ids.flatten().tolist()

        # Collect valid markers for THIS gate
        marker_centers: List[Tuple[float, float]] = []
        marker_areas: List[float] = []
        valid_ids: List[int] = []

        # Also keep centers by id for yaw estimation
        centers_by_id: Dict[int, Tuple[float, float]] = {}

        for marker_corners, mid in zip(corners_list, ids):
            if mid not in GATE_IDS:
                continue

            c = marker_corners.reshape(4, 2).astype(np.float32)

            area = self._poly_area(c)
            skew = self._skew_metric(c)

            if area < MIN_AREA:
                continue
            if skew > MAX_SKEW:
                continue

            u = float(np.mean(c[:, 0]))
            v = float(np.mean(c[:, 1]))

            marker_centers.append((u, v))
            marker_areas.append(area)
            valid_ids.append(mid)
            centers_by_id[mid] = (u, v)

        N = len(marker_centers)
        if N == 0:
            return VisionPacket(
                gate_detected=False,
                N_markers=0,
                u_c=0.0, v_c=0.0,
                A_avg=0.0,
                theta=0.0,
                theta_valid=False,
                marker_ids=[]
            )

        # Gate center estimate (average of marker centers)
        u_c = float(np.mean([p[0] for p in marker_centers]))
        v_c = float(np.mean([p[1] for p in marker_centers]))
        A_avg = float(np.mean(marker_areas))

        # Optional yaw estimation (theta)
        # If we have at least two markers, estimate a horizontal direction by picking
        # the leftmost and rightmost marker centers and measuring the line angle.
        theta_valid = False
        theta = 0.0
        if N >= 2:
            pts = marker_centers
            left = min(pts, key=lambda p: p[0])
            right = max(pts, key=lambda p: p[0])
            du = right[0] - left[0]
            dv = right[1] - left[1]
            if abs(du) > 1e-3:
                theta = math.atan2(dv, du)  # radians
                theta_valid = True

        return VisionPacket(
            gate_detected=True,
            N_markers=N,
            u_c=u_c, v_c=v_c,
            A_avg=A_avg,
            theta=theta,
            theta_valid=theta_valid,
            marker_ids=valid_ids
        )


# =============================================================================
# ============================ CONTROLLER MODULE ==============================
# =============================================================================

class Controller:
    def __init__(self):
        # filtered errors (EMA)
        self.ex_f = 0.0
        self.ey_f = 0.0
        self.ez_f = 0.0
        self.theta_f = 0.0

        # state machine
        self.state = "SEARCH"
        self.last_seen_t = time.time()
        self.stable_count = 0
        self.pass_hold = 0
        self.pass_start_t = 0.0

    def update(self, vp: VisionPacket, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """
        Returns RC commands: (lr, fb, ud, yaw) as ints.
        """
        now = time.time()
        u0, v0 = frame_w / 2.0, frame_h / 2.0

        # If detection lost, handle timeout
        if not vp.gate_detected:
            if (now - self.last_seen_t) > LOST_TIMEOUT_S:
                self.state = "SEARCH"
                self.stable_count = 0
                self.pass_hold = 0
            # SEARCH behavior: gentle yaw scan
            if self.state == "SEARCH":
                yaw = int(20 * math.sin(now * 1.2))
                return (0, 0, 0, yaw)
            return (0, 0, 0, 0)

        # We have a detection
        self.last_seen_t = now

        # Normalize pixel errors
        ex = (vp.u_c - u0) / (frame_w / 2.0)   # left/right image error normalized
        ey = (vp.v_c - v0) / (frame_h / 2.0)   # up/down image error normalized
        ez = (A_TARGET - vp.A_avg) / max(A_TARGET, 1.0)  # depth error normalized

        # Smooth errors
        self.ex_f = ema(self.ex_f, ex, EMA_ALPHA)
        self.ey_f = ema(self.ey_f, ey, EMA_ALPHA)
        self.ez_f = ema(self.ez_f, ez, EMA_ALPHA)
        if vp.theta_valid:
            self.theta_f = ema(self.theta_f, vp.theta, EMA_ALPHA)

        # Apply deadbands
        ex_cmd = deadband(self.ex_f, DEADBAND_XY)
        ey_cmd = deadband(self.ey_f, DEADBAND_XY)
        ez_cmd = deadband(self.ez_f, DEADBAND_Z)

        # State transitions based on stability / marker count
        if self.state in ("SEARCH", "ACQUIRE"):
            self.state = "ACQUIRE"
            if vp.N_markers >= 3:
                self.stable_count += 1
            else:
                self.stable_count = max(0, self.stable_count - 1)
            if self.stable_count >= STABLE_FRAMES:
                self.state = "ALIGN_APPROACH"
                self.pass_hold = 0

        # Alignment quality check (for PASS trigger)
        aligned_xy = (abs(ex_cmd) < 0.06) and (abs(ey_cmd) < 0.06)
        aligned_z = abs(ez_cmd) < 0.08

        if self.state == "ALIGN_APPROACH":
            if aligned_xy and aligned_z and vp.N_markers >= 3:
                self.pass_hold += 1
            else:
                self.pass_hold = 0

            if self.pass_hold >= PASS_HOLD_FRAMES:
                self.state = "PASS_THROUGH"
                self.pass_start_t = now

        if self.state == "PASS_THROUGH":
            if (now - self.pass_start_t) < PASS_DURATION_S:
                # Go straight forward; keep only big corrections if you want (here: none)
                return (0, PASS_SPEED_FB, 0, 0)
            else:
                self.state = "SEARCH"
                self.stable_count = 0
                self.pass_hold = 0
                return (0, 0, 0, 0)

        # Compute RC commands in ACQUIRE / ALIGN_APPROACH
        # left/right strafe
        lr = clip(KX * ex_cmd, -MAX_LR, MAX_LR)
        # up/down: negative because +ey means gate appears lower in image, so drone should go up
        ud = clip(-KY * ey_cmd, -MAX_UD, MAX_UD)

        # forward/back: only move forward when in ALIGN_APPROACH and detection is decent
        fb_raw = KZ * ez_cmd
        if self.state == "ACQUIRE":
            fb = 0.0
        else:
            fb = clip(fb_raw, 0, MAX_FB)  # forward only

        # yaw (optional)
        if vp.theta_valid and self.state == "ALIGN_APPROACH":
            yaw = clip(-KTH * self.theta_f, -MAX_YAW, MAX_YAW)
        else:
            yaw = 0.0

        return (int(round(lr)), int(round(fb)), int(round(ud)), int(round(yaw)))


# =============================================================================
# ================================ MAIN LOOP ==================================
# =============================================================================

def main():
    tello = Tello()
    tello.connect()
    print("Battery:", tello.get_battery(), "%")

    tello.streamon()
    frame_reader = tello.get_frame_read()

    vision = GateVision()
    ctrl = Controller()

    tello.takeoff()
    tello.send_rc_control(0, 0, 0, 0)

    try:
        while True:
            frame = frame_reader.frame
            if frame is None:
                continue

            h, w = frame.shape[:2]

            vp = vision.process(frame)
            lr, fb, ud, yaw = ctrl.update(vp, w, h)

            tello.send_rc_control(lr, fb, ud, yaw)

            # Debug overlay (optional)
            if vp.gate_detected:
                cv2.circle(frame, (int(vp.u_c), int(vp.v_c)), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"N={vp.N_markers} A={vp.A_avg:.0f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if vp.theta_valid:
                    cv2.putText(frame, f"theta={vp.theta:.2f} rad",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"STATE={ctrl.state} rc(lr,fb,ud,yaw)=({lr},{fb},{ud},{yaw})",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Tello Gate Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.2)
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
