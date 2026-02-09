"""
================================================================================
Tello Gate Controller (CORNER-AWARE)
================================================================================

What you asked for:
- Each gate has 4 barcodes/markers, each one represents a SPECIFIC CORNER
  (TL, TR, BL, BR).
- You want the movement algorithm to properly handle cases like:
  - only top two corners visible (TL + TR)
  - only bottom two corners visible (BL + BR)
  - only left side visible (TL + BL)
  - only right side visible (TR + BR)
- Completely rewritten script with that functionality.

Dependencies:
  pip install djitellopy opencv-python numpy

IMPORTANT:
- This is a FULL script: vision + movement + Tello RC control.
- If you already have a different detector, you can keep ONLY the Controller
  and feed it the same VisionPacket fields.

================================================================================
"""


import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from djitellopy import Tello


# =============================================================================
# ============================== GATE DEFINITIONS =============================
# =============================================================================
# Each gate has 4 marker IDs mapped to corners.
# You MUST set these to match your printed barcodes.
#
# Example scheme:
# Gate 1: 10 TL, 11 TR, 12 BL, 13 BR
# Gate 2: 20 TL, 21 TR, 22 BL, 23 BR
#
# If you only have ONE gate, just keep one entry.
# =============================================================================

GATES: Dict[str, Dict[str, int]] = {
    "GATE_1": {"TL": 10, "TR": 11, "BL": 12, "BR": 13},
    # "GATE_2": {"TL": 20, "TR": 21, "BL": 22, "BR": 23},
}

# Convenience: reverse lookup id -> (gate_name, corner)
ID_TO_GATE_CORNER: Dict[int, Tuple[str, str]] = {}
for gname, cmap in GATES.items():
    for corner, mid in cmap.items():
        ID_TO_GATE_CORNER[mid] = (gname, corner)


# =============================================================================
# ============================= CONTROL PARAMETERS ============================
# =============================================================================

# --- smoothing / stability ---
EMA_ALPHA = 0.85
DEADBAND_XY = 0.04
DEADBAND_Z = 0.05

# --- RC limits (Tello uses roughly -100..100) ---
MAX_LR = 35
MAX_UD = 30
MAX_FB = 35
MAX_YAW = 35

# --- gains ---
KX = 40.0     # strafe gain
KY = 35.0     # up/down gain
KZ = 30.0     # forward gain
KTH = 80.0    # yaw gain (radians -> RC)

# --- marker quality thresholds ---
MIN_AREA = 250.0
MAX_SKEW = 0.75

# --- state machine ---
LOST_TIMEOUT_S = 0.35
STABLE_FRAMES = 8
PASS_HOLD_FRAMES = 10
PASS_DURATION_S = 1.2
PASS_SPEED_FB = 35

# --- depth setpoint (you should calibrate this) ---
A_TARGET = 5500.0

# --- center correction fallback factors ---
# If we only see one edge, we shift midpoint by HALF of gate width/height.
# If width/height is unknown (never saw 3-4 corners), we keep a conservative
# default based on "typical" pixels (you can tune these).
DEFAULT_GATE_W_PX = 220.0
DEFAULT_GATE_H_PX = 220.0


# =============================================================================
# =============================== UTILITIES ===================================
# =============================================================================

def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def deadband(x: float, db: float) -> float:
    return 0.0 if abs(x) < db else x

def ema(prev: float, cur: float, alpha: float) -> float:
    return alpha * prev + (1 - alpha) * cur

def dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return float(math.hypot(p[0] - q[0], p[1] - q[1]))


# =============================================================================
# ====================== !!! VISION OUTPUT VARIABLES !!! ======================
# =============================================================================
# Your detector (ArUco, AprilTag, ML model, etc.) can be ANYTHING,
# as long as it outputs these per frame.
#
# ========================= !!! REQUIRED INPUTS !!! ===========================
# gate_detected : bool
#   - True if we have a trusted gate measurement in THIS frame.
#
# gate_name : str
#   - Which gate we are tracking (ex: "GATE_1").
#
# N_corners : int
#   - Number of CORNERS seen for that gate (1..4).
#
# u_c, v_c : float (pixels)
#   - Estimated TRUE gate center in pixel coordinates.
#   - (Corner-aware correction happens BEFORE this is output!)
#
# A_avg : float (pixels^2)
#   - Average marker area for seen corners (depth proxy).
#
# ========================= !!! OPTIONAL INPUTS !!! ===========================
# theta : float (radians), theta_valid : bool
#   - Yaw misalignment estimate; if invalid, controller sets yaw=0.
#
# corners_seen : List[str]  e.g. ["TL","TR"]
# marker_ids : List[int]    e.g. [10,11]
# =============================================================================

@dataclass
class VisionPacket:
    # ==================== !!! INPUT VARIABLES !!! ====================
    gate_detected: bool
    gate_name: str

    N_corners: int
    u_c: float
    v_c: float
    A_avg: float

    theta: float
    theta_valid: bool

    corners_seen: List[str]
    marker_ids: List[int]
    # ================================================================


# =============================================================================
# ============================= CORNER-AWARE VISION ===========================
# =============================================================================

class CornerAwareGateVision:
    """
    Uses ArUco detections and the corner mapping to:
    - group detections by gate
    - compute a corrected gate center even when only one edge is visible
    - choose the "best" gate each frame (closest/largest area)
    """

    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)

        # Keep last-known gate width/height in pixels (per gate)
        self.last_gate_w_px: Dict[str, float] = {g: DEFAULT_GATE_W_PX for g in GATES}
        self.last_gate_h_px: Dict[str, float] = {g: DEFAULT_GATE_H_PX for g in GATES}

    @staticmethod
    def _poly_area(corners: np.ndarray) -> float:
        x = corners[:, 0]
        y = corners[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @staticmethod
    def _skew_metric(corners: np.ndarray) -> float:
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
            return VisionPacket(False, "", 0, 0.0, 0.0, 0.0, 0.0, False, [], [])

        ids = ids.flatten().tolist()

        # Group detections by gate
        # gate_dets[gate_name][corner] = (center_uv, area)
        gate_dets: Dict[str, Dict[str, Tuple[Tuple[float, float], float, int]]] = {}

        for marker_corners, mid in zip(corners_list, ids):
            if mid not in ID_TO_GATE_CORNER:
                continue

            gname, corner_name = ID_TO_GATE_CORNER[mid]

            c = marker_corners.reshape(4, 2).astype(np.float32)
            area = self._poly_area(c)
            skew = self._skew_metric(c)

            if area < MIN_AREA:
                continue
            if skew > MAX_SKEW:
                continue

            u = float(np.mean(c[:, 0]))
            v = float(np.mean(c[:, 1]))

            if gname not in gate_dets:
                gate_dets[gname] = {}
            gate_dets[gname][corner_name] = ((u, v), area, mid)

        if not gate_dets:
            return VisionPacket(False, "", 0, 0.0, 0.0, 0.0, 0.0, False, [], [])

        # Choose the "best" gate to follow THIS frame:
        # We’ll pick the one with the largest total detected area (closest gate).
        best_gate = None
        best_score = -1.0
        for gname, dets in gate_dets.items():
            score = sum(info[1] for info in dets.values())  # sum of areas
            if score > best_score:
                best_score = score
                best_gate = gname

        assert best_gate is not None
        dets = gate_dets[best_gate]

        corners_seen = sorted(dets.keys())
        marker_ids = [dets[c][2] for c in corners_seen]

        # Compute raw midpoints and update last-known width/height when possible
        # If both top corners exist -> width estimate from TL<->TR
        if "TL" in dets and "TR" in dets:
            self.last_gate_w_px[best_gate] = dist(dets["TL"][0], dets["TR"][0])
        elif "BL" in dets and "BR" in dets:
            self.last_gate_w_px[best_gate] = dist(dets["BL"][0], dets["BR"][0])

        # If both left corners exist -> height estimate from TL<->BL
        if "TL" in dets and "BL" in dets:
            self.last_gate_h_px[best_gate] = dist(dets["TL"][0], dets["BL"][0])
        elif "TR" in dets and "BR" in dets:
            self.last_gate_h_px[best_gate] = dist(dets["TR"][0], dets["BR"][0])

        gate_w = self.last_gate_w_px[best_gate]
        gate_h = self.last_gate_h_px[best_gate]

        # Average area for depth proxy
        A_avg = float(np.mean([info[1] for info in dets.values()]))

        # ------------------------------
        # CORNER-AWARE CENTER ESTIMATION
        # ------------------------------
        # Goal: output TRUE gate center (u_c, v_c).
        #
        # Cases:
        # - If we have 3 or 4 corners -> simplest: average of all seen corners,
        #   (it tends to be close to actual center).
        # - If we have exactly 2 corners:
        #     - If they are top edge (TL+TR): center is midpoint shifted DOWN by h/2
        #     - If bottom edge (BL+BR): center is midpoint shifted UP by h/2
        #     - If left edge (TL+BL): center is midpoint shifted RIGHT by w/2
        #     - If right edge (TR+BR): center is midpoint shifted LEFT by w/2
        #     - If diagonal (TL+BR or TR+BL): center is midpoint (already center-ish)
        # - If only 1 corner:
        #     - We can’t know the center exactly, but we can shift by half width/height
        #       using the corner identity.
        #
        # Pixel coordinates: u right+, v down+
        # ------------------------------

        seen = set(corners_seen)

        def midpoint(p: Tuple[float, float], q: Tuple[float, float]) -> Tuple[float, float]:
            return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)

        u_c, v_c = 0.0, 0.0
        N = len(dets)

        if N >= 3:
            pts = [info[0] for info in dets.values()]
            u_c = float(np.mean([p[0] for p in pts]))
            v_c = float(np.mean([p[1] for p in pts]))

        elif N == 2:
            # Pick the two corners
            c1, c2 = corners_seen[0], corners_seen[1]
            p1, p2 = dets[c1][0], dets[c2][0]
            um, vm = midpoint(p1, p2)

            # Edge-only corrections
            if seen == {"TL", "TR"}:         # top edge
                u_c, v_c = um, vm + gate_h / 2.0
            elif seen == {"BL", "BR"}:       # bottom edge
                u_c, v_c = um, vm - gate_h / 2.0
            elif seen == {"TL", "BL"}:       # left edge
                u_c, v_c = um + gate_w / 2.0, vm
            elif seen == {"TR", "BR"}:       # right edge
                u_c, v_c = um - gate_w / 2.0, vm
            else:
                # Diagonal pair: midpoint is already close to center
                u_c, v_c = um, vm

        else:  # N == 1
            c1 = corners_seen[0]
            p = dets[c1][0]
            # Shift from corner to center using half width/height
            if c1 == "TL":
                u_c, v_c = p[0] + gate_w / 2.0, p[1] + gate_h / 2.0
            elif c1 == "TR":
                u_c, v_c = p[0] - gate_w / 2.0, p[1] + gate_h / 2.0
            elif c1 == "BL":
                u_c, v_c = p[0] + gate_w / 2.0, p[1] - gate_h / 2.0
            else:  # "BR"
                u_c, v_c = p[0] - gate_w / 2.0, p[1] - gate_h / 2.0

        # ------------------------------
        # Yaw estimation (OPTIONAL)
        # ------------------------------
        # If we have a top edge or bottom edge, we can estimate tilt angle from it.
        theta_valid = False
        theta = 0.0
        if "TL" in dets and "TR" in dets:
            left = dets["TL"][0]
            right = dets["TR"][0]
            theta = math.atan2(right[1] - left[1], right[0] - left[0])
            theta_valid = True
        elif "BL" in dets and "BR" in dets:
            left = dets["BL"][0]
            right = dets["BR"][0]
            theta = math.atan2(right[1] - left[1], right[0] - left[0])
            theta_valid = True

        return VisionPacket(
            gate_detected=True,
            gate_name=best_gate,
            N_corners=N,
            u_c=float(u_c),
            v_c=float(v_c),
            A_avg=A_avg,
            theta=float(theta),
            theta_valid=theta_valid,
            corners_seen=corners_seen,
            marker_ids=marker_ids,
        )


# =============================================================================
# ============================== MOVEMENT CONTROLLER ==========================
# =============================================================================

class Controller:
    """
    Detector-agnostic movement controller:
    feeds on VisionPacket values, outputs RC commands (lr, fb, ud, yaw).
    """

    def __init__(self):
        # filtered errors
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
        now = time.time()
        u0, v0 = frame_w / 2.0, frame_h / 2.0

        # ================= LOST DETECTION HANDLING =================
        if not vp.gate_detected:
            if (now - self.last_seen_t) > LOST_TIMEOUT_S:
                self.state = "SEARCH"
                self.stable_count = 0
                self.pass_hold = 0

            # SEARCH: gentle yaw scan
            if self.state == "SEARCH":
                yaw = int(20 * math.sin(now * 1.2))
                return (0, 0, 0, yaw)

            return (0, 0, 0, 0)

        # ================= HAVE DETECTION =================
        self.last_seen_t = now

        # Normalized errors
        ex = (vp.u_c - u0) / (frame_w / 2.0)                 # left/right
        ey = (vp.v_c - v0) / (frame_h / 2.0)                 # up/down
        ez = (A_TARGET - vp.A_avg) / max(A_TARGET, 1.0)      # depth proxy

        # Smooth
        self.ex_f = ema(self.ex_f, ex, EMA_ALPHA)
        self.ey_f = ema(self.ey_f, ey, EMA_ALPHA)
        self.ez_f = ema(self.ez_f, ez, EMA_ALPHA)
        if vp.theta_valid:
            self.theta_f = ema(self.theta_f, vp.theta, EMA_ALPHA)

        # Deadband
        ex_cmd = deadband(self.ex_f, DEADBAND_XY)
        ey_cmd = deadband(self.ey_f, DEADBAND_XY)
        ez_cmd = deadband(self.ez_f, DEADBAND_Z)

        # ================= STATE MACHINE =================
        if self.state in ("SEARCH", "ACQUIRE"):
            self.state = "ACQUIRE"

            # Require stability + at least 2 corners, ideally 3+
            if vp.N_corners >= 3:
                self.stable_count += 1
            else:
                self.stable_count = max(0, self.stable_count - 1)

            if self.stable_count >= STABLE_FRAMES:
                self.state = "ALIGN_APPROACH"
                self.pass_hold = 0

        # Alignment checks for PASS
        aligned_xy = (abs(ex_cmd) < 0.06) and (abs(ey_cmd) < 0.06)
        aligned_z = abs(ez_cmd) < 0.08

        if self.state == "ALIGN_APPROACH":
            if aligned_xy and aligned_z and vp.N_corners >= 3:
                self.pass_hold += 1
            else:
                self.pass_hold = 0

            if self.pass_hold >= PASS_HOLD_FRAMES:
                self.state = "PASS_THROUGH"
                self.pass_start_t = now

        if self.state == "PASS_THROUGH":
            if (now - self.pass_start_t) < PASS_DURATION_S:
                return (0, PASS_SPEED_FB, 0, 0)
            else:
                self.state = "SEARCH"
                self.stable_count = 0
                self.pass_hold = 0
                return (0, 0, 0, 0)

        # ================= CONTROL LAW =================
        # Strafe left/right
        lr = clip(KX * ex_cmd, -MAX_LR, MAX_LR)

        # Up/down (note: +v is down in images, so invert for drone movement)
        # Because our vision is corner-aware, ey_cmd is much more reliable even
        # when only top or bottom corners are seen.
        ud = clip(-KY * ey_cmd, -MAX_UD, MAX_UD)

        # Forward/back:
        # - In ACQUIRE: do not approach (safer) unless you want it to creep forward
        # - In ALIGN_APPROACH: allow forward
        if self.state == "ACQUIRE":
            fb = 0.0
        else:
            fb = clip(KZ * ez_cmd, 0, MAX_FB)

        # Yaw
        if vp.theta_valid and self.state == "ALIGN_APPROACH":
            yaw = clip(-KTH * self.theta_f, -MAX_YAW, MAX_YAW)
        else:
            yaw = 0.0

        return (int(round(lr)), int(round(fb)), int(round(ud)), int(round(yaw)))


# =============================================================================
# ================================== MAIN =====================================
# =============================================================================

def main():
    tello = Tello()
    tello.connect()
    print("Battery:", tello.get_battery(), "%")

    tello.streamon()
    fr = tello.get_frame_read()

    vision = CornerAwareGateVision()
    ctrl = Controller()

    tello.takeoff()
    tello.send_rc_control(0, 0, 0, 0)

    try:
        while True:
            frame = fr.frame
            if frame is None:
                continue

            h, w = frame.shape[:2]
            vp = vision.process(frame)

            lr, fb, ud, yaw = ctrl.update(vp, w, h)
            tello.send_rc_control(lr, fb, ud, yaw)

            # -------------------- Debug overlay --------------------
            cv2.putText(frame, f"STATE={ctrl.state}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

            cv2.putText(frame, f"RC(lr,fb,ud,yaw)=({lr},{fb},{ud},{yaw})", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            if vp.gate_detected:
                cv2.circle(frame, (int(vp.u_c), int(vp.v_c)), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"Gate={vp.gate_name} corners={vp.corners_seen}", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(frame, f"N={vp.N_corners} A_avg={vp.A_avg:.0f}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                if vp.theta_valid:
                    cv2.putText(frame, f"theta={vp.theta:.2f} rad", (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No gate detected", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.imshow("Corner-Aware Tello Gate Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.2)
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
