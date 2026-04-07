import cv2
import numpy as np
import time


class VisionProcessor:
    def __init__(self):
        # Marker IDs on the gate:
        # 1 = top left, 2 = top right, 3 = bottom left, 4 = bottom right
        self.marker_positions = {
            1: (-40.64, 24.77, 0),
            2: (40.64, 24.77, 0.0),
            3: (-40.64, -40.64, 0.0),
            4: (40.64, -40.64, 0.0),
        }

        self.expected_ids = set(self.marker_positions.keys())

        # Marker size in cm
        self.marker_size_cm = 10.0

        # Placeholder camera matrix / distortion
        # Replace with real calibration values later
        self.camera_matrix = np.array(
            [
                [920.0, 0.0, 480.0],
                [0.0, 920.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

        # Debug throttling
        self.last_debug_time = 0.0
        self.debug_interval_s = 0.5

        self._reset_public_outputs()
        print("[INFO] VisionProcessor initialized")

    def _reset_public_outputs(self):
        self.gate_detected = False
        self.gate_name = ""
        self.N_corners = 0
        self.u_c = 0.0
        self.v_c = 0.0
        self.A_avg = 0.0
        self.theta = 0.0
        self.theta_valid = False
        self.corners_seen = []
        self.marker_ids = []

        # Pose errors
        self.x_err = 0.0
        self.y_err = 0.0
        self.z_err = 0.0

    def _debug_print(self, msg):
        now = time.time()
        if now - self.last_debug_time >= self.debug_interval_s:
            print(msg)
            self.last_debug_time = now

    def process_frame(self, frame):
        self._reset_public_outputs()
        output = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            cv2.putText(
                output,
                "No markers detected",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            return output

        ids = ids.flatten().tolist()

        cv2.aruco.drawDetectedMarkers(output, corners, np.array(ids, dtype=np.int32))

        valid_corners = []
        valid_ids = []
        ignored_ids = []

        for marker_corner, marker_id in zip(corners, ids):
            if marker_id in self.expected_ids:
                valid_corners.append(marker_corner)
                valid_ids.append(marker_id)
            else:
                ignored_ids.append(marker_id)

        self._draw_debug_text(output, ids, valid_ids, ignored_ids)

        if len(valid_ids) == 0:
            cv2.putText(
                output,
                "Only unknown markers detected",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return output

        # Gate is detected as soon as we see valid gate markers
        self.gate_detected = True
        self.gate_name = "GATE_1"

        self._draw_marker_centers(output, valid_corners, valid_ids)

        # Public outputs for controller
        self.marker_ids = valid_ids[:]
        self.N_corners = len(valid_ids)
        self.corners_seen = [self._id_to_corner_name(mid) for mid in valid_ids]

        self.u_c, self.v_c = self._compute_gate_center(valid_corners)
        self.A_avg = self._compute_average_marker_area(valid_corners)

        pose = self._estimate_gate_pose(valid_corners, valid_ids, output)

        if pose is not None:
            x_err, y_err, z_err, yaw_err = pose

            self.x_err = x_err
            self.y_err = y_err
            self.z_err = z_err
            self.theta = yaw_err
            self.theta_valid = True

            debug_msg = (
                f"x={x_err:.1f}cm y={y_err:.1f}cm z={z_err:.1f}cm yaw={yaw_err:.1f}deg"
            )
            self._debug_print(f"[DEBUG] Vision: IDs={valid_ids}, {debug_msg}")

            cv2.putText(
                output,
                debug_msg,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )
        else:
            self.theta_valid = False
            cv2.putText(
                output,
                "Pose estimate unavailable",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 165, 255),
                2,
            )

        cv2.putText(
            output,
            f"Gate center: ({self.u_c:.1f}, {self.v_c:.1f})",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            output,
            f"A_avg={self.A_avg:.1f}  corners={self.corners_seen}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )

        return output

    def _id_to_corner_name(self, marker_id):
        if marker_id == 1:
            return "TL"
        if marker_id == 2:
            return "TR"
        if marker_id == 3:
            return "BL"
        if marker_id == 4:
            return "BR"
        return f"ID{marker_id}"

    def _compute_gate_center(self, corners):
        all_pts = np.vstack([corner[0] for corner in corners]).astype(np.float32)
        u_c = float(np.mean(all_pts[:, 0]))
        v_c = float(np.mean(all_pts[:, 1]))
        return u_c, v_c

    def _compute_average_marker_area(self, corners):
        areas = []
        for corner in corners:
            pts = corner[0].astype(np.float32)
            areas.append(cv2.contourArea(pts))
        return float(np.mean(areas)) if areas else 0.0

    def _draw_debug_text(self, frame, all_ids, valid_ids, ignored_ids):
        cv2.putText(
            frame,
            f"Detected IDs: {all_ids}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Valid IDs: {valid_ids}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        if ignored_ids:
            cv2.putText(
                frame,
                f"Ignored IDs: {ignored_ids}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

    def _draw_marker_centers(self, frame, corners, ids):
        for marker_corner, marker_id in zip(corners, ids):
            pts = marker_corner[0]
            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))

            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.putText(
                frame,
                f"{self._id_to_corner_name(marker_id)} ({marker_id})",
                (center_x + 10, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
            )

    def _estimate_gate_pose(self, corners, detected_ids, frame):
        if len(detected_ids) == 0:
            return None

        if len(detected_ids) == 1:
            return self._single_marker_pose(corners[0], detected_ids[0], frame)

        return self._multi_marker_pose(corners, detected_ids, frame)

    def _single_marker_pose(self, corner, marker_id, frame):
        if marker_id not in self.marker_positions:
            self._debug_print(f"[WARNING] Unknown marker ID in single marker pose: {marker_id}")
            return None

    # 2D image points from detected marker
        image_points = corner[0].astype(np.float32)

    # 3D object points for a single square marker centered at origin
        half = self.marker_size_cm / 2.0
        object_points = np.array(
            [
                [-half,  half, 0.0],   # top-left
                [ half,  half, 0.0],   # top-right
                [ half, -half, 0.0],   # bottom-right
                [-half, -half, 0.0],   # bottom-left
            ],
            dtype=np.float32,
        )

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        cv2.drawFrameAxes(
            frame,
            self.camera_matrix,
            self.dist_coeffs,
            rvec,
            tvec,
            self.marker_size_cm * 0.5,
        )

        x_err = float(tvec[0][0])
        y_err = float(tvec[1][0])
        z_err = float(tvec[2][0])
        yaw_err = self._yaw_from_rvec(rvec)

        return x_err, y_err, z_err, yaw_err

    def _multi_marker_pose(self, corners, detected_ids, frame):
        object_points = []
        image_points = []

        for marker_corner, marker_id in zip(corners, detected_ids):
            if marker_id not in self.marker_positions:
                continue

            obj_center = np.array(self.marker_positions[marker_id], dtype=np.float32)
            half = self.marker_size_cm / 2.0

            marker_obj_pts = np.array(
                [
                    [obj_center[0] - half, obj_center[1] + half, obj_center[2]],
                    [obj_center[0] + half, obj_center[1] + half, obj_center[2]],
                    [obj_center[0] + half, obj_center[1] - half, obj_center[2]],
                    [obj_center[0] - half, obj_center[1] - half, obj_center[2]],
                ],
                dtype=np.float32,
            )

            marker_img_pts = marker_corner[0].astype(np.float32)

            object_points.extend(marker_obj_pts)
            image_points.extend(marker_img_pts)

        if len(object_points) < 4:
            return None

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        cv2.drawFrameAxes(
            frame,
            self.camera_matrix,
            self.dist_coeffs,
            rvec,
            tvec,
            self.marker_size_cm,
        )

        x_err = float(tvec[0][0])
        y_err = float(tvec[1][0])
        z_err = float(tvec[2][0])
        yaw_err = self._yaw_from_rvec(rvec)

        return x_err, y_err, z_err, yaw_err

    def _yaw_from_rvec(self, rvec):
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        yaw = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
        return float(yaw)