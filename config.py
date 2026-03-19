"""
config.py
Configuration and constants for Tello drone autonomous navigation.
"""

# ========== MARKER & GATE CONFIGURATION ==========
# ArUCo marker IDs at each corner of the gate
MARKER_IDS = {
    'top_left': 1,
    'top_right': 2,
    'bottom_left': 3,
    'bottom_right': 4,
}

# Gate physical dimensions (in centimeters)
GATE_WIDTH_CM = 200         # Distance between left and right markers
GATE_HEIGHT_CM = 150        # Distance between top and bottom markers

# Expected marker size (in cm) - used for pose estimation
MARKER_SIZE_CM = 15


# ========== VISION PROCESSING CONFIG ==========
# Frame skipping to reduce computation load
FRAME_SKIP = 12  # Process every 12th frame from ~30 fps stream

# ArUCo dictionary and detection parameters
ARUCO_DICT_NAME = "DICT_4X4_50"  # Standard 4x4 dictionary with 50 markers
MARKER_DETECTION_CONFIDENCE = 0.8  # Minimum confidence for marker detection

# Camera calibration (Tello typical values - can be refined)
# These are approximate values; real calibration is recommended
CAMERA_FOCAL_LENGTH = 921  # pixels
CAMERA_CENTER_X = 480
CAMERA_CENTER_Y = 360
CAMERA_MATRIX = [
    [CAMERA_FOCAL_LENGTH, 0, CAMERA_CENTER_X],
    [0, CAMERA_FOCAL_LENGTH, CAMERA_CENTER_Y],
    [0, 0, 1]
]


# ========== PLANNER/CONTROL THRESHOLDS ==========
# Alignment tolerances (drone considers gate "aligned" when within these)
X_ERROR_THRESHOLD_CM = 15      # Lateral alignment threshold
Y_ERROR_THRESHOLD_CM = 15      # Vertical alignment threshold
Z_ERROR_THRESHOLD_CM = 20      # Depth alignment threshold
YAW_ERROR_THRESHOLD_DEG = 10    # Rotation alignment threshold

# Confidence threshold for proceeding with approach
MIN_CONFIDENCE_TO_APPROACH = 0.7

# State machine thresholds
APPROACH_DISTANCE_CM = 100  # When to switch from ALIGN to APPROACH state


# ========== DRONE CONTROL CONFIG ==========
# Tello connection parameters
TELLO_IP = "192.168.10.1"
TELLO_PORT = 8889
TELLO_COMMAND_TIMEOUT = 5  # seconds

# Movement commands (in cm or degrees)
MOVE_STEP_CM = 30           # Distance for each movement command
ROTATE_STEP_DEG = 15        # Angle for each rotation command
APPROACH_STEP_CM = 50       # Distance when approaching gate

# Velocity settings (Tello range: 10-100 cm/s)
DEFAULT_VELOCITY = 50       # cm/s

# Safety parameters
MAX_ALTITUDE_CM = 250       # Maximum height above ground
MIN_ALTITUDE_CM = 50        # Minimum safe height


# ========== OPERATION MODES ==========
OPERATION_MODES = {
    'SEARCH': 0,             # Searching for gate markers
    'ALIGN': 1,              # Aligning drone with gate
    'APPROACH': 2,           # Moving toward gate
    'COMMIT': 3,             # Flying through the gate
    'LAND': 4,               # Landing
}

# Timeout for each state (seconds)
STATE_TIMEOUT_SEC = 30


# ========== DEBUG & LOGGING ==========
DEBUG_MODE = True           # Enable debug logging
SAVE_FRAMES = False         # Save processed frames to disk
FRAME_OUTPUT_DIR = "./frames_output"

# Vision output verbosity
VERBOSE_VISION = True       # Print detailed vision diagnostics
VERBOSE_PLANNING = True     # Print planning decisions
