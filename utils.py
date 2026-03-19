"""
utils.py
Shared utilities, data structures, and helper functions.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import json


@dataclass
class VisionOutput:
    """
    Simplified vision output from the vision module.
    This is the main data structure passed from vision.py to planner.py
    """
    gate_detected: bool
    visible_ids: List[int]           # IDs of detected markers
    x_error_cm: float                # Lateral error (cm)
    y_error_cm: float                # Vertical error (cm)
    z_error_cm: float                # Depth error (cm)
    yaw_error_deg: float             # Rotation error (degrees)
    confidence: float                # 0.0 to 1.0, confidence in gate detection
    debug_info: Optional[Dict] = None  # Optional debug data
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/debugging."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class DroneCommand:
    """
    Command sent from planner to drone controller.
    """
    command_type: str  # 'move', 'rotate', 'hover', 'land', 'takeoff'
    x_cm: float = 0    # Move distance (cm) - positive = right, negative = left
    y_cm: float = 0    # Move distance (cm) - positive = up, negative = down
    z_cm: float = 0    # Move distance (cm) - positive = forward, negative = backward
    yaw_deg: float = 0 # Rotation (degrees) - positive = clockwise
    velocity: int = 50 # Movement velocity (cm/s)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PlannerState:
    """
    Simple state machine for the planner.
    """
    def __init__(self):
        self.current_state = "SEARCH"
        self.time_in_state = 0
        self.frames_processed = 0
        self.last_vision_output = None
    
    def set_state(self, new_state: str):
        self.current_state = new_state
        self.time_in_state = 0
    
    def update_time(self, delta_sec: float):
        self.time_in_state += delta_sec
    
    def update_vision(self, vision_output: VisionOutput):
        self.last_vision_output = vision_output


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(value, max_val))


def calculate_distance_error(estimated_pos, target_pos) -> float:
    """
    Calculate Euclidean distance between two 3D positions.
    Positions are tuples/lists: (x, y, z)
    """
    if not estimated_pos or not target_pos:
        return float('inf')
    
    dx = estimated_pos[0] - target_pos[0]
    dy = estimated_pos[1] - target_pos[1]
    dz = estimated_pos[2] - target_pos[2]
    
    return (dx**2 + dy**2 + dz**2) ** 0.5


def normalize_angle(angle_deg: float) -> float:
    """
    Normalize angle to range [-180, 180] degrees.
    """
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    return angle_deg


def log_message(message: str, level: str = "INFO"):
    """
    Simple logging function.
    """
    levels = {"DEBUG": "🔍", "INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌"}
    symbol = levels.get(level, "•")
    print(f"[{symbol} {level}] {message}")
