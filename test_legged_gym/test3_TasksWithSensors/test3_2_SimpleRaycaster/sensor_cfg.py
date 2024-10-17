from typing import Tuple, Callable, Optional, Any, List
from test_legged_gym.test3_TasksWithSensors.test3_2_SimpleRaycaster.sensor_utils import my_pattern_func
class SensorCfgBase:
    enable_debug_vis: bool = False

class GridPatternCfg(SensorCfgBase):
    resolution: float = 0.1
    width: float = 1.0
    length: float = 1.6
    max_xy_drift: float = 0.05
    direction: Tuple = (0.0, 0.0, -1.0)
    pattern_func: Callable = my_pattern_func    

class RaycasterCfg(SensorCfgBase):
    class_name: str = "Raycaster"
    terrain_mesh_names: Tuple[str, ...] = ("terrain",)
    robot_name: str = "robot"
    body_attachement_name: str = "base"
    attachement_pos: Tuple = (0.0, 0.0, 0.0)
    attachement_quat: Tuple = (0.0, 0.0, 0.0, 1.0)
    attach_yaw_only: bool = True  # do not use the roll and pitch of the robot to update the rays
    default_hit_value: float = -10.0  # which value to return when a ray misses the hit
    default_hit_distance: float = 10.0  # which distance to return when a ray misses the hit
    pattern_cfg: Any = GridPatternCfg()
    post_process_func: Optional[Callable] = None  # function to apply to the raycasted values
    visualize_friction: bool = False