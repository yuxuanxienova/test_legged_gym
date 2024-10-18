
from isaacgym.torch_utils import quat_apply
from test_legged_gym.test3_TasksWithSensors.test3_2_SimpleRaycaster.sensor_cfg import RaycasterCfg
from test_legged_gym.utils.math_utils import quat_apply_yaw
from test_legged_gym.utils.warp_utils import ray_cast
from test_legged_gym.utils.visualization_utils import BatchWireframeSphereGeometry 
import torch
import numpy as np
import matplotlib.pyplot as plt 
from typing import TYPE_CHECKING, Optional, Sequence
class SensorBase:
    def __init__(self, cfg, env):
        # prepare some buffers
        # enable corresponding sensors in sim
        raise NotImplementedError()

    def update(self, dt: float, env_ids=None):
        # compute stuff
        raise NotImplementedError()

    def get_data(self):
        # returns sensor data
        raise NotImplementedError()

    def reset(self):
        pass
class Raycaster(SensorBase):
    def __init__(self, cfg: RaycasterCfg, env ):
        #1. Configuration and Environment Setup
        self.cfg = cfg
        self.env=env
        # self.terrain_mesh = env.terrain.wp_meshes[self.cfg.terrain_mesh_name]
        # self.terrain_mesh = env.terrain.get_wp_mesh_from_names(self.cfg.terrain_mesh_names)
        self.terrain_mesh = env.wp_terrain_mesh


        self.num_envs = self.env.num_envs
        self.device = self.env.device

        #2. Raycasting Setup
        self.ray_starts, self.pattern_ray_directions = cfg.pattern_cfg.pattern_func(self.device)#Dim:[num_rays,3],Dim:[num_rays,3]
        self.num_rays = len(self.pattern_ray_directions)#num_rays

        #3. Sensor Attachment Transformation
        offset_pos = torch.tensor(list(cfg.attachement_pos), device=env.device)
        offset_quat = torch.tensor(list(cfg.attachement_quat), device=env.device)
        self.ray_directions = quat_apply(
            offset_quat.repeat(len(self.pattern_ray_directions), 1), self.pattern_ray_directions
        )
        self.ray_starts += offset_pos

        #4. Environment Replication
        self.ray_starts = self.ray_starts.repeat(self.num_envs, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self.num_envs, 1, 1)

        #5. Buffers Initialization
        self.ray_hits_world = torch.zeros(self.num_envs, self.num_rays, 3, device=self.device)
        self.ray_distances = torch.zeros(self.num_envs, self.num_rays, 1, device=self.device)
        self.sphere_geom = None
        self.sphere_geoms = {}

    def update(self, dt, sensor_states,env_ids=...):
        """Perform raycasting on the terrain.

        Args:
            sensor_states (torch.Tensor): Dimensions: [num_envs, 13]. The robot states in quaternion format.
            env_ids (List[int], optional): Subset of environments for which to return the ray hits. Defaults to ....
        """
        #1. Robot State Retrieval
        pos = sensor_states[env_ids, :3] + torch.tensor([0,0,0.4]).repeat(len(env_ids),1).to(self.device)#Dim:[num_envs,3]
        quats = sensor_states[env_ids, 3:7]#Dim:[num_envs,4]

        #2. Transforming Rays to World Frame
        if self.cfg.attach_yaw_only:
            ray_starts_world = quat_apply_yaw(quats.repeat(1, self.num_rays), self.ray_starts[env_ids]) + pos.unsqueeze(1)#Dim:[num_envs, num_rays, 3]
            ray_directions_world = self.ray_directions[env_ids]#Dim:[num_envs, num_rays, 3]
        else:
            ray_starts_world = quat_apply(quats.repeat(1, self.num_rays), self.ray_starts[env_ids]) + pos.unsqueeze(1)#Dim:[num_envs, num_rays, 3]
            ray_directions_world = quat_apply(quats.repeat(1, self.num_rays), self.ray_directions[env_ids])#Dim:[num_envs, num_rays, 3]

        #2.(Optional)----Debug Raycasting----
        # self.num_rays=3
        # self.ray_hits_world = torch.zeros(self.num_envs, self.num_rays, 3, device=self.device)
        # self.ray_distances = torch.zeros(self.num_envs, self.num_rays, 1, device=self.device)
        # ray_starts_debug = torch.tensor([[0,0,0] , [0,0,0] , [0,0,0]]).repeat(len(env_ids),1).to(self.device).float()#Dim:[num_envs,3]
        # ray_starts_world = quat_apply_yaw(quats.repeat(1, 1), ray_starts_debug ) + pos.unsqueeze(1)

        # ray_directions_debug = torch.tensor([[0,0,-1], [0,1,0] , [1,0,0]]).repeat(len(env_ids),1).to(self.device).float()#Dim:[num_envs,3]
        # ray_directions_world = quat_apply_yaw(quats.repeat(1, 1), ray_directions_debug) 
        #-------------------------------

        #3. Raycasting
        #ray_starts_world: torch.Size([num_envs, num_rays, 3])
        #ray_directions_world: torch.Size([num_envs, num_rays, 3])
        self.ray_hits_world[env_ids], self.ray_distances[env_ids] = ray_cast(
            ray_starts_world, ray_directions_world, self.terrain_mesh
        )

    def get_data(self) -> torch.Tensor:
        """Returns the ray hit positions, ensuring that any NaN values 
        (from rays that didn't hit anything) are replaced with a default value."""
        return torch.nan_to_num(self.ray_hits_world, posinf=self.cfg.default_hit_value)

    def get_distances(self) -> torch.Tensor:
        """Returns the distances to the ray hit points, ensuring that any NaN values 
        (from rays that didn't hit anything) are replaced with a default value."""
        return torch.nan_to_num(self.ray_distances, posinf=self.cfg.default_hit_distance)

    def debug_vis(self, env, env_ids: Optional[Sequence[int]] = None):
        """Visualizes the ray hits in the simulation environment"""

        #1. Environment Selection
        if env_ids is None:
            num_envs = self.num_envs
            env_ids = ...
        else:
            num_envs = len(env_ids)

        #2. Initializing Visualization Geometry
        if self.sphere_geom is None or self.sphere_geom.num_spheres != num_envs * self.num_rays:
            self.sphere_geom = BatchWireframeSphereGeometry(num_envs * self.num_rays, 0.02, 4, 4, None, color=(0, 1, 0))
        
        #3. Drawing Ray Hits
        env.gym.clear_lines(env.viewer)  
        self.sphere_geom.draw(self.ray_hits_world[env_ids], env.gym, env.viewer, env.envs[0])

    def post_process(self, data: torch.Tensor, env):
        return data