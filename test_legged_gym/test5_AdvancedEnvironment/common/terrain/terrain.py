import os
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix
from isaacgym import gymapi

from test_legged_gym.utils.warp_utils import convert_to_wp_mesh
import torch
class Terrain:
    def __init__(self,gym,sim,device,num_envs,env_spacing=5.0):



        # self.terrain.sample_new_init_poses(env_ids)

        # 1.3 Create Terrain Mesh (replace ground plane)

        self.gym = gym
        self.sim = sim
        self.device = device
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        # Load your terrain mesh here
        asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),"assets")
        terrain_file = "/terrain/simple_terrain.obj"
        if(os.path.exists(asset_root + terrain_file)):
            print("[INFO]Terrain file found")
        else:
            print("[ERROR]Terrain file not found")
            exit(1)

        # Load the terrain mesh from a .obj file
        self.terrain_mesh = trimesh.load(asset_root + terrain_file)  # Replace 'terrain.obj' with your mesh file path

        # Calculate the center of the mesh
        mesh_center = self.terrain_mesh.centroid

        # Optionally, translate the mesh so that its center is at the origin
        self.terrain_mesh.apply_translation(-mesh_center)

        # Set the heading angle in degrees
        heading_angle_degrees = 90  # Replace with your desired angle in degrees
        heading_angle_radians = np.deg2rad(heading_angle_degrees)

        # Set the rotation axis (e.g., Z-axis)
        rotation_axis = [1.0, 0.0, 0.0]  # Rotate around Z-axis

        # Create the rotation matrix
        R = rotation_matrix(heading_angle_radians, rotation_axis)

        # Apply the rotation to the mesh
        self.terrain_mesh.apply_transform(R)

        # Extract vertices and triangle indices from the mesh
        self.vertices = np.array(self.terrain_mesh.vertices, dtype=np.float32)
        self.triangles = np.array(self.terrain_mesh.faces, dtype=np.uint32)

        # Convert to wp mesh
        self.wp_terrain_mesh = convert_to_wp_mesh(self.terrain_mesh.vertices, self.terrain_mesh.faces, self.device)

        # Create triangle mesh parameters
        self.tm_params = gymapi.TriangleMeshParams()
        self.tm_params.nb_vertices = self.vertices.shape[0]
        self.tm_params.nb_triangles = self.triangles.shape[0]

        # Initialize the transform without any rotation
        self.tm_params.transform = gymapi.Transform()
        self.tm_params.transform.p = gymapi.Vec3(0.0, 0.0, 0.0)  # Adjust as needed

        # Set friction and restitution
        self.tm_params.static_friction = 0.5
        self.tm_params.dynamic_friction = 0.5
        self.tm_params.restitution = 0.0

        # Get the environment origins
        self._calcu_env_origins()

    def add_to_sim(self):
        # Add the terrain mesh to the simulation
        self.gym.add_triangle_mesh(self.sim, self.vertices.flatten(), self.triangles.flatten(), self.tm_params)

    def _calcu_env_origins(self):
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)#Dim:(num_envs, 3)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.0
        
    def sample_new_init_poses(self,env_ids):
        # Sample new initial poses for the environments
        return self.env_origins[env_ids] 

