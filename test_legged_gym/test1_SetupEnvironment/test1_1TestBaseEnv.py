
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import numpy as np
import torch
from test_legged_gym.test1_SetupEnvironment.legged_robot_config import LeggedRobotCfg



if __name__ == "__main__":
    #-----------1. Initialize Simulator----------------
    cfg = LeggedRobotCfg
    #1.1 reate GymAPI Instance
    gym = gymapi.acquire_gym()

    #1.2 Create the SImulator
    #get default set of sim params
    sim_params = gymapi.SimParams()
    #set common parameters
    sim_params.dt = 1/60
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.use_gpu_pipeline = False #use GPU pipeline, False for CPU pipeline

    sim = gym.create_sim(compute_device=0, graphics_device=0, type=gymapi.SIM_PHYSX, params=sim_params)

    #1.3 Create Ground Plane
    #configure the ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0,0,1)#z-up axis
    plane_params.distance = 0 #distance from the origin
    #create the graound plane
    gym.add_ground(sim, plane_params)

    #2.--------------------Construct a Environment----------------

    #2.1 Create the Environment Space
    env_lower = gymapi.Vec3(0.0,0.0,0.0)
    env_upper = gymapi.Vec3(0.0,0.0,0.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)
    #Load an Asset
    asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"assets")
    # asset_file = "anymal_c/urdf/anymal_c.urdf"
    asset_file = "anymal_c/urdf/anymal_c.urdf"
    #load assets with default control type of position for all the joints
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = cfg.asset.default_dof_drive_mode
    asset_options.collapse_fixed_joints = cfg.asset.collapse_fixed_joints
    asset_options.replace_cylinder_with_capsule = cfg.asset.replace_cylinder_with_capsule
    asset_options.flip_visual_attachments = cfg.asset.flip_visual_attachments
    asset_options.fix_base_link = cfg.asset.fix_base_link
    asset_options.density = cfg.asset.density
    asset_options.angular_damping = cfg.asset.angular_damping
    asset_options.linear_damping = cfg.asset.linear_damping
    asset_options.max_angular_velocity = cfg.asset.max_angular_velocity
    asset_options.max_linear_velocity = cfg.asset.max_linear_velocity
    asset_options.armature = cfg.asset.armature
    asset_options.thickness = cfg.asset.thickness
    asset_options.disable_gravity = cfg.asset.disable_gravity
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    if asset is None:
        print("Failed to load asset at:", os.path.join(asset_root, asset_file))
        sys.exit(1)
    else:
        print("Successfully loaded asset:", os.path.join(asset_root, asset_file))

    #2.2 Create the Actor
    
    start_pose = gymapi.Transform()
    start_pose.p = gymapi.Vec3(*torch.tensor(cfg.init_state.pos,device="cuda:0", requires_grad=False))#initial position
    actor_handle = gym.create_actor(env, asset, start_pose, cfg.asset.name, 0,cfg.asset.self_collisions, 0)
    #3.----------------------Simulation and Rendering Loop------------------------

    # Prepare the simulator after all assets and environments have been added
    gym.prepare_sim(sim)

    #set the viewer
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)

    while True:
        #step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)#wait for the results on in cpu

        #step the rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        #step the graphics
        gym.sync_frame_time(sim)





