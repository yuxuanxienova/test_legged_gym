import os
from test_legged_gym.test2_BasicTasks.utils import class_to_dict
from isaacgym import gymapi
from isaacgym import gymutil, gymtorch
import numpy as np
import torch
import os
from isaacgym.torch_utils import quat_rotate_inverse, to_torch, get_axis_params, torch_rand_float
class BaseEnv:
    #-----------0. Initialize the Environment----------------
    def __init__(self, task_cfg_class, sim_cfg_class):
        #-----------1. Initialize GymAPI ,Simulator----------------
        self.task_cfg_class = task_cfg_class
        self.sim_cfg_class = sim_cfg_class

        #1.0 Parse Configs

        #parse sim params
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 1/60
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.use_gpu_pipeline = sim_cfg_class.use_gpu_pipeline #use GPU pipeline, False for CPU pipeline
        self.device = sim_cfg_class.device

        #parse task config: env
        self.num_envs = task_cfg_class.env.num_envs
        self.num_obs = task_cfg_class.env.num_observations
        self.num_privileged_obs = task_cfg_class.env.num_privileged_obs
        self.num_actions = task_cfg_class.env.num_actions
        self.env_spacing = task_cfg_class.env.env_spacing
        self.episode_length_s = task_cfg_class.env.episode_length_s
        #parse task config: init_state
        self.initial_state_pos = task_cfg_class.init_state.pos
        self.init_state_default_joint_angles = task_cfg_class.init_state.default_joint_angles
        #parse task config: control
        self.control_type = task_cfg_class.control.control_type
        self.control_damping = task_cfg_class.control.damping
        self.control_stiffness = task_cfg_class.control.stiffness
        #parse task config: noise
        self.add_noise  = task_cfg_class.noise.add_noise
        self.noise_scales  = task_cfg_class.noise.noise_scales
        self.noise_level = task_cfg_class.noise.noise_level
        #parse task config: commands
        self.num_commands = task_cfg_class.commands.num_commands
        self.commands_ranges = task_cfg_class.commands.ranges
        #parse task config: domain_rand
        self.domain_rand_randomize_friction = task_cfg_class.domain_rand.randomize_friction
        self.domain_rand_friction_range =   task_cfg_class.domain_rand.friction_range
        self.domain_rand_randomize_base_mass = task_cfg_class.domain_rand.randomize_base_mass
        self.domain_rand_added_mass_range =    task_cfg_class.domain_rand.added_mass_range
        #parse task config: rewards
        self.rewards_soft_dof_pos_limit = task_cfg_class.rewards.soft_dof_pos_limit
        self.reward_scales = class_to_dict(task_cfg_class.rewards.scales)
        #parse task config: normalization
        self.obs_scales = task_cfg_class.normalization.obs_scales
        #parse task config: asset
        asset_options = gymapi.AssetOptions()
        self.asset_name = task_cfg_class.asset.name
        self.asset_self_collisions= task_cfg_class.asset.self_collisions
        asset_options.default_dof_drive_mode = task_cfg_class.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = task_cfg_class.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = task_cfg_class.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = task_cfg_class.asset.flip_visual_attachments
        asset_options.fix_base_link = task_cfg_class.asset.fix_base_link
        asset_options.density = task_cfg_class.asset.density
        asset_options.angular_damping = task_cfg_class.asset.angular_damping
        asset_options.linear_damping = task_cfg_class.asset.linear_damping
        asset_options.max_angular_velocity = task_cfg_class.asset.max_angular_velocity
        asset_options.max_linear_velocity = task_cfg_class.asset.max_linear_velocity
        asset_options.armature = task_cfg_class.asset.armature
        asset_options.thickness = task_cfg_class.asset.thickness
        asset_options.disable_gravity = task_cfg_class.asset.disable_gravity

        #compute other quantities
        self.dt  = self.task_cfg_class.control.decimation * self.sim_params.dt
        self.max_episode_length_s = self.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt).astype(int)
        self.task_cfg_class.domain_rand.push_interval = np.ceil(self.task_cfg_class.domain_rand.push_interval_s / self.dt)



        #1.1 reate GymAPI Instance
        self.gym = gymapi.acquire_gym()

        #1.2 Create the SImulator
        self.sim = self.gym.create_sim(compute_device=0, graphics_device=0, type=gymapi.SIM_PHYSX, params=self.sim_params)

        #1.3 Create Ground Plane
        #configure the ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0,0,1)#z-up axis
        plane_params.distance = 0 #distance from the origin
        #create the graound plane
        self.gym.add_ground(self.sim, plane_params)

        #1.4 Allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None

        #2.--------------------Construct a Environment----------------
        #2.0 Load an Asset
        asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"assets")
        asset_file = "anymal_c/urdf/anymal_c.urdf"        

        #load the asset
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        if robot_asset is None:
            print("Failed to load asset at:", os.path.join(asset_root, asset_file))
            sys.exit(1)
        else:
            print("Successfully loaded asset:", os.path.join(asset_root, asset_file))

        #get the asset properties
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        feet_names = [s for s in body_names if self.task_cfg_class.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.task_cfg_class.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.task_cfg_class.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])        


        #2.1 Create the Environment Space
        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0,0.0,0.0)
        env_upper = gymapi.Vec3(0.0,0.0,0.0)
        # self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1)
        #2.2 Create the Actor
        
        start_pose = gymapi.Transform()
        # start_pose.p = gymapi.Vec3(*torch.tensor(task_cfg_dict["init_state"]["pos"],device="cuda:0", requires_grad=False))#initial position
        # actor_handle = self.gym.create_actor(self.env, robot_asset, start_pose, task_cfg_dict["asset"]["name"], 0,task_cfg_dict["asset"]["self_collisions"], 0)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos_env_origin = self.env_origins[i].clone()
            pos_env_origin[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(pos_env_origin[0]+self.initial_state_pos[0] ,pos_env_origin[1]+self.initial_state_pos[1],pos_env_origin[2]+self.initial_state_pos[2])
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.asset_name, i, self.asset_self_collisions, 0)

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        #3.----------------------Simulation and Rendering Loop------------------------

        # Prepare the simulator after all assets and environments have been added
        self.gym.prepare_sim(self.sim)

        #set the viewer
        cam_props = gymapi.CameraProperties()
        self.viewer = self.gym.create_viewer(self.sim, cam_props)
        
        self._init_buffers()
        # self._prepare_reward_function()
        self.init_done = True
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.init_state_default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.control_stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.control_stiffness[dof_name]
                    self.d_gains[i] = self.control_damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
    def _get_env_origins(self):
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
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.domain_rand_randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.domain_rand_friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props
    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        noise_scales = self.noise_scales
        noise_level = self.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        # if self.task_cfg_class.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.rewards_soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.rewards_soft_dof_pos_limit
        return props
    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.domain_rand_randomize_base_mass:
            rng = self.domain_rand_added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    #-----------1. Step the Environment----------------
    def step(self,actions):
        """
        actions (torch.Tensor):  Dim:(num_envs, num_actions_per_env)
        """
        action_bound = self.task_cfg_class.normalization.clip_actions
        self.actions = torch.clip(actions, -action_bound, action_bound)
        
        for _ in range(self.task_cfg_class.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if(self.device != "cpu"):
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.render()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.task_cfg_class.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions ,Dim:(num_envs, num_actions_per_env)

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.task_cfg_class.control.action_scale#Dim:(num_envs, num_actions_per_env)
        control_type = self.task_cfg_class.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    def post_physics_step(self):
        pass
    def render(self):
        #step the rendering
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        #step the graphics
        self.gym.sync_frame_time(self.sim)
    #-----------2. Reset the Environment----------------
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs
    def reset_idx(self, env_ids):
        """Reset selected robots"""
        pass
    #-----------3. get methods----------------
    def get_observations(self):
        return self.obs_buf
    def get_privileged_observations(self):
        return self.privileged_obs_buf
