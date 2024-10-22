from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from test_legged_gym.test5_AdvancedEnvironment.envs.legged_env import LeggedEnv

    ANY_ENV = Union[LeggedEnv]

import torch
from test_legged_gym.utils.math_utils import wrap_to_pi
"""
Common reward Functions
"""


def torques(env: "ANY_ENV", params):
    # Penalize torques
    return torch.sum(torch.square(env.robot.dof_torques), dim=1)


def torques_selected(env: "ANY_ENV", params):
    # Penalize torques on selected dofs
    return torch.sum(torch.square(env.robot.dof_torques[:, params["dof_indices"]]), dim=1)


def motor_torques(env: "ANY_ENV", params):
    # Penalize motor torques
    motor_torques = env.robot.dof_torques / env.robot.gear_ratio
    return torch.sum(torch.square(motor_torques), dim=1)


def motor_torques_selected(env: "ANY_ENV", params):
    # Penalize motor torques on selected dofs
    motor_torques = env.robot.dof_torques / env.robot.gear_ratio
    return torch.sum(torch.square(motor_torques[:, params["dof_indices"]]), dim=1)


def dof_vel(env: "ANY_ENV", params):
    # Penalize dof velocities
    return torch.sum(torch.square(env.robot.dof_vel), dim=1)


def dof_acc(env: "ANY_ENV", params):
    # Penalize dof accelerations
    return torch.sum(torch.square(env.robot.dof_acc), dim=1)


def action_rate(env: "ANY_ENV", params):
    # Penalize changes in actions
    return torch.sum(torch.square(env.last_actions - env.actions), dim=1)


def collision(env: "ANY_ENV", params):
    # Penalize collisions on selected bodies
    collision_force = torch.norm(env.robot.net_contact_forces[:, params["body_indices"], :], dim=-1)
    collision_force = torch.where(collision_force > 1., collision_force.clip(min=200.), collision_force) # minimum penalty of 100
    return torch.sum(collision_force, dim=1) / 200.


def termination(env: "ANY_ENV", params):
    # Terminal reward / penalty
    return env.reset_buf * (1-env.termination_manager.time_out_buf)


def dof_pos_limits(env: "ANY_ENV", params):
    # Penalize dof positions too close to the limit. The soft limit is computed by the env
    out_of_limits = -(env.robot.dof_pos - env.robot.soft_dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
    out_of_limits += (env.robot.dof_pos - env.robot.soft_dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def dof_vel_limits(env: "ANY_ENV", params):
    # Penalize dof velocities too close to the limit
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    # "ratio" defines the soft limit as a percentage of the hard limit
    return torch.sum(
        (torch.abs(env.robot.dof_vel) - env.robot.soft_dof_vel_limits * params["soft_ratio"]).clip(min=0.0, max=1.0),
        dim=1,
    )


def torque_limits(env: "ANY_ENV", params):
    # penalize torques too close to the limit
    # "ratio" defines the soft limit as a percentage of the hard limit
    return torch.sum(
        (torch.abs(env.robot.des_dof_torques) - env.robot.soft_dof_torque_limits * params["soft_ratio"]).clip(min=0.0),
        dim=1,
    )


def contact_forces(env: "ANY_ENV", params):
    # penalize high contact forces
    return torch.sum(
        (torch.norm(env.robot.net_contact_forces, dim=-1) - params["max_contact_force"]).clip(min=0.0),
        dim=1,
    )

def contact_forces_sq(env: "ANY_ENV", params):
    # penalize high contact forces
    contact_forces = torch.norm(env.robot.net_contact_forces, dim=-1).clip(max = 2.*params["max_contact_force"])
    return torch.sum(
        torch.square((contact_forces - params["max_contact_force"]).clip(min=0.0)),
        dim=1,
    )

def mech_power(env: "ANY_ENV", params):
    # penalize high contact forces
    pos_power = (env.robot.dof_torques * env.robot.dof_vel).clip(min=0.0).sum(dim=-1)
    neg_power = (env.robot.dof_torques * env.robot.dof_vel).clip(max=0.0).sum(dim=-1)
    total_power = pos_power + params["recuperation"]*neg_power
    return torch.square(total_power)
    

"""
Locomotion specific reward Functions
"""


def lin_vel_z(env: "LeggedEnv", params):
    # Penalize z axis base linear velocity
    return torch.square(env.robot.root_lin_vel_b[:, 2])


def ang_vel_xy(env: "LeggedEnv", params):
    # Penalize xy axes base angular velocity
    return torch.sum(torch.square(env.robot.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation(env: "LeggedEnv", params):
    # Penalize non flat base orientation
    return torch.sum(torch.square(env.robot.projected_gravity_b[:, :2]), dim=1)


def vertical_orientation(env: "LeggedEnv", params):
    # Penalize non flat base orientation
    return torch.sum(torch.square(env.robot.projected_gravity_b[:, 1:]), dim=1)


def base_height(env: "LeggedEnv", params):
    # Penalize base height away from target
    measured_heights = env.sensors[params["sensor"]].get_data()
    base_height = torch.mean(env.robot.root_pos_w[:, 2].unsqueeze(1) - measured_heights, dim=1)
    return torch.square(base_height - params["height_target"])


def tracking_lin_vel(env: "LeggedEnv", params):
    # Tracking of linear velocity commands (xy axes)
    # "std" defines the width of the bel curve
    lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.robot.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / params["std"])


def tracking_ang_vel(env: "LeggedEnv", params):
    # Tracking of angular velocity commands (yaw)
    # "std" defines the width of the bel curve
    ang_vel_error = torch.square(env.commands[:, 2] - env.robot.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / params["std"])


def feet_air_time(env: "LeggedEnv", params):
    # Reward long steps
    first_contact = env.robot.feet_last_air_time > 0.0
    reward = torch.sum((env.robot.feet_last_air_time - params["time_threshold"]) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.commands[:, :2], dim=1) > 0.1
    return reward


def stumble(env: "LeggedEnv", params):
    # Penalize feet hitting vertical surfaces
    return torch.any(
        torch.norm(env.robot.net_contact_forces[:, :, :2], dim=2)
        > params["hv_ratio"] * torch.abs(env.robot.net_contact_forces[:, :, 2]),
        dim=1,
    )


def stand_still(env: "LeggedEnv", params):
    # Penalize motion at zero commands
    return torch.sum(torch.abs(env.robot.dof_pos - env.robot.default_dof_pos), dim=1) * (
        torch.norm(env.commands[:, :2], dim=1) < 0.1
    )


"""
Position tracking rewards
"""
def _command_duration_mask(env: "LeggedEnvPos", duration):
    mask = env.command_time_left <= duration
    return mask / duration


def tracking_pos(env: "LeggedEnvPos", params):
    distance = torch.norm(env.pos_target[:, :2] - env.robot.root_pos_w[:, :2], dim=1)
    return (1. /(1. + torch.square(distance))) * _command_duration_mask(env, params["duration"])


def tracking_pos2(env: "LeggedEnvPos", params):
    distance = torch.norm(env.pos_target[:, :2] - env.robot.root_pos_w[:, :2], dim=1)
    return (1 - 0.5*distance)* _command_duration_mask(env, params["duration"])

def tracking_pos3d(env: "LeggedEnvPos", params):
    distance = torch.norm(env.pos_target[:, :3] - env.robot.root_pos_w[:, :3], dim=1)
    return (1 - 0.5*distance)* _command_duration_mask(env, params["duration"])


def tracking_heading(env: "LeggedEnvPos", params):
    distance = wrap_to_pi(env.heading_target - env.robot.heading_w)
    position_distance = torch.norm(env.pos_target[:, :2] - env.robot.root_pos_w[:, :2], dim=1)
    return (1. /(1. + torch.square(distance))) * (position_distance < params["max_pos_distance"]) * _command_duration_mask(env, params["duration"])

def tracking_heading2(env: "LeggedEnvPos", params):
    distance = torch.abs(wrap_to_pi(env.heading_target - env.robot.heading_w))
    position_distance = torch.norm(env.pos_target[:, :2] - env.robot.root_pos_w[:, :2], dim=1)
    return (1 - 0.5*distance)* (position_distance < params["max_pos_distance"]) * _command_duration_mask(env, params["duration"])


def dont_wait(env: "LeggedEnvPos", params):
    #TODO add params
    far_away = torch.norm(env.pos_target[:, :2] - env.robot.root_pos_w[:, :2], dim=1) > 0.25
    waiting = torch.norm(env.robot.root_lin_vel_w[:, :2], dim=1) < params["min_vel"]
    return far_away * waiting


def dont_wait_b(env: "LeggedEnvPos", params):
    far_away = torch.norm(env.pos_target[:, :2] - env.robot.root_pos_w[:, :2], dim=1) > 0.25
    waiting = torch.norm(env.robot.root_lin_vel_b[:, :2], dim=1) < params["min_vel"]
    return far_away * waiting


def move_in_direction(env: "LeggedEnvPos", params):
    vel_target = env.pos_commands[:, :2] / (torch.norm(env.pos_commands[:, :2], dim=1).unsqueeze(1) + 0.1)
    vel = env.robot.root_lin_vel_b[:, :2] / (torch.norm(env.robot.root_lin_vel_b[:, :2], dim=1).unsqueeze(1) + 0.1)
    return vel[:, 0] * vel_target[:, 0] + vel[:, 1] * vel_target[:, 1] #- self.base_ang_vel[:, 2].abs()


def stand_still_pose(env: "LeggedEnvPos", params):
    should_stand = torch.norm(env.pos_target[:, :2] - env.robot.root_pos_w[:, :2], dim=1) < 0.25
    should_stand &= torch.abs(env.heading_target - env.robot.heading_w) < 0.5
    return torch.sum(torch.square(env.robot.dof_pos - env.robot.default_dof_pos), dim=1) * should_stand * _command_duration_mask(env, params["duration"])


def base_acc(env: "LeggedEnv", params):
    return torch.sum(torch.square((env.robot.last_root_states[:, 7:10] - env.robot.root_states[:, 7:10]) / env.dt), dim=1) + \
        0.02 * torch.sum(torch.square((env.robot.last_root_states[:, 10:13] - env.robot.root_states[:, 10:13]) / env.dt), dim=1)


def feet_acc(env: "LeggedEnv", params):
    return torch.sum(torch.norm(env.robot.body_acc[:, env.robot.feet_indices], dim=-1), dim=1)