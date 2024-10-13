import numpy as np
import os
from datetime import datetime
import isaacgym
import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
import yaml
from Env import RobotEnv

if __name__ == "__main__":

    # Load the configuration files


    # env = RobotEnv(task_cfg_dict, sim_cfg_dict)
    # runner = OnPolicyRunner(env, train_cfg_dict, log_dir=None, device="cuda:0")
    # runner.learn(num_learning_iterations=train_cfg_dict["runner"]["max_iterations"], init_at_random_ep_len=True)