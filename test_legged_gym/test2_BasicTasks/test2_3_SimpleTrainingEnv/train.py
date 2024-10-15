import os
from test_legged_gym.test2_BasicTasks.test2_1_Config.task_cfg import TaskConfig
from test_legged_gym.test2_BasicTasks.test2_1_Config.sim_cfg import SimConfig
from test_legged_gym.test2_BasicTasks.test2_1_Config.train_cfg import TrainConfig
from test_legged_gym.test2_BasicTasks.utils import class_to_dict
from test_legged_gym.test2_BasicTasks.test2_3_SimpleTrainingEnv.TrainEnv import TrainEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    task_cfg = TaskConfig
    sim_cfg = SimConfig
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)


    env = TrainEnv(task_cfg, sim_cfg)
    runner = OnPolicyRunner(env,train_cfg_dict , log_dir=log_dir, device="cuda:0")
    runner.learn(num_learning_iterations=train_cfg_dict["runner"]["max_iterations"], init_at_random_ep_len=True)