
from test_legged_gym.test2_BasicTasks.test2_1_Config.task_cfg import TaskConfig
from test_legged_gym.test2_BasicTasks.test2_1_Config.sim_cfg import SimConfig
from test_legged_gym.test2_BasicTasks.test2_1_Config.train_cfg import TrainConfig
from test_legged_gym.test2_BasicTasks.utils import class_to_dict
from test_legged_gym.test2_BasicTasks.test2_2_SImpleBaseEnv.BaseEnv import BaseEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


if __name__ == "__main__":
    task_cfg = TaskConfig
    sim_cfg = SimConfig
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)



    env = BaseEnv(task_cfg, sim_cfg)
    runner = OnPolicyRunner(env,train_cfg_dict , log_dir=None, device="cuda:0")
    runner.learn(num_learning_iterations=train_cfg_dict["runner"]["max_iterations"], init_at_random_ep_len=True)