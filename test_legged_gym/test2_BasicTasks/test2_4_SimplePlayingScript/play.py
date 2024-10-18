import os
from test_legged_gym.test2_BasicTasks.test2_1_Config.task_cfg import TaskConfig
from test_legged_gym.test2_BasicTasks.test2_1_Config.sim_cfg import SimConfig
from test_legged_gym.test2_BasicTasks.test2_1_Config.train_cfg import TrainConfig
from test_legged_gym.test2_BasicTasks.utils import class_to_dict
from test_legged_gym.test2_BasicTasks.test2_4_SimplePlayingScript.Env import Env
from test_legged_gym.rsl_rl.runners.on_policy_runner import OnPolicyRunner
from test_legged_gym.rsl_rl.modules.actor_critic import ActorCritic
import torch


if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    task_cfg = TaskConfig
    sim_cfg = SimConfig
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    env = Env(task_cfg, sim_cfg)

    num_critic_obs = env.num_obs
    policy_cfg = train_cfg_dict["policy"]
    model = ActorCritic( env.num_obs,num_critic_obs,env.num_actions,**policy_cfg).to(device)
    
    # Path to the pretrained model
    pretrained_model_path = os.path.join(log_dir, "model_5000.pt")
    
    # Load the pretrained model
    try:
        loaded_dict = torch.load(pretrained_model_path)
        model.load_state_dict(loaded_dict['model_state_dict'])
    except FileNotFoundError as e:
        print(e)
        print("Proceeding with randomly initialized model.")

    obs, privileged_obs = env.reset()
    while True:
        with torch.no_grad():
            actions = model.act(obs)
        obs, privileged_obs, rewards, dones, infos = env.step(actions)