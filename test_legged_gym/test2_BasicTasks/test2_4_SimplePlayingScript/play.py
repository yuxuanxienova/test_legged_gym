import os
from test_legged_gym.test2_BasicTasks.test2_4_SimplePlayingScript.task_cfg import TaskConfig
from test_legged_gym.test2_BasicTasks.test2_4_SimplePlayingScript.sim_cfg import SimConfig
from test_legged_gym.test2_BasicTasks.test2_4_SimplePlayingScript.train_cfg import TrainConfig
from test_legged_gym.test2_BasicTasks.utils import class_to_dict
from test_legged_gym.test2_BasicTasks.test2_4_SimplePlayingScript.Env import Env
from test_legged_gym.rsl_rl.runners.on_policy_runner import OnPolicyRunner
from test_legged_gym.rsl_rl.modules.actor_critic import ActorCritic
import torch
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import threading

# Ensure that matplotlib uses a non-blocking backend
matplotlib.use('TkAgg')  # You can choose a different backend if preferred

# Shared variable to store the target position
target_pos = None
target_lock = threading.Lock()

def set_target_position(event):
    global target_pos
    if event.inaxes:
        x, y = event.xdata, event.ydata
        with target_lock:
            target_pos = np.array([x, y])
        print(f"Target position set to: ({x:.2f}, {y:.2f})")
        # Optionally, update the plot with the target
        ax.plot(x, y, 'ro')  # Red dot for target
        fig.canvas.draw()

def start_gui():
    global fig, ax
    fig, ax = plt.subplots()
    ax.set_title("Click to set target position")
    ax.set_xlim(-10, 10)  # Adjust according to your environment's coordinate system
    ax.set_ylim(-10, 10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.canvas.mpl_connect('button_press_event', set_target_position)
    plt.show()

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

    # Start the GUI in a separate thread
    gui_thread = threading.Thread(target=start_gui, daemon=True)
    gui_thread.start()

    while True:
        with torch.no_grad():
            actions = model.act(obs)
        obs, privileged_obs, rewards, dones, infos = env.step(actions)

        with target_lock:
            if target_pos is not None:
                env.apply_command_position(env_ids=0, target_pos=target_pos)
                print(f"Moving towards target position: {target_pos}")
                # Reset target_position after setting the target
                target_pos = None