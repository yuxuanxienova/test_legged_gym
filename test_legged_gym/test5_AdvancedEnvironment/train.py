from test_legged_gym.test5_AdvancedEnvironment.envs.legged_env import LeggedEnv
from test_legged_gym.test5_AdvancedEnvironment.envs.legged_env_config import LeggedEnvCfg
import torch

if __name__ == "__main__":
    env = LeggedEnv(LeggedEnvCfg())
    env.reset()
    while True:
        env.step(torch.zeros((env.num_envs, env.num_actions)).to(env.device))
        env.render()

