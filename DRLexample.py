# Code snippets adapted from : https://huggingface.co/learn/deep-rl-course/en/unit1/hands-on


import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

#CREATE ENVIRONMENT
env = make_vec_env('LunarLander-v2', n_envs=1)

#CREATE MODEL
model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1)

# TRAIN MODEL
model.learn(total_timesteps=1000)

# SAVE MODEL
model.save("ppo-LunarLander-v2")

# LOAD MODEL
model = PPO.load("ppo-LunarLander-v2")

# EVALUATE MODEL
eval_env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")