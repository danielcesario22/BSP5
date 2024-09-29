
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


class DRLmodel:
     class RLModel:
     def __init__(self, policy , env , n_steps , batch_size , \
          n_epochs, gamma , gae_lambda , ent_coef , verbose):
          self.policy=policy
          self.env=env
          self.n_steps=n_steps
          self.batch_size=batch_size
          self.n_epochs=n_epochs
          self.gamma=gamma
          self.gae_lambda=gae_lambda
          self.ent_coef=ent_coef
          self.verbose=verbose
          self.env_name= env.envs[0].spec.id
          self.model = PPO(
               policy =policy,
               env =env,
               n_steps= n_steps,
               batch_size=batch_size,
               n_epochs=n_epochs,
               gamma=gamma,
               gae_lambda=gae_lambda,
               ent_coef=ent_coef,
               verbose=verbose)

     def train(self,total_timesteps):
          self.model.learn(total_timesteps=total_timesteps)

     def save(self):
          self.model.save(f"{self.env_name}_model")

     def load(self,model_name):
          self.model.load(model_name)

     def evaluate(self,n_eval_episodes,deterministic):
          eval_env = Monitor(gym.make(self.env_name, render_mode='rgb_array'))
          mean_reward, std_reward = evaluate_policy(self.model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=deterministic)
          print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

     

