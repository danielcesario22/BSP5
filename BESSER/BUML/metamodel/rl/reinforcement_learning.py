from __future__ import annotations
from typing import List, Self, Union
from abc import ABC
import gymnasium as gym


class Hyperparameters:
   '''
   Represents a collection of parameters specifically related to the training process of a reinforcement learning agent,
   such as learning rate, batch size, and discount factor.

   Args:
      learning_rate (float): The step size used to update the model 
         parameters during optimization.
      optimizer (str): Optimizer name used for training the agent (e.g., 'Adam').
      num_iterations (int): Total number of training iterations.
      collect_steps_per_iteration (int): Number of steps to collect experience at each training iteration.
      log_interval (int): Frequency (in iterations) of logging training progress.
      eval_interval (int): Frequency (in iterations) of evaluating the agent's performance during training.
      replay_buffer_capacity (int): Maximum size of the replay buffer.
      batch_size (int): Number of samples per training batch.

   Attributes:
      learning_rate (float): The step size used to update the model 
         parameters during optimization.
      optimizer (str): Optimizer name used for training the agent (e.g., 'Adam').
      num_iterations (int): Total number of training iterations.
      collect_steps_per_iteration (int): Number of steps to collect experience at each training iteration.
      log_interval (int): Frequency (in iterations) of logging training progress.
      eval_interval (int): Frequency (in iterations) of evaluating the agent's performance during training.
      replay_buffer_capacity (int): Maximum size of the replay buffer.
      batch_size (int): Number of samples per training batch.
   '''
   def __init__(self, learning_rate: float, optimizer: str, num_iterations: int, 
               collect_steps_per_iteration: int, log_interval: int, eval_interval: int, 
               replay_buffer_capacity: int, batch_size: int):
      self.learning_rate = learning_rate
      self.optimizer = optimizer
      self.num_iterations = num_iterations
      self.collect_steps_per_iteration = collect_steps_per_iteration
      self.log_interval = log_interval
      self.eval_interval = eval_interval
      self.__replay_buffer_capacity = replay_buffer_capacity
      self.__batch_size = batch_size

   @property
   def learning_rate(self) -> float:
      """float: Get the step size used to update the model."""
      return self.__learning_rate
   
   @learning_rate.setter
   def learning_rate(self, learning_rate: float):
      """float: Set the step size used to update the model."""
      self.__learning_rate = learning_rate
   
   @property
   def optimizer(self) -> str:
      """str: Get the optimizer used for training."""
      return self.__optimizer
   
   @optimizer.setter
   def optimizer(self, optimizer: str):
      """
      str: Set the optimizer used for training..
      
      Raises:
         ValueError: If optimizer is not one of the allowed 
         options: 'adam'
      """

      if optimizer not in ['adam']:
          raise ValueError("Invalid value of optimizer")
      self.__optimizer = optimizer
   
   @property
   def num_iterations(self) -> int:
      """int: Get the number of training iterations."""
      return self.__num_iterations
   
   @num_iterations.setter
   def num_iterations(self, num_iterations: int):
      """float: Set the number of training iterations."""
      self.__num_iterations = num_iterations
   
   @property
   def collect_steps_per_iteration(self) -> int:
      """int: Get the number of steps to collect experience."""
      return self.__collect_steps_per_iteration
   
   @collect_steps_per_iteration.setter
   def collect_steps_per_iteration(self, collect_steps_per_iteration: int):
      """float: Set the number of steps to collect experience."""
      self.__collect_steps_per_iteration = collect_steps_per_iteration
   
   @property
   def log_interval(self) -> int:
      """int: Get the number of iteration before logging training progress."""
      return self.__log_interval
   
   @log_interval.setter
   def log_interval(self, log_interval: int):
      """float: Set the number of iteration before logging training progress."""
      self.__log_interval = log_interval
   
   @property
   def eval_interval(self) -> int:
      """int: Get the number of iteration before evaluating the agent."""
      return self.__eval_interval
   
   @eval_interval.setter
   def eval_interval(self, eval_interval: int):
      """float: Set the number of iteration before evaluating the agent."""
      self.__eval_interval = eval_interval
   
   @property
   def replay_buffer_capacity(self) -> int:
      """int: Get the maximum size of the replay buffer."""
      return self.__replay_buffer_capacity

   @replay_buffer_capacity.setter
   def replay_buffer_capacity(self, replay_buffer_capacity: int):
      """int: Set the maximum size of the replay buffer."""
      self.__replay_buffer_capacity = replay_buffer_capacity

   @property
   def batch_size(self) -> int:
      """int: Get the number of samples per training batch."""
      return self.__batch_size

   @batch_size.setter
   def batch_size(self, batch_size: int):
      """int: Set the number of samples per training batch."""
      self.__batch_size = batch_size

   def __repr__(self):
      return (f"Hyperparameters({self.num_iterations}, {self.collect_steps_per_iteration}, "
            f"{self.log_interval}, {self.eval_interval}, "
            f"{self.__replay_buffer_capacity}, {self.__batch_size})")

      

class AgentConfiguration(ABC):
   '''
   Represents parameters that define the structure and setup of the reinforcement learning agent itself,
   including algorithm-specific settings (e.g., epsilon decay for DQN or clip ratio for PPO).
   '''

class DQNConfiguration(AgentConfiguration):
   '''
   Represents parameters that define the structure and setup of DQN agent itself,
   including algorithm-specific settings.

   Args:
         layer_params List[int]: Number of units in a hidden and fully connected layer.
         loss_function (str): The method used to calculate the difference 
            between predicted and actual values.

   Attributes:
         layer_params List[int]: Number of units in a hidden and fully connected layer.
         loss_function (str): The method used to calculate the difference 
            between predicted and actual values.
   '''
   def __init__(self, layer_params: List[int], loss_function: str):
      self.layer_params: List[int] = layer_params
      self.loss_function: str = loss_function

   @property
   def layer_params(self) -> List[int]:
      """List[int]: Get the number of units in a hidden and fully connected layer."""
      return self.__layer_params

   @layer_params.setter
   def layer_params(self, layer_params: List[int]):
      """List[int]: Set the number of units in a hidden and fully connected layer."""
      self.__layer_params = layer_params
   
   @property
   def loss_function(self) -> str:
      """str: Get the method used to calculate the difference between 
            predicted and actual values."""
      return self.__loss_function

   @loss_function.setter
   def loss_function(self, loss_function: str):
      """str: Set the method used to calculate the difference between 
            predicted and actual values.

         Raises:
            ValueError: If loss_function is not one of the allowed 
            options: 'mse'      
      """
      if loss_function not in ['mse']:
            raise ValueError("Invalid value of loss_function")
      self.__loss_function = loss_function
   
   def __repr__(self):
      return (f"DQNConfiguration({self.layer_params}, {self.loss_function})")

class Environment:
   """
   This class represents a simulated environment in which a reinforcement learning agent interacts. 
   
   Args:
         id (str): Id of the gym environment (e.g.: 'CartPole-v1').

   Attributes:
         id (str): Id of the gym environment (e.g.: 'CartPole-v1').
   """

   def __init__(self, id: str):
      self.id: str = id

   @property
   def id(self) -> str:
      """str: Get the id of the environment."""
      return self.__id
   
   @id.setter
   def id(self, id: str):
      """str: Set the id of the environment.

      Raises:
            ValueError: If gym environment id is non-existent.
      """

      env_ids = gym.registry.keys()
      if id not in env_ids:
            raise ValueError("Invalid environment id")
      self.__id = id

   def __repr__(self):
      return (
         f'Environment({self.id})'
      )

class Agent:
   """
   A base class for reinforcement learning agents, providing the structure to initialize
   an agent with a specific algorithm name, training hyperparameters, and agent configuration settings. 
   
   Args:
         name (str): Name of the reinforcement algorithm.
         hyper_param (Hyperparameters): The parameters related to the training of the agent.
         agent_config (AgentConfiguration): The parameters related to the reinforcement learning algorithm.

   Attributes:
         name (str): Name of the reinforcement algorithm.
         hyper_param (Hyperparameters): The parameters related to the training of the agent.
         agent_config (AgentConfiguration): The parameters related to the reinforcement learning algorithm.
   """
   
   def __init__(self, name: str, agent_config: AgentConfiguration, hyper_param: Hyperparameters):
      self.name : str = name
      self.agent_config: AgentConfiguration = agent_config
      self.hyper_param: Hyperparameters = hyper_param
   
   @property
   def name(self) -> str:
      """str: Get the name of the agent algorithm."""
      return self.__name
   
   @name.setter
   def name(self, name: str):
      """str: Set the name of the agent algorithm.
      
         Raises:
            ValueError: If algorithm name is not one of the allowed 
            options: 'dqn'
      """
      if name not in ['dqn']:
            raise ValueError("Invalid agent algorithm name")
      self.__name = name
   
   @property
   def agent_config(self) -> AgentConfiguration:
      """AgentConfiguration: Get the configuration of the agent."""
      return self.__agent_config
   
   @agent_config.setter
   def agent_config(self, agent_config: AgentConfiguration):
      """AgentConfiguration: Set the configuration of the agent."""
      self.__agent_config = agent_config
   
   @property
   def hyper_param(self) -> Hyperparameters:
      """Hyperparameters: Get the training parameters."""
      return self.__hyper_param
   
   @hyper_param.setter
   def hyper_param(self, hyper_param: Hyperparameters):
      """Hyperparameters: Set the training parameters."""
      self.__hyper_param = hyper_param

   def __repr__(self):
      return (
         f'Agent({self.name}, {self.agent_config}, {self.hyper_param})'
      )

class Result:
      '''
      A class to manage evaluation metrics and settings for assessing agents.

      Args:
            metrics List[str]: Quantitative measures used to evaluate 
               the performance of the agent.
            num_eval_episodes (int): Number of episodes used to evaluate the agent.

      Attributes:
            metrics List[str]: Quantitative measures used to evaluate 
               the performance of the agent.
            num_eval_episodes (int): Number of episodes used to evaluate the agent.
      '''

      def __init__(self, metrics: List[str], num_eval_episodes: int):
         self.metrics : List[str] = metrics
         self.num_eval_episodes: int = num_eval_episodes


      @property
      def metrics(self) -> List[str]:
         """
         List[str]: Get the measures for evaluating the performance of 
            the agent.
         """
         return self.__metrics
   
      @metrics.setter
      def metrics(self, metrics: List[str]):
         """
         List[str]: Set the measures for evaluating the performance of 
            the model.
         
         Raises:
            ValueError: If metrics is not one of the allowed 
            options: 'avg_return'
         """
         if isinstance(metrics, list) and \
               all(isinstance(metric, str) for metric in metrics):
               if all(metric in ['avg_return'] for metric in metrics):
                  self.__metrics = metrics
               else:
                  invalid_metrics = [
                     metric for metric in metrics 
                     if metric not in ['avg_return']
                  ]
                  raise ValueError(
                     f"Invalid metric(s) provided: {invalid_metrics}"
                  )
         else:
            raise ValueError("'metrics' must be a list of strings.")
      
      @property
      def num_eval_episodes(self) -> int:
         """int: Get the number of episodes used to evaluate the agent."""
         return self.__num_eval_episodes

      @num_eval_episodes.setter
      def num_eval_episodes(self, num_eval_episodes: int):
         """int: Set the number of episodes used to evaluate the agent.."""
         self.__num_eval_episodes = num_eval_episodes
      
      def __repr__(self):
         return f'Result({self.metrics}, {self.num_eval_episodes})'
         
      
      

class RL:
      '''
      A central class for managing the components of a reinforcement learning workflow, including
      the environment, hyperparameters, agent configuration, and evaluation metrics.

      Args:
          result (Result): Evaluation metrics and settings for assessing agents.
          agents List[Agent]: List of reinforcement learning agents with specified algorithm name, training 
              hyperparameters, and configuration settings.
          environment (Environment): A simulated environment in which a reinforcement learning agent interacts.

      Attributes:
          result (Result): Evaluation metrics and settings for assessing agents.
          agents List[Agent]: A reinforcement learning agent with specified algorithm name, training 
              hyperparameters, and configuration settings.
          environment (Environment): A simulated environment in which a reinforcement learning agent interacts.
      '''

      def __init__(self, result: Result, agents: List[Agent], environment: Environment):
         self.result: Result = result
         self.agents: List[Agent] = agents
         self.environment: Environment = environment

      @property
      def result(self) -> Result:
         """
         Result: Get the evaluation metrics and settings for assessing agents.
         """
         return self.__result

      @result.setter
      def result(self, result: Result):
         """
         Result: Set the evaluation metrics and settings for assessing agents.
         """
         
         self.__result = result

      @property
      def agents(self) -> Agent:
         """
         List[Agent]: Get the reinforcement learning agents with specified algorithm name, 
         training hyperparameters, and configuration settings.
         """
         return self.__agents

      @agents.setter
      def agents(self, agents: List[Agent]):
         """
         List[Agent]: Set the reinforcement learning agents with specified algorithm name, 
         training hyperparameters, and configuration settings.
         """
         if isinstance(agents, list) and \
               all(isinstance(agent, Agent) for agent in agents):
               self.__agents = agents
         else:
            raise ValueError("'agents' must be a list of Agent's.")


      @property
      def environment(self) -> Environment:
         """
         Environment: Get the simulated environment in which the reinforcement learning 
         agent interacts.
         """
         return self.__environment

      @environment.setter
      def environment(self, environment: Environment):
         """
         Environment: Set the simulated environment in which the reinforcement learning 
         agent interacts.
         """
         self.__environment = environment

      def __repr__(self):
         return f'RL({self.result}, {self.agents}, {self.environment})'


