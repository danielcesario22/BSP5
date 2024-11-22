import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks.q_network import QNetwork
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory

class DQNTrainer:
    def __init__(self, trainer_config, eval_param, train_env, eval_env):
        # Store configurations and environments
        self.agent_name = trainer_config["agent_name"]
        self.agent_config = trainer_config["agent_config"]
        self.hyper_param = trainer_config["hyper_param"]
        self.eval_param = eval_param
        self.train_env = train_env
        self.eval_env = eval_env

        # Extract agent configurations
        fc_layer_params = self.agent_config["fc_layer_params"]
        loss_function = self.agent_config["loss_function"]

        # Extract hyperparameters
        optimizer = self.hyper_param["optimizer"]
        learning_rate = self.hyper_param["learning_rate"]

        # Define the Q-Network
        q_net = QNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=fc_layer_params
        )

        # Create the DQN Agent
        train_step_counter = tf.Variable(0)
        self.agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer(learning_rate),
            td_errors_loss_fn=loss_function,
            train_step_counter=train_step_counter
        )
        self.agent.initialize()

    def _collect_step(self, environment, policy, buffer):
        """Collect a single step of experience."""
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        buffer.add_batch(traj)

    def setup_replay_buffer(self, replay_buffer_capacity=None):
        """Set up the replay buffer."""
        replay_buffer_capacity = replay_buffer_capacity or self.hyper_param["replay_buffer_capacity"]
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=replay_buffer_capacity
        )
        return replay_buffer

    def train(self, num_iterations=None, collect_steps_per_iteration=None, 
              log_interval=None, eval_interval=None, batch_size=None, replay_buffer_capacity=None):
        """
        Train the agent with optional override for parameters.
        """
        # Use provided parameters or defaults
        num_iterations = num_iterations or self.hyper_param["num_iterations"]
        collect_steps_per_iteration = collect_steps_per_iteration or self.hyper_param["collect_steps_per_iteration"]
        log_interval = log_interval or self.hyper_param["log_interval"]
        eval_interval = eval_interval or self.hyper_param["eval_interval"]
        batch_size = batch_size or self.hyper_param["batch_size"]

        # Metrics and evaluation parameters
        metrics = self.eval_param["metrics"]
        num_eval_episodes = self.eval_param["num_eval_episodes"]

        # Initialize replay buffer
        replay_buffer = self.setup_replay_buffer(replay_buffer_capacity)

        # Collect initial random experience
        random_policy = random_tf_policy.RandomTFPolicy(
            self.train_env.time_step_spec(), self.train_env.action_spec()
        )
        for _ in range(1000):
            self._collect_step(self.train_env, random_policy, replay_buffer)

        # Prepare dataset
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            sample_batch_size=batch_size,
            num_steps=2
        ).prefetch(3)
        iterator = iter(dataset)

        # Training loop
        print(f'-- Training {self.agent_name} --')
        for iteration in range(num_iterations):

            # Collect experience
            for _ in range(collect_steps_per_iteration):
                self._collect_step(self.train_env, self.agent.collect_policy, replay_buffer)

            # Sample from replay buffer and train
            experience, _ = next(iterator)
            train_loss = self.agent.train(experience).loss
            step = self.agent.train_step_counter.numpy()

            # Log progress
            if step % log_interval == 0:
                print(f"Step {step}: loss = {train_loss}")

            # Evaluate the agent
            if step % eval_interval == 0:
                self._evaluate_and_log(metrics, num_eval_episodes, step)


    def _evaluate_and_log(self, metrics, num_eval_episodes, step):
        """Evaluate the agent and log metrics."""
        for name, metric in metrics.items():
            value = metric(self.eval_env, self.agent.policy, num_eval_episodes)
            print(f"Step {step}: {name} = {value}")

    def evaluate(self, num_eval_episodes=None):
        """
        Evaluate the trained agent.
        """
        num_eval_episodes = num_eval_episodes or self.eval_param["num_eval_episodes"]
        metrics = self.eval_param["metrics"]
        results = {}
        for name, metric in metrics.items():
            value = metric(self.eval_env, self.agent.policy, num_eval_episodes)
            results[name] = value
        return results