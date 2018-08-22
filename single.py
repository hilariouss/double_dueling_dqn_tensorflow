import argparse
import logging
import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf

from models.DDDQN import DDDQN
from utilities import get_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

config = get_config("config.json")

"""
# Using tf flags instead of argparse
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_episodes", config.num_episodes,
                            "Maximum number of episodes to train.")
"""
parser = argparse.ArgumentParser()

# Environment configurations
parser.add_argument(
    "--env", type=str, default=config.env,
    help="Environment name.")

# Netowrk configurations
parser.add_argument(
    "--learning-rate", type=float,
    default=config.learning_rate,
    help="Learning rate for the model.")

parser.add_argument(
    "--steps_per_target_update", type=int,
    default=config.steps_per_target_update,
    help="Number of steps after which to update target network weights.")

parser.add_argument(
    "--gamma", type=float, default=config.gamma,
    help="Discount factor.")

parser.add_argument(
    "--common_net_hidden_dimensions", type=int, nargs="+",
    default=config.common_net_hidden_dimensions,
    help="List of hidden layers dimensions for common network.")

parser.add_argument(
    "--buffer_size", type=float, default=config.buffer_size,
    help="Size of experience replay buffer.")

# Trainer configurations
parser.add_argument(
    "--num-episodes", type=int,
    default=config.num_episodes,
    help="Maximum number of episodes to train.")

parser.add_argument(
    "--max_checkpoints_to_keep", type=int,
    default=config.max_checkpoints_to_keep,
    help="Maximum number of recent checkpoints to store.")

parser.add_argument(
    "--mini_batch_size", type=int, default=config.mini_batch_size,
    help="Batch size.")

parser.add_argument(
    "--consecutive_successful_episodes_to_stop", type=int,
    default=config.consecutive_successful_episodes_to_stop,
    help="Consecutive number successful episodes above min avg to stop "
         "training.")

# Filesystem configurations
parser.add_argument(
    "--summray_dir", type=str, default=config.summray_dir,
    help="Directory path to store tensorboard summaries.")

parser.add_argument(
    "--save_after_num_episodes", type=int,
    default=config.save_after_num_episodes,
    help="Number of episodes after which to save our model.")

parser.add_argument(
    "--checkpoints_dir", type=str, default=config.checkpoints_dir,
    help="Directory path to store model checkpoints.")


config = parser.parse_args()
env = gym.make(config.env)

num_input_neurons = len(env.reset())
num_ouptut_neurons = env.action_space.n


replay_buffer = deque(maxlen=config.buffer_size)
last_n_rewards = deque(maxlen=config.consecutive_successful_episodes_to_stop)


def train_dqn(main_dqn, target_dqn, mini_batch):
    """
    param: mini_batch: From the randomly sampled minbatch from replay-buffer,
                       it's a list of experiences in the form of
                       `(state, action, reward, next_state, done)`
    """
    states = [x[0] for x in mini_batch]
    states = np.vstack(states)

    actions = np.array([x[1] for x in mini_batch])
    rewards = np.array([x[2] for x in mini_batch])
    next_states = np.vstack([x[3] for x in mini_batch])
    done = np.array([x[4] for x in mini_batch])

    target_output_next_states = target_dqn.predict(next_states)

    # For double DQN: select the best action for next state
    main_output_next_states = main_dqn.predict(next_states)

    selected_best_actions = np.argmax(main_output_next_states, axis=1)
    target_output_for_selected_actions = target_output_next_states[
        np.arange(len(states)), selected_best_actions]

    target_q_vals = (
        rewards + config.gamma * target_output_for_selected_actions * (1 - done))

    main_output = main_dqn.predict(states)
    main_output[np.arange(len(states)), actions] = target_q_vals

    loss, optimizer = main_dqn.update(states, main_output)

    return loss


with tf.Session() as sess:
    main_dqn = DDDQN(session=sess,
                     scope_name="q_main",
                     input_size=num_input_neurons,
                     hidden_layer_sizes=config.common_net_hidden_dimensions,
                     output_size=num_ouptut_neurons,
                     learning_rate=config.learning_rate)

    target_dqn = DDDQN(session=sess,
                       scope_name="q_target",
                       input_size=num_input_neurons,
                       hidden_layer_sizes=config.common_net_hidden_dimensions,
                       output_size=num_ouptut_neurons,
                       learning_rate=config.learning_rate)

    writer = tf.summary.FileWriter(config.summray_dir)
    writer.add_graph(sess.graph)

    episode_reward_to_log = tf.Variable(0.0)
    last_n_avg_reward_to_log = tf.Variable(0.0)

    tf.summary.scalar('loss', main_dqn.loss)
    tf.summary.scalar('episode_reward', episode_reward_to_log)
    tf.summary.scalar('last_n_avg_reward', last_n_avg_reward_to_log)

    writer_op = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    # Make them identical to begin with
    sess.run(DDDQN.create_copy_operations("q_main", "q_target"))

    saver = tf.train.Saver(tf.global_variables(),
                           max_to_keep=config.max_checkpoints_to_keep)

    for ep_num in range(config.num_episodes):
        state = env.reset()
        done = False
        episode_reward, loss, steps = 0, 0, 0

        # epsilon decay
        epsilon = 1. / ((ep_num / 10) + 1)

        while not done:
            # select the action
            action = None
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(main_dqn.predict(state))

            # execute the action
            next_state, reward, done, _ = env.step(action)

            if done:
                reward = -1

            # add to the buffer
            replay_buffer.append((state, action, reward, next_state, done))

            # sample from the buffer and train
            if len(replay_buffer) > config.mini_batch_size:
                mini_batch = random.sample(replay_buffer, config.mini_batch_size)
                loss = train_dqn(main_dqn, target_dqn, mini_batch)

            if steps % config.steps_per_target_update == 0:
                sess.run(DDDQN.create_copy_operations("q_main", "q_target"))

            episode_reward += reward
            steps += 1
            state = next_state

        last_n_rewards.append(episode_reward)
        last_n_avg_reward = np.mean(last_n_rewards)

        summary = sess.run(writer_op, {
            main_dqn.loss: loss,
            episode_reward_to_log: np.array(episode_reward),
            last_n_avg_reward_to_log: last_n_avg_reward})

        writer.add_summary(summary, ep_num)
        writer.flush()

        logger.info("Episode: {}  reward: {}  loss: {}  last_{}_avg_reward: {}"
                    .format(ep_num,
                            episode_reward,
                            loss,
                            config.consecutive_successful_episodes_to_stop,
                            last_n_avg_reward))

        # Save after every `save_after_num_episodes` episodes
        if ep_num % config.save_after_num_episodes == 0:
            saver.save(sess, config.checkpoints_dir, ep_num)

        # Stopping criteria
        if len(last_n_rewards) == config.consecutive_successful_episodes_to_stop \
                and last_n_avg_reward > config.min_average_reward_for_stopping:
            logger.info("Solved after {} epsiodes".format(ep_num))
            break
