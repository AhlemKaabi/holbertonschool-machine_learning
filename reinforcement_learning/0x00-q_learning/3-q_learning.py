#!/usr/bin/env python3
"""
    Q-learning
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Method:
     -------
         Uses epsilon-greedy to determine the next action.

    Parameters:
    -----------
        Q(numpy.ndarray): containing the q-table
        state: the current state
        epsilon: the epsilon to use for the calculation
    Returns:
    --------
        The next action index.
    """
    if np.random.uniform(0, 1) < epsilon:
        # Explore: select a random action
        index = np.random.randint(Q.shape[1])
    else:
        # Exploit: select the action with max value (future reward)
        index = np.argmax(Q[state, :])
    return index


def train(env, Q, episodes=5000,
          max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1,
          min_epsilon=0.1, epsilon_decay=0.05):
    """
    Method:
    -------
        performs Q-learning:

    Parameters:
    -----------
    env: the FrozenLakeEnv instance
    Q(numpy.ndarray): containing the Q-table
    episodes(integer): the total number of episodes to train over
    max_steps(integer): the maximum number of steps per episode
    alpha(float): the learning rate
    gamma(float): the discount rate
    epsilon(integer): the initial threshold for epsilon greedy
    min_epsilon(float): the minimum value that epsilon should decay to
    epsilon_decay(float): the decay rate for updating epsilon between episodes

     Returns:
    --------
        (Q, total_rewards)
            Q the updated Q-table
            total_rewards(list) containing the rewards per episode


    ** When the agent falls in a hole, the reward should be updated to be -1 **
    """
    # https://www.youtube.com/watch?v=QK_PP_2KgGE
    # https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb

    # List of rewards
    total_rewards = []
    max_epsilon = epsilon

    # For life or until learning is stopped
    for episode in range(episodes):
        # Reset the environment
        state = env.reset()
        done = False
        steps_rewards = 0

        for step in range(max_steps):
            # Choose an action a in the current world state (s)
            action = epsilon_greedy(Q, state, epsilon)

            # Take the action (a) and observe the outcome state(s')
            # and reward (r)
            new_state, reward, done, info = env.step(action)

            # Update
            # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # Q[new_state,:] : all the actions we can take from new state
            Q[state, action] = (
                Q[state, action] + alpha *
                (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
                )

            steps_rewards += reward

            # Our new state is state
            state = new_state

            # If done (if we're dead) : finish episode
            if done is True:
                break

        # Reduce epsilon (because we need less and less exploration)
        epsilon = (
            min_epsilon + (max_epsilon - min_epsilon) *
            np.exp(-epsilon_decay * episode)
            )

        total_rewards.append(steps_rewards)

    return Q, total_rewards
