#!/usr/bin/env python3

import gym
import keras

from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
env = gym.make('BreakoutNoFrameskip-v4')
env.reset()
DQN_agent = DQNAgent(
    model=keras.models.load_model('policy.h5'),
    nb_actions=env.action_space.n,
    memory=SequentialMemory(
        limit=1000000,
        window_length=4
    ),
    policy=GreedyQPolicy()
)
DQN_agent.compile(
    optimizer=Adam(
        lr=.00025,
        clipnorm=1.0
    ),
    metrics=['mae']
)