#!/usr/bin/env python3
import gym
import random

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
print("Import dependencies OK")

# Environment

env = gym.make('BreakoutNoFrameskip-v4') # create the environment
height, width, channels = env.observation_space.shape
actions = env.action_space.n # number of actions

# Actions
print(env.action_space)
print(env.get_action_meanings())

# Create a Deep Learning Model with Keras

def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model = build_model(height, width, channels, actions)
print(model.summary())

# Build Agent with Keras-RL


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.,
        value_min=.1,
        value_test=.2,
        nb_steps=1000
        )
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000
                  )
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-4))



dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

dqn.save_weights('policy.h5', overwrite=True)