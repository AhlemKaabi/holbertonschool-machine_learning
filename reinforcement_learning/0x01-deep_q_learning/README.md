# **Deep Q-learning**

## **Learning Objectives**

* What is Deep Q-learning?
* What is the policy network?
* What is replay memory?
* What is the target network?
* Why must we utilize two separate networks during training?
* What is keras-rl? How do you use it?
	* keras-rl implements some state-of-the art deep reinforcement learning algorithms in Python and seamlessly integrates with the deep learning library Keras.

### **Installing Keras-RL**

```
pip install --user keras-rl
```

### **Dependencies**
```
pip install --user keras==2.2.4
pip install gym[atari]
pip install --user Pillow
pip install --user h5py
```

TRAIN: utilizes keras, keras-rl, and gym to train an agent that can play Atari’s Breakout:

* Your script should utilize keras-rl‘s DQNAgent, SequentialMemory, and EpsGreedyQPolicy
* Your script should save the final policy network as policy.h5


https://www.codeproject.com/Articles/5271947/Introduction-to-OpenAI-Gym-Atari-Breakout

https://github.com/nicknochnack/KerasRL-OpenAI-Atari-SpaceInvadersv0/blob/main/Space%20Invaders%20Walkthrough.ipynb

https://www.youtube.com/watch?v=hCeJeq8U0lo

https://www.youtube.com/watch?v=wrBUkpiRvCA&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=12