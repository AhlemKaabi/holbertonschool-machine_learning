# **Policy Gradients**

## **Learning Objectives**

* What is Policy?
	* A policy is defined as the probability distribution of actions given a state - [Policy Gradients in a Nutshell](https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d)
	* ![policy](./img/policy.png)
* How to calculate a Policy Gradient?
	* The objective of a Reinforcement Learning agent is to maximize the “expected” reward when following a policy π. Like any Machine Learning setup, we define a set of parameters θ (e.g. the coefficients of a complex polynomial or the weights and biases of units in a neural network) to parametrize this policy — π_θ​ (also written a π for brevity). - [Policy Gradients in a Nutshell](https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d)
	* **Reinforcement Learning Objective**: Maximize the “expected” reward following a parametrized policy - [Policy Gradients in a Nutshell](https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d)
	* ![J of theta](./img/J_of_theta.png)
	* Like any other Machine Learning problem, if we can find the parameters θ⋆ which maximize J, we will have solved the task. A standard approach to solving this maximization problem in Machine Learning Literature is to use Gradient Descent. In gradient Descent, we keep stepping through the parameters using the following update rule - [Policy Gradients in a Nutshell](https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d)
	* ![gradient](./img/gradient.png)
	* **The Policy Gradient Theorem**: The derivative of the expected reward is the expectation of the product of the reward and gradient of the log of the policy π_θ​. - [Policy Gradients in a Nutshell](https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d)
	* ![theorem](./img/theorem.png)
* What and how to use a Monte-Carlo policy gradient?
	* [#Policy Gradient Algorithms#REINFORCE](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#policy-gradient-algorithms)


## **Resources**

* [How Policy Gradient Reinforcement Learning Works](https://www.youtube.com/watch?v=A_2U6Sx67sE)
* [Policy Gradients in a Nutshell](https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d)
* [RL Course by David Silver - Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs)
* [Reinforcement Learning 6: Policy Gradients and Actor Critics](https://www.youtube.com/watch?v=bRfUxQs6xIM)
* [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
