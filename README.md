# Assessing Generalization in Reward Learning

To read the blog post writeup, [click here](https://towardsdatascience.com/assessing-generalization-in-reward-learning-intro-and-background-da6c99d9e48).


## Intro

We want to build scalable, aligned RL agents; one approach is with _reward learning_.
The ideal reward learning algorithm demonstrates good generalization properties; here, we use the Procgen benchmark to test some of these algorithms.

## Approach

So far, we investigated two different reward learning architectures:
- [Deep Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
- [Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations](https://arxiv.org/abs/1904.06387v5)

There are others we can test, but given limited time we could only look at these two.
We implemented PPO in Pytorch using the [baselines](https://github.com/openai/baselines/) package (upgrading to [stable-baselines](https://github.com/hill-a/stable-baselines) pending) by adapting them to allow for a custom reward learning function, which we would be able to learn.

Procgen has 16 different environments; based on some initial testing we decided to focus on 4 environments: coinrun, bigfish, starpilot, and fruitbot.

## Code

An explanation for how to run the T-REX code is written in the `trex` repository.
