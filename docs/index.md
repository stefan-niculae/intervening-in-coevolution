# Intervening in Co-evolution

Stefan Niculae, Alejandro Marin Parra, Daniel Paul Pena, Allen Kim, Abel John
Team *ASADA*, TA: Karl Pertsch



[TOC]

## Summary



## 1 Introduction

Reinforcement Learning (RL) is paradigm of Machine Learning in which agents take actions in an environment in order to maximize some particular reward. In a Multi-Agent Reinforcement Learning (MARL) system, multiple agents interact with the environment and each-other, each having their own objectives.

This project explores ways to better train agents in an adversarial setting, such that they both co-evolve together at a commensurate pace.

### 1.1 Motivation

![img](https://lh3.googleusercontent.com/sVZ0Qaa6kPWK8yAvQ6D5FcN4EGDHwtH4wta3BZZ_qMTk8kz6oep1oq9EpzFmPPc0bT2vJlLTcATbpGWwtrxlrlurQ6mZQ63EjJpYnp_xacLizGE4N0ZM6IF2Cm7PchuNwYYkg3A1di0)

Baker et al. [1] explore how abstract concepts such as tool use can be successfully learned through a physics-based Hide-and-Seek simulation. The hiders learn to use more and more complex strategies, such as blocking passage ways when their opponents become more and more keen at finding them. By developing stronger strategies each side facilitates the other to evolve further and develop stronger counter-strategies to match them. 

![img](https://lh5.googleusercontent.com/q7tGn2d98qcpQ2c3KMST4mVOYo2Km3vJfMgs6YaCkiq7BSI6tG9S2kKELDdeBWfEj53gRAVwH_9sQlWrrciKurMxylY-lZrOwnEUN0Ld0kLw5Ad2SlpawMPERWIjpWrePRoPuN5AFDY)

RL models are notoriously hard to train and oftentimes days of training are needed to develop a strategy which is immediately apparent to humans. Paine et al. [2] showed how human demonstrations can be efficiently used (in a DQN-based algorithm) to guide the agent's exploration and point them in the right direction. Complex, sequential tasks are orders of magnitude easier to learn, in which the objectives would have otherwise not been even encountered through random, or heuristic-based exploration.

Although Hide-and-Seek environment is conceptually simple, it is computationally expensive which makes it require large models and long training times to be learned successfully. Expert demonstrations can be a way to significantly increase convergence time and success rate. But given the adversarial, multi-agent nature of the task, providing help to one team is likely to put the other at a disadvantage.

### 1.2 Hypotheses

In GANs, the Generator can learn to out-skill the Discriminator and fool it at all times. Because the Discriminator can no longer distinguish fake from real samples, it can provide no more information to the Generator on how to improve. Learning thus stagnates for both sides due to this unbalance in skill. On the other hand, if the Discriminator is allowed to catch on, both sides continue to grow stronger together.

We generalize this technique to a zero-sum game between two sides. Given:

- A measure of relative performance (e.g.: the win-rate between two tennis players, or how well the Discriminator can distinguish real from generated samples);
- A measure of absolute skill (e.g.: the precision, power and technique of each tennis player, or how believable generated samples are to a human eye)

We call *balanced co-evolution* a training instance in which both sides display similar relative performance throughout training.

We set out to test two hypotheses

1. Models that co-evolve in a balanced way develop ultimately better absolute skill;
2. Co-evolution can be balanced by intervening at when relative performance becomes unbalanced.

## 2 Setup

We test our hypotheses in a RL setting.

### 2.1 Environment

The availability of research-ready single-agent RL environments is abundant, multi-agent environments are much rarer. If we bar physics-based ones due to computational constrains, the options near none. With the extra benefit of total control and a pure Python implementation, we chose to implement our own environment.

In the *Thieves-and-Guardians* environment the two teams compete in a two-dimensional discrete grid. Each team can have multiple avatars (labeled zero through the number of total avatars, for identification convenience). The *Thieves* (illustrated in red) win an episode by collecting two treasures (yellow) while the *Guardians*(blue) win by catching all thieves. An episode ends in a tie after 100 time-steps.

Each avatar can move in four directions, one cell at a time. Trying to move to a cell occupied by a wall results in a null action, and so will trying to move to a cell occupied by a teammate. Moving to a cell occupied by an avatar of the opposing team results in the thief being caught and disappearing from the map. A thief can collect a treasure by moving on the cell occupied by it, at which point it will disappear from the map. Guardians cannot interact with treasures. At each time-step, each avatar performs one action, in order of their label.

![img](https://lh5.googleusercontent.com/8rh63sjQLyvFn8mgq7FRst16dNkBf8MwSS8cX4t-gL9W6YTcTyOrB9ORpssAtc0LIKMHsmh2hnyC5QhwPzBvTs0B8u89VJTV-zvpYVV6q_0sEiSXVHmXz2vACwNbNXcuzruzmRqZFBw)

The guardian avatar who catches a thief avatar receives a reward of +1. The same reward is received by the thief avatar  who collects a treasure. Due to the way actions are resolved, it is possible that a thief collects a treasure, only to be ambushed by a guardian on the other side of it, sacrificing itself after successfully collecting the treasure.

Policy convergence and learned strategies are heavily influenced by reward function design. We had some success with a providing a reward to thieves proportional to the number of steps they took to reach it, since the beginning of the episode. Another variation was to awarding +1 reward to all guardian avatars upon catching a thief, going by the intuition that even if one avatar was the one to catch the thief, all guardians contributed to this catch by blocking passages or otherwise forcing the thief into submission. Negative rewards, either for being caught/failing to stop a treasure collection, or a small constant for every time-step turned out to interact badly with the RL algorithm and weights optimizer, severely hindering learning. In the end, we favored parsimony and went with the formulation above.

### 2.2 Agents

Each agent receives a 



### 2.3 Model

### 2.4 Intervention

### 2.5 Evaluation



## 3 Results



## 4 Future Work



env: 

- reward