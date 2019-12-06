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
2. Co-evolution can be balanced by intervening when relative performance becomes unbalanced.

## 2 Setup

We test our hypotheses in a RL setting.

> desc

### 2.1 Environment

The availability of research-ready single-agent RL environments is abundant, multi-agent environments are much rarer. If we bar physics-based ones due to computational constrains, the options near none. With the extra benefit of total control and a pure Python implementation, we chose to implement our own environment.

In the *Thieves-and-Guardians* environment the two teams compete in a two-dimensional discrete grid. Each team can have multiple avatars (labeled zero through the number of total avatars, for identification convenience). The *Thieves* (illustrated in red) win an episode by collecting two treasures (yellow) while the *Guardians*(blue) win by catching all thieves. An episode ends in a tie after 100 time-steps.

Each avatar can move in four directions, one cell at a time. Trying to move to a cell occupied by a wall results in a null action, and so will trying to move to a cell occupied by a teammate. Moving to a cell occupied by an avatar of the opposing team results in the thief being caught and disappearing from the map. A thief can collect a treasure by moving on the cell occupied by it, at which point it will disappear from the map. Guardians cannot interact with treasures. At each time-step, each avatar performs one action, in order of their label.

![img](https://lh5.googleusercontent.com/8rh63sjQLyvFn8mgq7FRst16dNkBf8MwSS8cX4t-gL9W6YTcTyOrB9ORpssAtc0LIKMHsmh2hnyC5QhwPzBvTs0B8u89VJTV-zvpYVV6q_0sEiSXVHmXz2vACwNbNXcuzruzmRqZFBw)

The guardian avatar who catches a thief avatar receives a reward of +1. The same reward is received by the thief avatar  who collects a treasure. Due to the way actions are resolved, it is possible that a thief collects a treasure, only to be ambushed by a guardian on the other side of it, sacrificing itself after successfully collecting the treasure.

Policy convergence and learned strategies are heavily influenced by reward function design. We had some success with a providing a reward to thieves proportional to the number of steps they took to reach it, since the beginning of the episode. Another variation was to awarding +1 reward to all guardian avatars upon catching a thief, going by the intuition that even if one avatar was the one to catch the thief, all guardians contributed to this catch by blocking passages or otherwise forcing the thief into submission. Negative rewards, either for being caught/failing to stop a treasure collection, or a small constant for every time-step turned out to interact badly with the RL algorithm and weights optimizer, severely hindering learning. In the end, we favored parsimony and went with the formulation above.

The asymmetry of the objectives is a main cause of unbalanced co-evolution.

### 2.2 Agents

![img](https://lh5.googleusercontent.com/zdStsQuh5KJIMe4LVsrFpEvx_U2FU0VwYbMOGWHExg-l4aSor3u6flgLqWN3Lb60RvK66eDtQ1_Te-Zz8d9ST6iTs7lfrYlesuqG65v4mUabgLSIFAx-wF0GDZP3NGqtlSCIjl17fdk)

There are two agents, one for the thieves and one for the guardians. An agent picks an action for each avatar in the team in isolation. Each agent has their own copy of the model, to prevent intervention cross-contamination. Each avatar has their own transitions buffer, to facilitate dealing with cases in which one thief dies while the game continues.

Each avatar receives a different view of environment state. The state features are made up of five 9 by 9 boolean matrices, indicating the positions of:

1. Itself
2. Teammates
3. Opponents
4. Treasures
5. Walls

The entire map is visible to all agents at all times.

A 2D convolution, with 8 filters encodes each one-hot representation of cells individually (kernel size 1). Two more convolutions encode information about immediate and extended neighbors (kernel size 3). No padding is used in order to drastically reduce the dimensionality of the encoding. The result is the latent representation of the state. It is flattened and passed through a series of fully-connected layers which ultimately branch into actor and critic heads.

ReLU non-linearity and Batch Normalization is applied after each layer (except the final ones). We found that other variations of ReLU do not out-weigh their computational cost. Hyperbolic tangent non-linearities performed worse.

A recurrent component can be added to the latent representation. Despite respecting the Markovian assumption, it significantly improves training performance, but reduces computational one by an order of magnitude, rendering it unfeasible.

The models are trained with a Policy Gradient algorithm, with a discount factor of 0.97. PPO, and SAC (for discrete actions [12]) performed slightly, respectively significantly worse given the same training time. We suspect this to have been the case because the additional parameters to learn introduce an overhead which brings no additional benefit in the relatively short training sessions.

A (normalized) coordinates-based state representation (in which convolutional layers are replaced with fully-connected ones) failed to converge to any meaningful policies.

Deeper and/or wider (larger layer sizes), even though they have more expressive power, would have required much longer training which rendered them impractical for our constrains.

### 2.3 Intervention

We use the relative win-rate of the previous policy iteration (around a hundred episodes) to gauge the relative performance of thieves and agents. If the win-rate is not within a target range, for the next iteration one of four intervention tactics will be applied.

When the win-rate dips too low, we provide some exploration guidance to the losing side. We observed that in this case, the team's policy degenerates into a sequence of null actions, likely due to the fact that they never managed to discover sources of positive rewards. In this case, providing them with a successful trajectory would increase the chances of that the agent explores and exploits similar trajectories in the future. At each time-step, x% of the time, we force the agent to take a step in the direction of their objective (either a treasure or a goal), instead of sampling from their regular action distribution. Well trained models out-maneuver this fixed strategy, but its purpose is not to be optimal, rather help in cases where one side is severely lacking.

In contrast, when one side wins too often we want to impede their evolution, in order for the loser to catch up. We address this by either:

1. Halting the winner's training, allowing the loser to continue learning and giving them time to study their opponents better;
2. Constraining the winner's quality of the latent representation, as a non-intrusive way of regularizing the its prowess;
3. Degrading the policy by making the winner pick actions uniformly x% of the time rather than their learned policy. This is equivalent to $\epsilon$-greedy exploration which can backfire as a technique to help the winner in the long term.

Constraining the latent representation is a deceivingly complex operation. Since it is nearly entirely dependent on the input features (bar the bias terms), ensuring the latent representation does not encode too much information about the raw features would be a good formulation. Due to their significantly different structures, it is extremely non-trivial to measure non-linear correlations between two arbitrary sets of values, of different sizes, with no particular prior distribution in which most of the information is encoded by the number positions rather their magnitudes [?, ?]. To address this, we take a slide a gaussian kernel across all values, to produce a differentiable soft-histogram which can be discretized. We are looking for a measure similar to Mutual Information, which is formulated as follows for discrete distribution PMFs:
$$
\text I(X; Y) = \sum_{y}\sum_x p_{X,Y}(x, y) \log \frac {p_{X,Y}(x, y)} {p_X(x) p_Y(y)}
$$
We can have $p_X(x)$ and $p_Y(y)$ straight from the PMF, but the joint distribution makes little sense since there is no immediate pairing between value buckets of the raw and latent state representation. We then turn our attention to a related measure, cross entropy, which is formulated as follows for discrete distribution PMFs:
$$
\text H(p, q) = - \sum_x p(x) \log q(x)
$$
We purposefully forgo the assumption of the same support, by discretizing the PMF of inputs and latent values between their minimum and maximum values. This quantity serves as an indication of how good of an encoding $q$ is for events encoded by $p$. Events in this case are soft-histogram magnitudes. Since our hand-crafted encoding is obviously not optimal, encoding the input as it is also has a cost itself, so we normalize by subtracting $\text H(p, p)$, which is precisely its entropy. Thus, the latent loss is formulated as:
$$
\mathcal L_{\text{latent}} = \text H(i, l) - \text H(i, i)
$$
where $i$ and $l$ represent the discretized soft-histograms of the input and latent features, respectively.

A more theoretically-sound measure could be formulated if the latent representation was variational, representing multi-gaussian means and standard deviations. Then a straightforward Conditional VAE-like loss could be employed to constrain it to a uniform distribution. But this formulation requires a major change in model architecture, by sampling according to encoder (convolutional layers) outputs and constraining the latent distribution to be uniform. This turned out to be a too high of a cost, as such a variational architecture failed to converge on any meaningful strategy.

We do not explore adding (zero-mean gaussian) noise to the input features or model parameters, in order to avoid the trap of the opponent learning to rely on these artificial weaknesses which would be taken away as soon as they learn how to exploit them successfully.

### 2.4 Evaluation



## 3 Results



## 4 Future Work



env: 

- reward