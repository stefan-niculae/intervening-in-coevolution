# Intervening in Co-evolution

Stefan Niculae, Alejandro Marin Parra, Daniel Paul Pena, Allen Kim, Abel John
Team *ASADA*, TA: Karl Pertsch



[TOC]

## Summary



## 1 Introduction

Reinforcement Learning (RL) is paradigm of Machine Learning in which agents take actions in an environment in order to maximize some particular reward. In a Multi-Agent Reinforcement Learning (MARL) system, multiple agents interact with the environment and each-other, each having their own objectives.

This project explores ways to better train agents in an adversarial setting, such that they both co-evolve together at a commensurate pace.

### 1.1 Motivation

Baker et al. [1] explore how abstract concepts such as tool use can be successfully learned through a physics-based Hide-and-Seek simulation. The hiders learn to use more and more complex strategies, such as blocking passage ways when their opponents become more and more keen at finding them. By developing stronger strategies each side facilitates the other to evolve further and develop stronger counter-strategies to match them. 

RL models are notoriously hard to train and oftentimes days of training are needed to develop a strategy which is immediately apparent to humans. Paine et al. [2] showed how human demonstrations can be efficiently used (in a DQN-based algorithm) to guide the agent's exploration and point them in the right direction. Complex, sequential tasks are orders of magnitude easier to learn, in which the objectives would have otherwise not been even encountered through random, or heuristic-based exploration.

Although Hide-and-Seek environment is conceptually simple, it is computationally expensive which makes it require large models and long training times to be learned successfully. Expert demonstrations can be a way to significantly increase convergence time and success rate. But given the adversarial, multi-agent nature of the task, providing help to one team is likely to put the other at a disadvantage.

### 1.2 Hypotheses

In GANs, the Generator can learn to out-skill the Discriminator and fool it at all times. Because the Discriminator can no longer distinguish fake from real samples, it can provide no more information to the Generator on how to improve. Learning thus stagnates for both sides due to this unbalance in skill. On the other hand, if the Discriminator is allowed to catch on, both sides continue to grow stronger together.

We generalize this technique to a zero-sum game between two sides. Given:

- A measure of relative performance (e.g. the win-rate between two tennis players, or how well the Discriminator can distinguish real from generated samples);
- A measure of absolute skill (e.g. the precision, power and technique of each tennis player, or how believable generated samples are to a human eye)

We call *balanced co-evolution* a training instance in which both sides display similar relative performance throughout training.

We set out to test two hypotheses

1. Models that co-evolve in a balanced way develop ultimately better absolute skill
2. Co-evolution can be balanced by intervening at when relative performance becomes unbalanced

## 2 Setup

We test our hypotheses in a RL setting

### 2.1 Environment

In order to test out

### 2.2 Agents

### 2.3 Model

### 2.4 Intervention



## 3 Results



## 4 Future Work