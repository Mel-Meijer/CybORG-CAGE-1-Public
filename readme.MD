# Project Description
The game of cyber security defence is intrinsically unbalanced, as a defender must remain continuously alert in order to identify and counter each attack, whereas an attacker only requires one successful attempt. To address this inequality, this project explores the capability of autonomous deep reinforcement learning models to defend a network against a simulated attacker. This project aims to tackle the first challenge presented by the Cyber Autonomy Gym for Experimentation (CAGE TTCP Working Group). The solution presented in this project trains and compares three different algorithms - PPO, DQN, and A2C - in their ability to defend a network from two different attacking agents: one with prior knowledge of the network structure and another that must explore the network subnets to reach its target. The aim for this experiment is to evaluate which of these three models achieves the best performance, and whether they receive higher rewards than a baseline agent that takes random actions. The results show that all trained models are able to outperform the baseline agent, with the PPO algorithm achieving the best results.

# Code Reference
The code produced for the purpose of this project is based on, and makes use of the source code from the Cyber Autonomy Gym for Experimentation (CAGE) Challenge 1, as cited by the below reference:
```
@misc{cage_challenge_1,
  Title = {Cyber Autonomy Gym for Experimentation Challenge 1},
  Note = {Created by Maxwell Standen, David Bowman, Son Hoang, Toby Richer, Martin Lucas, Richard Van Tassel},
  Publisher = {GitHub},
  Howpublished = {\url{https://github.com/cage-challenge/cage-challenge-1}},
  Year = {2021},
}
```

# Setup and Installation

## Clone and install the CAGE Challenge repository
```git clone https://github.com/cage-challenge/cage-challenge-1.git```

## From the cage-challenge-1/CybORG directory
```pip install -e .```

## Install this project's requirements
```pip install -r requirements.txt```

# What is included in this project repository?

```New_Env_Wrappers/:``` Directoy for new environment wrappers created for this project, in addition to those provided by the original CybORG environment. 

```Tuning/:``` Directory for tuning files including tuning python progam, tuning results databases for each model and a python program to access said results.  

```Training:``` Directory for training files including the train.py program and saved final models for each algorithm. 

```Evaluation:``` Directory for evaluation files including the evaluation.py program, output evaluation text files for each agent and the blue random agent class. 

```Visualisation:``` Directory for visualisation files including network diagrams, program to run one user-specified episode, an example episode results log file, the log_to_gif pyton program to convert such an episode log file into a GIF animation of the network state and saved final GIFs for each agent. 

Example commands for how to run each file have been included in the program code where applicable.