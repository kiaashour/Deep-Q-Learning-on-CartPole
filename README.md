# Deep Q-Learning on the CartPole environment

This repository implements and compares the DQN and DDQN algorithms on the CartPole environment.

A discussions of these methods can be found in the papers [Mnih et al., 2013](https://arxiv.org/abs/1312.5602) and [Hasselt et al., 2015](https://arxiv.org/abs/1509.06461).

---

## Contents of repository

1. The `utils` directory contains scripts for:

  1. Implementing DQN networks with PyTorch

  2. Implementing DDQN networks with Pytroch

  3. Utility functions for plotting, calculating losses, etc.

2. The `Experiments` directory contains a Jupyter notebook for the experiments ran in this project.
   The experiments compare the performance of DQN and DDQN on the CartPole environment.  
  

---

## Prerequisites

Before you begin, ensure that you have the following:

- Python 3.8 or higher
- Virtualenv (optional, but recommended)

---

## Setting up a virtual environment

It is recommended to use a virtual environment to keep the dependencies for this project separate from other projects on your system. To set up a virtual environment:

1. If you don't have virtualenv installed, run `pip install virtualenv`
2. Navigate to the project directory and create a virtual environment by running `virtualenv env`
3. Activate the virtual environment by running `source env/bin/activate`

---

## Installing dependencies

To install the dependencies for this project, run the following command:

`pip install -r requirements.txt`

This will install all of the packages listed in the `requirements.txt` file.

---

## Cloning the repository

To clone this repository, run the following command:

`git clone https://github.com/kiaashour/Deep-Q-Learning-on-CartPole.git`
