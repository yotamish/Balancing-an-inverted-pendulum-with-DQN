# Balancing-an-inverted-pendulum-with-DQN

Deep Q-network for self balancing an inverted pendulum both in simulation and reality.

Simulation (Python code, big credit to Ayal Taitler):
* The learning environment is a custom built simulation based on the 'Cartpole' simulation of OpenAI gym ('cartrpole2.py'; add it as a new environment to open AI gym).
* The agent is a double DQN agent implemented in Python using Tensorflow ('DQN2.py').

Reality (Matlab code)
* The agent is a DQN agent implemented in Matlab (with several changes to the standard DQN algorithm, you can read about the changes in the project report).
* The agent interacts with the real system using two incremental encoders and a DC motor (more about specs in the report).
