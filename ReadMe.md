REINFORCE CartPole Implementation

Overview

This project implements a REINFORCE policy gradient agent to solve the CartPole-v1 environment from OpenAI Gym. The implementation includes improvements to address common reinforcement learning issues such as policy forgetting, overfitting, and poor exploration.

Key Features

- State normalization using a blend of learned and fixed statistics
- Conservative policy updates using a slowly updated target network
- Replay of high-reward episodes to reinforce successful strategies
- Entropy regularization to maintain exploration
- Shaped rewards that emphasize pole angle control and penalize poor positioning
- Plateau detection to trigger adaptive learning behavior

Directory Structure

- main.py: Contains the training and testing logic
- save_model/: Stores trained model checkpoints
- results/: Contains graphs showing training progress
- README.md: This file

Setup

1. Install required packages and python if not there

Run the following command to install dependencies
```bash
pip install "gym==0.26.2"
pip install torch 
pip install matplotlib 
pip install numpy==1.23.5

```


Training

Run the following command to train the agent
```bash
python cartpole_reinforce.py

```
```code
- Trains for 4000 episodes by default
- Saves model checkpoints in the save_model directory
- Saves graphs to the results directory every 100 episodes
```
Testing
```jsunicoderegexp
The script automatically tests the agent after training. It


- Loads the best available model from save_model
- Runs 10 test episodes
- Renders the CartPole environment and displays scores

Model Files

- pg_cartpole_best.pt: Best average score across episodes
- pg_cartpole_checkpoint.pt: Saved every 50 episodes
- pg_cartpole_solved_1000.pt: Saved when the agent scores 1500 for 150 episodes

Graphs

- progress_metrics_A_episode_{number}.png: Includes score trends, recent scores, and policy loss
- progress_metrics_B_episode_{number}.png: Shows learning rate, conservatism levels, and score distribution

Performance Summary

- Achieves consistent scores of 10000 in CartPole-v1
- Remains stable after long training
- Learns to control pole angle and resist overfitting to early rewards


```
Key Fixes Applied

Issue: Forgets good behavior  
Fix: Replay good episodes, conservative updates

Issue: Gets stuck at high score  
Fix: Entropy and exploration temperature increase

Issue: Overfits to short episodes  
Fix: Shaped rewards, diverse initial state injection

Issue: Ignores pole angle  
Fix: Penalizes high pole angles with custom reward shaping

Issue: Long learning plateaus  
Fix: Automatic replay and learning rate scheduling

Screenshot Instructions

1. Run the script to generate result images in the results directory
2. Sample graphs include training trends, recent scores, and score histograms
3. Use any image viewer to open the PNG files
