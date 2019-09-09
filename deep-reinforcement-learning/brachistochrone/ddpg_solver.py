import numpy as np
import gym
from gym import wrappers
from datetime import datetime
from ddpg import *
import brachistochrone as bc

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
  parser.add_argument('--env', type=str, default='Pendulum-v0')
  parser.add_argument('--hidden_layer_sizes', type=int, default=200)
  parser.add_argument('--num_layers', type=int, default=2)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--num_train_episodes', type=int, default=300)
  parser.add_argument('--save_folder', type=str, default=None)
  args = parser.parse_args()

  a = gym.wrappers.TimeLimit(bc.BrachistochroneEnv("ddpg-rl", None), 1000)
  
  ddpg(
    lambda : a,
    ac_kwargs=dict(hidden_sizes=[args.hidden_layer_sizes]*args.num_layers),
    gamma=args.gamma,
    seed=args.seed,
    save_folder=args.save_folder,
    num_train_episodes=args.num_train_episodes,
  )
