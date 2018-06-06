import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import gym
import tensorflow as tf

from brain import QLearningTable
from brain import SarsaTable
from brain import SarsaLambdaTable
from brain import DeepQNetwork
from brain import PolicyGradient

DISPLAY_REWARD_THRESHOLD = -2000
RENDER = False

if __name__ == "__main__":
	env = gym.make('MountainCar-v0')
	env = env.unwrapped
	env.seed(1)
	err = False

	parser = argparse.ArgumentParser()
	parser.add_argument('--method',
						default='DQN',
						help='input method used to solve problem: Q-learning / Sarsa / SarsaLambda / DQN / Policy Gradient')
	parser.add_argument('--episode',
						default='5000',
						help='input how many episodes to execute')
	args = parser.parse_args()
	
	if args.method == 'Q-learning':
		RL = QLearningTable(range(0, env.action_space.n))
	elif args.method == 'Sarsa':
		RL = SarsaTable(range(0, env.action_space.n))
	elif args.method == 'SarsaLambda':
		RL = SarsaLambdaTable(range(0, env.action_space.n))
	elif args.method == 'DQN':
		RL = DeepQNetwork(env.action_space.n, 2,
						  lr = 0.001,
						  reward_decay = 0.9,
						  e_greedy = 0.9,
						  replace_target_iter = 300,
					  	  memory_size = 3000,
					  	  e_greedy_increment = 0.0001
					  	 )
	elif args.method == 'PolicyGradient':
		RL = PolicyGradient(env.action_space.n, 2,
							lr = 0.001,
							reward_decay = 0.995
						   )
	else:
		print("Error method name, use default method - DQN to execute")
		err = True

	steps = []

	for i_episode in range(int(args.episode)):
		# Get the observation from env
		observation = env.reset()
		
		rewards = 0
		step = 0
			
		while True:
			
			# Fresh env
			env.render()

			# Select action
			action = RL.choose_action(observation)

			# RL take the action and get the next observation from updated env
			observation_, reward, done, info = env.step(action)

			rewards = rewards + reward
			reward = abs(observation_[0] - (-0.5)) + ((observation_[0] - observation[0]) * observation_[1] > 0) * 0.2 + \
			         (observation_[0] > 0.5) * 10
			
			
			if args.method == 'Q-learning':
				if done == True:
					RL.learn(observation, action, reward, 'terminal')
				else:
					RL.learn(observation, action, reward, observation_)

			elif args.method == 'SarsaLambda' or args.method == 'Sarsa':
				action_ = RL.choose_action(observation_)
				if done == True:
					RL.learn(observation, action, reward, 'terminal', action_)
				else:
					RL.learn(observation, action, reward, observation_, action_)

			elif args.method == 'DQN' or err == True:
				RL.store_transition(observation, action, reward, observation_)

				# use Deep Q Network
				if (step > 300) and (step % 5 == 0):
					RL.learn()

			elif args.method == 'PolicyGradient':
				RL.store_transition(observation, action, reward)

			# Renew the observation
			observation = observation_

			# Get the flag
			if done:
				
				if args.method == 'PolicyGradient':
					ep_rs_sum = sum(RL.ep_rs)
					if 'running_reward' not in globals():
						running_reward = ep_rs_sum
					else:
						running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

					if running_reward > DISPLAY_REWARD_THRESHOLD: 
						RENDER = True
					
					vt = RL.learn()

					print("Episode {} : {}".format(i_episode, int(running_reward)))
					print("Episode {} finished with reward: {}".format(i_episode, int(rewards)))

				else:
					print("Episode {} finished with reward: {}".format(i_episode, rewards))
				
				steps.append(step)
				break
			
			step = step + 1
	
	env.close()
	df = pd.DataFrame(steps, columns=["steps"])
	df.to_csv("count.csv", index=False)
