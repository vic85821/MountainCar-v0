import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from brain import QLearningTable
from brain import SarsaTable
from brain import SarsaLambdaTable
from brain import DeepQNetwork

if __name__ == "__main__":
	env = gym.make('MountainCar-v0')
	env = env.unwrapped
	#RL = QLearningTable(range(0, env.action_space.n))
	#RL = SarsaLambdaTable(range(0, env.action_space.n))
	RL = DeepQNetwork(env.action_space.n, 2,
					  lr=0.001,
					  reward_decay=0.7,
					  e_greedy=0.95,
					  replace_target_iter=300,
					  memory_size=3000,
					  e_greedy_increment=0.0001
					  )
	steps = []

	for i_episode in range(1000):
		# Get the observation from env
		observation = env.reset()
		time = 0
			
		while True:

			# Fresh env
			env.render()

			# Select action
			action = RL.choose_action(observation)

			# RL take the action and get the next observation from updated env
			observation_, reward, done, info = env.step(action)

			#action_ = RL.choose_action(observation_)
			reward = abs(observation_[0] - (-0.5)) + (observation_[0] > 0.55) * 10
			RL.store_transition(observation, action, reward, observation_)

			# use Deep Q Network
			if (time > 500) and (time % 5 == 0):
				RL.learn()

			# use QLearning or Sarsa to learn
			'''
			# learn from present observation
			if done == True:
				RL.learn(observation, action, reward, 'terminal', action_)
			else:
				RL.learn(observation, action, reward, observation_, action_)
			'''

			# Renew the observation
			observation = observation_

			# Get the flag
			if done:
				print("Episode {} finished after {} timesteps".format(i_episode, time))
				steps.append(time)
				break
			
			time = time + 1
	
	env.close()
	plt.hist(steps)
