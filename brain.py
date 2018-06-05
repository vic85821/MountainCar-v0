import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class RL(object):
	
	def __init__(self, actions, lr=0.01, reward_decay=0.9, e_greedy=0.9):
		self.actions = actions
		self.lr = lr
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.qtable = pd.DataFrame(columns=self.actions)

	def choose_action(self, observation):
		observation = self.transform_state(observation)
		self.state_exist(observation)
		
		# epsilon greedy
		if np.random.uniform() < self.epsilon:
			# choose the best choice
			state_action = self.qtable.loc[observation, :]
			state_action = state_action.reindex(np.random.permutation(state_action.index))
			action = state_action.values.argmax()

		else:
			# select random action
			action = np.random.choice(self.actions)

		return action

	def learn(self, *args):
		pass

	def state_exist(self, state):
		if state not in self.qtable.index:
			# new state
			self.qtable = self.qtable.append(
				pd.Series(
					[0]*len(self.actions),
					index=self.qtable.columns,
					name=state
				)
			)
	
	def transform_state(self, observation):
		state = str((round(observation[0], 2), round(observation[1], 2)))
		return state


class QLearningTable(RL):
	
	def __init__(self, actions, lr=0.01, reward_decay=0.9, e_greedy=0.9):
		super(QLearningTable, self).__init__(actions, lr, reward_decay, e_greedy)

	def learn(self, s, a, r, s_):
		s = self.transform_state(s)
		if s_ != 'terminal':
			s_ = self.transform_state(s_)

		# check the next state exists or not
		self.state_exist(s_)

		# predict Q value
		q_predict = self.qtable.loc[s, a]
		
		if s_ != 'terminal':
			q_target = r + self.gamma * self.qtable.loc[s_, :].max()
		else:
			q_target = r

		# update qtable
		self.qtable.loc[s, a] += self.lr * (q_target - q_predict)


class SarsaTable(RL):

	def __init__(self, actions, lr=0.01, reward_decay=0.9, e_greedy=0.9):
		super(SarsaTable, self).__init__(actions, lr, reward_decay, e_greedy)

	def learn(self, s, a, r, s_, a_):
		s = self.transform_state(s)
		if s_ != 'terminal':
			s_ = self.transform_state(s_)

		# check the next state exists or not
		self.state_exist(s_)

		# predict Q value
		q_predict = self.qtable.loc[s, a]
		
		if s_ != 'terminal':
			q_target = r + self.gamma * self.qtable.loc[s_, a_]
		else:
			q_target = r

		# update qtable
		self.qtable.loc[s, a] += self.lr * (q_target - q_predict)


class SarsaLambdaTable(RL):

	def __init__(self, actions, lr=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
		super(SarsaLambdaTable, self).__init__(actions, lr, reward_decay, e_greedy)

		self.lambda_ = trace_decay
		self.eligibility_trace = self.qtable.copy()

	def learn(self, s, a, r, s_, a_):
		s = self.transform_state(s)
		if s_ != 'terminal':
			s_ = self.transform_state(s_)

		# check the next state exists or not
		self.state_exist(s_)

		# predict Q value
		q_predict = self.qtable.loc[s, a]
		
		if s_ != 'terminal':
			q_target = r + self.gamma * self.qtable.loc[s_, a_]
		else:
			q_target = r

		# update qtable
		self.eligibility_trace.loc[s, :] *= 0
		self.eligibility_trace.loc[s, a] = 1

		self.qtable += self.lr * (q_target - q_predict) * self.eligibility_trace
		self.eligibility_trace *= self.gamma * self.lambda_

	def state_exist(self, state):
		if state not in self.qtable.index:
			# new state
			self.qtable = self.qtable.append(
				pd.Series(
					[0]*len(self.actions),
					index=self.qtable.columns,
					name=state
				)
			)
			
			self.eligibility_trace = self.eligibility_trace.append(
				pd.Series(
					[0]*len(self.actions),
					index=self.qtable.columns,
					name=state
				)
			)

class DeepQNetwork:
	def __init__(
		self,
		actions,
		features,
		lr=0.001,
		reward_decay=0.9,
		e_greedy=0.9,
		replace_target_iter=300,
		memory_size=500,
		batch_size=32,
		e_greedy_increment=None,
	):
		self.actions = actions
		self.n_features = features
		self.lr = lr
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

		# total learning step
		self.learn_step_counter = 0

		# initialize zero memory [s, a, r, s_]
		self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

		# consist of [target_net, evaluate_net]
		self.build_net()
		
		t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
		p_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pred_net')

		with tf.variable_scope('soft_replacement'):
			self.target_replace_op = [tf.assign(t, p) for t, p in zip(t_params, p_params)]

		self.sess = tf.Session()

		self.sess.run(tf.global_variables_initializer())
		self.cost_his = []
	
	def build_net(self):
		self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
		self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
		self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
		self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

		w_initializer, b_initializer = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)

		##### Predict Network #####
		with tf.variable_scope('pred_net'):
			pn = tf.layers.dense(self.s, 20, tf.nn.elu, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='pn')
			self.q_eval = tf.layers.dense(pn, self.actions, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='q')

		##### Target Network #####
		with tf.variable_scope('target_net'):
			tn = tf.layers.dense(self.s_, 20, tf.nn.elu, kernel_initializer=w_initializer,	bias_initializer=b_initializer, name='tn')
			self.q_next = tf.layers.dense(tn, self.actions, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='q_')

		# about per 500 iters updates one time
		with tf.variable_scope('q_target'):
			q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
			self.q_target = tf.stop_gradient(q_target)

		# update each iter
		with tf.variable_scope('q_eval'):
			a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
			self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)
		
		# loss function - mse
		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

		# optimizer
		with tf.variable_scope('train'):
			self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

	def store_transition(self, s, a, r, s_):

		# memory is empty
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0

		transition = np.hstack((s, [a, r], s_))
		index = self.memory_counter % self.memory_size
		self.memory[index, :] = transition
		self.memory_counter += 1

	def choose_action(self, observation):
		
		# new batch dimension
		observation = observation[np.newaxis, :]

		if np.random.uniform() < self.epsilon:
			actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
			action = np.argmax(actions_value)
		else:
			action = np.random.randint(0, self.actions)

		return action

	def learn(self):
		
		# check to update target network parameters
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.sess.run(self.target_replace_op)

		if self.memory_counter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
		batch_memory = self.memory[sample_index, :]

		_, cost = self.sess.run(
			[self._train_op, self.loss],
			feed_dict={
				self.s: batch_memory[:, :self.n_features],
				self.a: batch_memory[:, self.n_features],
				self.r: batch_memory[:, self.n_features + 1],
				self.s_: batch_memory[:, -self.n_features:]
		})

		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1
	
