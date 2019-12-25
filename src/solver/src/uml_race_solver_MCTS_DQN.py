#!/usr/bin/env python
import rospy
import math
from utils import *
from MCTS_DQN import *
from random import randint
from keras.utils import to_categorical
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from std_msgs.msg import String
import time
import os
import rosgraph
import pickle
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class RaceSolver(object):
	def __init__(self):
		rospy.loginfo("Initialising solver node..")
		self.model = self.state_predictor_NN('temp_files_MCTS_DQN/state_predictor.h5')
		self.rollout_model = self.rollout_NN('temp_files_MCTS_DQN/rollout_predictor.h5')
		self.laser_data = None
		self.finish = False
		self.error = False
		self.game_score = 0.0
		self.laserscan_sub = rospy.Subscriber('/robot/base_scan', LaserScan, self.laser_cb, queue_size=10)
		self.velocity_pub = rospy.Publisher('/robot/cmd_vel', Twist, queue_size=10, latch=True)
		self.finish_sub = rospy.Subscriber('/robot/finish', String, self.finish_cb, queue_size=10)
		self.error_sub = rospy.Subscriber('/robot/error', String, self.error_cb, queue_size=10)
		self.counter_plot = []
		self.game_number = 0
		self.rate = 10
		self.memory = []
		self.positive_reward_history = []
		self.gamma = 0.9
		
		try:
			with open('temp_files_MCTS_DQN/counter_plot.pickle', 'rb') as fp:
				self.counter_plot = pickle.load(fp)
		except:
			self.counter_plot = []
		
		try:
			f = open('temp_files_MCTS_DQN/game_number.txt', 'r')
			self.game_number = int(f.readline())
			f.close()
		except:
			self.game_number = 0

		try:
			with open('temp_files_MCTS_DQN/memory.pickle', 'rb') as fp:
				self.memory = pickle.load(fp)
		except:
			print("The memory is not loaded properly...")
			self.memory = []

		rospy.sleep(1.0)
		rospy.loginfo("all objects created...")
		
	def state_predictor_NN(self, weights=None):
		model = Sequential()
		model.add(Dense(180, input_dim=183, activation='relu'))
		model.add(Dense(90, activation='relu'))
		model.add(Dense(45, activation='relu'))
		model.add(Dense(10, activation='relu'))
		model.add(Dense(45, activation='relu'))
		model.add(Dense(90, activation='relu'))
		model.add(Dense(180, activation='tanh'))
		model.compile(loss='mse', optimizer='adam', metrics=["mean_squared_error"])
		if os.path.isfile(weights):
			model.load_weights(weights)
		return model

	def rollout_NN(self, weights=None):
		model = Sequential()
		model.add(Dense(units=10, activation='relu', input_dim=180))
		model.add(Dense(units=10, activation='relu'))
		model.add(Dense(units=10, activation='relu'))
		model.add(Dense(units=10, activation='relu'))
		model.add(Dense(units=10, activation='relu'))
		model.add(Dense(units=3, activation='linear'))
		model.compile(loss='mse', optimizer='adam')
		if os.path.isfile(weights):
			model.load_weights(weights)
		return model

	def update_reward(self, curr_time):
		if((int(curr_time)%2 == 0) and (int(curr_time) > 0)):
			if(int(curr_time) not in self.positive_reward_history):
				reward = 1
				self.positive_reward_history.append(int(curr_time))

	def fit_save_NN(self, X, y):
		self.model.fit(X, y, epochs=1, verbose=1)
		self.model.save_weights('temp_files_MCTS_DQN/state_predictor.h5')

	def fit_save_rollout_NN(self, episode_memory):
		for i in range(len(episode_memory)):
			(state, action, reward, next_state, error, finish) = episode_memory[i]
			target = reward
			target_f = self.rollout_model.predict(state.reshape((1, 180)))
			if not (error or finish):
				prediction = self.rollout_model.predict(next_state.laser_data.reshape((1, 180)))
				target = reward + self.gamma * np.amax(prediction[0])
			target_f[0][np.argmax(action)] = target
			self.rollout_model.fit(state.reshape((1, 180)), target_f, epochs=1, verbose=0)

	def experience_replay(self, memory):
		batch_size = 32
		if len(memory) < 300:
			if len(memory) > 100:
				minibatch = random.sample(memory, batch_size)
			else:
				return
		else:
			batch_size = int(0.12*(len(self.memory)))
			minibatch = random.sample(memory, batch_size)
		X = []
		y = []
		for state, action, reward, next_state, error, finish in minibatch:
			target = reward
			target_f = self.rollout_model.predict(state.reshape((1, 180)))
			if not (error or finish):
				prediction = self.rollout_model.predict(next_state.laser_data.reshape((1, 180)))
				target = reward + self.gamma * np.amax(prediction[0])
			target_f[0][np.argmax(action)] = target
			X.append(state)
			y.append(target_f[0])
		X = np.array(X)
		y = np.array(y)
		self.rollout_model.fit(X.reshape((batch_size, 180)), y, epochs=1, verbose=0)

	def write_memory(self, episode_memory):
		for i in range(len(episode_memory)):
			self.memory.append(episode_memory[i])
			if(len(self.memory) > 50000):
				coin_toss = randint(0, 50000)
				del self.memory[coin_toss]
		with open('temp_files_MCTS_DQN/memory.pickle', 'wb') as fp:
			pickle.dump(self.memory, fp)

	def do_move(self, final_move):
		velocity = Twist()
		velocity.linear.x = 5
		if(np.array_equal(final_move, [1, 0, 0])):
			velocity.angular.z = -180*math.pi
		elif(np.array_equal(final_move, [0, 1, 0])):
			velocity.angular.z = 180*math.pi
		elif(np.array_equal(final_move, [0, 0, 1])):
			velocity.angular.z = 0
		self.velocity_pub.publish(velocity)

	def get_state(self):
		b = list(self.laser_data.ranges)
		b = np.array(b)
		b = normalize(b)
		return b

	def laser_cb(self, data):
		self.laser_data = data

	def finish_cb(self, data):
		if('1' in str(data)):
			self.finish = True
		elif('0' in str(data)):
			self.finish = False

	def error_cb(self, data):
		if('1' in str(data)):
			self.error = True
		elif('0' in str(data)):
			self.error = False

	def random_move(self):
		return to_categorical(randint(0, 2), num_classes=3)

	def run(self):
		model = self.model
		rollout = self.rollout_model
		rate = rospy.Rate(self.rate)
		state_training_X = []
		state_training_y = []
		dt = 1.0/self.rate
		episode_memory = []
		start_time = time.time()
		while not (self.error or self.finish):
			state_old = self.get_state()
			mcts = MCTS(state_old, dt, model, (time.time() - start_time), rollout, self.finish, self.positive_reward_history)
			prediction = mcts.predict_move()
			self.do_move(prediction)
			episode_memory = episode_memory + mcts.correction_list
			print(prediction)
			print("Number of rollout in one MCTS: " + str(mcts.num_rollout))
			state_training_X.append(np.append(state_old, prediction, axis=0))
			rate.sleep()
			state_new = self.get_state()
			self.update_reward(time.time() - start_time)
			state_training_y.append(state_new)
		game_score = time.time() - start_time
		print('Game', self.game_number, '    Time Elapsed:', game_score)
		self.counter_plot.append(game_score)
		with open('temp_files_MCTS_DQN/counter_plot.pickle', 'wb') as fp:
			pickle.dump(self.counter_plot, fp)
		self.game_number += 1
		f = open('temp_files_MCTS_DQN/game_number.txt','w')
		f.write('{}'.format(self.game_number))
		f.close()
		state_training_X = np.array(state_training_X)
		state_training_y = np.array(state_training_y)
		self.fit_save_NN(state_training_X, state_training_y)
		self.write_memory(episode_memory)
		self.fit_save_rollout_NN(episode_memory)
		self.experience_replay(self.memory)
		self.rollout_model.save_weights('temp_files_MCTS_DQN/rollout_predictor.h5')

rospy.init_node('uml_race_solver_MCTS_DQN')
solver = RaceSolver()
solver.run()