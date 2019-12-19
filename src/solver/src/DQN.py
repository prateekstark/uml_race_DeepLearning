#!/usr/bin/env python
import math
from utils import *
from random import randint
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DQN(object):
	def __init__(self):
		self.reward = 0.0
		self.gamma = 0.9
		self.learning_rate = 0.0005
		self.model = self.NN('temp_files/weights.h5')
		self.epsilon = 0
		self.memory = []
		self.counter_plot = []
		self.positive_reward_history = []
		try:
			with open('temp_files/counter_plot', 'rb') as fp:
				self.counter_plot = pickle.load(fp)
		except:
			self.counter_plot = []
		try:
			with open('temp_files/memory', 'rb') as fp:
				self.memory = pickle.load(fp)
		except:
			print("The memory is not loaded properly...")
			self.memory = []

	def get_reward(self, new_state, collision, finish, curr_time):
		self.reward = 0
		if finish:
			self.reward = 1
		elif collision:
			self.reward = -1
		elif((int(curr_time)%2 == 0) and (int(curr_time) > 0)):
			if(int(curr_time) not in self.positive_reward_history):
				self.reward = 1
				self.positive_reward_history.append(int(curr_time))
		return self.reward

	def NN(self, weights=None):
		model = Sequential()
		model.add(Dense(units=10, activation='relu', input_dim=180))
		model.add(Dense(units=10, activation='relu'))
		model.add(Dense(units=10, activation='relu'))
		model.add(Dense(units=10, activation='relu'))
		model.add(Dense(units=10, activation='relu'))
		model.add(Dense(units=3, activation='linear'))
		opt = Adam()
		model.compile(loss='mse', optimizer=opt)
		if os.path.isfile(weights):
			model.load_weights(weights)
		return model

	def write_memory(self, state, action, reward, next_state, error, finish):
		self.memory.append((state, action, reward, next_state, error, finish))
		if(len(self.memory) > 100000):
			coin_toss = randint(0, 100000)
			del self.memory[coin_toss]

	def experience_replay(self, memory):
		batch_size = 32
		if len(memory) < 300:
			if len(memory) > 100:
				minibatch = random.sample(memory, batch_size)
			else:
				return
		else:
			batch_size = int(0.12*(len(self.memory)))
			# batch_size = 256
			minibatch = random.sample(memory, batch_size)
		X = []
		y = []
		for state, action, reward, next_state, error, finish in minibatch:
			target = reward
			target_f = self.model.predict(state.reshape((1, 180)))
			if not (error or finish):
				prediction = self.model.predict(next_state.reshape((1, 180)))
				target = reward + self.gamma * np.amax(prediction[0])
			target_f[0][np.argmax(action)] = target
			X.append(state)
			y.append(target_f[0])
		X = np.array(X)
		y = np.array(y)
		self.model.fit(X.reshape((batch_size, 180)), y, epochs=1, verbose=0)
			
	def SGD_fit(self, state, action, reward, next_state, error, finish):
		target = reward
		target_f = self.model.predict(state.reshape((1, 180)))
		if not (error or finish):
			prediction = self.model.predict(next_state.reshape((1, 180)))
			target = reward + self.gamma * np.amax(prediction[0])
		target_f[0][np.argmax(action)] = target
		self.model.fit(state.reshape((1, 180)), target_f, epochs=1, verbose=0)
