#!/usr/bin/env python
import rospy
import math
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
import tensorflow as tf
from keras.regularizers import l2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def normalize(v):
	mean_val = np.mean(v)
	return (v - mean_val)/5.0

def get_learning_rate(num_games):
	if(num_games < 1000.0):
		return 0.01*(math.exp(-3*num_games/1000.0))
	return 0.0005

def get_agent_epsilon(num_games):
	if(num_games < 400):
		return math.exp(-3*num_games/400.0)
	return 0.04

def renormalize(v):
	sum = np.sum(v[0])
	v[0] = v[0]/(sum+0.0)
	return v

class DQN(object):
	def __init__(self):
		self.reward = 0.0
		self.gamma = 0.9
		self.learning_rate = 0.0005
		self.model = self.NN('weights.h5')
		self.epsilon = 0
		self.memory = []
		self.counter_plot = []
		self.positive_reward_history = []
		try:
			with open('counter_plot', 'rb') as fp:
				self.counter_plot = pickle.load(fp)
		except:
			self.counter_plot = []
		try:
			with open('memory', 'rb') as fp:
				self.memory = pickle.load(fp)
		except:
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
		model.add(Dense(units=10, activation='relu', input_dim=5))
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
		if(len(self.memory) > 2000):
			self.memory = self.memory[1:]

	def experience_replay(self, memory):
		batch_size = 32
		if len(memory) < 300:
			if len(memory) > 100:
				minibatch = random.sample(memory, batch_size)
			else:
				return
		else:
			batch_size = 256
			minibatch = random.sample(memory, batch_size)
		X = []
		y = []
		for state, action, reward, next_state, error, finish in minibatch:
			target = reward
			target_f = self.model.predict(state.reshape((1, 5)))
			if not (error or finish):
				prediction = self.model.predict(next_state.reshape((1, 5)))
				target = reward + self.gamma * np.amax(prediction[0])
			target_f[0][np.argmax(action)] = target
			X.append(state)
			y.append(target_f[0])
		X = np.array(X)
		y = np.array(y)
		self.model.fit(X.reshape((batch_size, 5)), y, epochs=1, verbose=0)
			
	def train_short_memory(self, state, action, reward, next_state, error, finish):
		target = reward
		target_f = self.model.predict(state.reshape((1, 5)))
		if not (error or finish):
			prediction = self.model.predict(next_state.reshape((1, 5)))
			target = reward + self.gamma * np.amax(prediction[0])
		target_f[0][np.argmax(action)] = target
		self.model.fit(state.reshape((1, 5)), target_f, epochs=1, verbose=0)

class RaceSolver(object):
	def __init__(self):
		rospy.loginfo("Initialising solver node..")
		self.counter_plot = []
		self.laser_data = None
		self.finish = False
		self.error = False
		self.game_score = 0.0
		self.laserscan_sub = rospy.Subscriber('/robot/base_scan', LaserScan, self.laser_cb, queue_size=10)
		self.velocity_pub = rospy.Publisher('/robot/cmd_vel', Twist, queue_size=10, latch=True)
		self.finish_sub = rospy.Subscriber('/robot/finish', String, self.finish_cb, queue_size=10)
		self.error_sub = rospy.Subscriber('/robot/error', String, self.error_cb, queue_size=10)
		rospy.sleep(1.0)
		rospy.loginfo("all objects created...")
		
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
		state = list(self.laser_data.ranges)
		b = []
		b.append(state[0])
		b.append(state[60])
		b.append(state[90])
		b.append(state[120])
		b.append(state[179])
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

	def run(self):
		try:
			f = open('game_number.txt', 'r')
			counter_games = int(f.readline())
			f.close()
		except:
			counter_games = 0
		agent = DQN()
		agent.learning_rate = get_learning_rate(counter_games)
		agent.epsilon = get_agent_epsilon(counter_games)
		try:
			f = open('record.txt', 'r')
			record = int(f.readline())
			f.close()
		except:
			record = 0
		start_time = time.time()
		alpha = 0.01
		beta_dash = 1 - alpha
		beta = 200 - counter_games
		rate = rospy.Rate(10)
		print("The current learning rate is: " + str(agent.learning_rate))
		print("The current epsilon value is: " + str(agent.epsilon))
		print("The current memory length is: " + str(len(agent.memory)))
		while not (self.error or self.finish):
			# print("a")
			state_old = self.get_state()
			coin_toss = randint(0, 100)/100.0
			if(coin_toss < agent.epsilon):
				final_move = to_categorical(randint(0, 2), num_classes=3)
				# else:
				# 	a = state_old[0]
				# 	b = state_old[60]
				# 	c = state_old[90]
				# 	d = state_old[120]
				# 	e = state_old[179]
				# 	m = max(a,c,e,b,d)
				# 	if(m==a):
				# 		pred = 0
				# 	if(m==b):
				# 		pred = 0
				# 	if(m==c):
				# 		pred = 2
				# 	if(m==d):
				# 		pred = 1
				# 	if(m==e):
				# 		pred = 1
					
				# 	if pred == 0:
				# 		final_move = [1, 0, 0]
				# 	elif pred == 1:
				# 		final_move = [0, 1, 0]
				# 	else:
				# 		final_move = [0, 0, 1]
			else:
				prediction = agent.model.predict(state_old.reshape((1, 5)))
				print(prediction)
				final_move = to_categorical(np.argmax(prediction[0]), num_classes = 3)
			self.do_move(final_move)
			rate.sleep()
			state_new = self.get_state()
			reward = agent.get_reward(state_new, self.error, self.finish, (time.time()-start_time))
			agent.train_short_memory(state_old, final_move, reward, state_new, self.error, self.finish)
			agent.write_memory(state_old, final_move, reward, state_new, self.error, self.finish)
		game_score = time.time() - start_time
		record = max(game_score, record)
		agent.experience_replay(agent.memory)
		print('Game', counter_games, '    Time Elapsed:', game_score)
		agent.counter_plot.append(game_score)
		# print(agent.counter_plot)
		with open('counter_plot', 'wb') as fp:
			pickle.dump(agent.counter_plot, fp)
		with open('memory', 'wb') as fp:
			pickle.dump(agent.memory, fp)
		counter_games += 1
		f = open('game_number.txt','w')
		f.write('{}'.format(counter_games))
		f.close()
		f = open('record.txt','w')
		f.write('{}'.format(int(record)))
		f.close()
		agent.model.save_weights('weights.h5')
rospy.init_node('uml_race_solver_rl')
solver = RaceSolver()
solver.run()