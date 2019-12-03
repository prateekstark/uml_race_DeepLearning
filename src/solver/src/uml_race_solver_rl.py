#!/usr/bin/env python
# import roslib; roslib.load_manifest('uml_race')
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
# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DQN(object):
	def __init__(self):
		self.reward = 0.0
		self.gamma = 0.9
		self.agent_target = 1
		self.agent_predict = 0
		self.learning_rate = 0.0005
		self.model = self.network('weights.h5')
		self.epsilon = 0
		self.memory = []
		self.counter_plot = []
		try:
			with open('counter_plot', 'rb') as fp:
				counter_plot = pickle.load(fp)
		except:
			self.counter_plot = []
		try:
			with open('memory', 'rb') as fp:
				memory = pickle.load(fp)
		except:
			self.memory = []

	def set_reward(self, new_state, collision, finish, curr_time):
		self.reward = 0
		if collision:
			self.reward = -10
		elif finish:
			self.reward = 100
		else:
			self.reward = -10*abs(new_state[0] - new_state[179])
		return self.reward

	def network(self, weights=None):
		model = Sequential()
		model.add(Dense(units=32, activation='relu', kernel_initializer='uniform', input_dim=180))
		# model.add(Dropout(0.15))
		model.add(Dense(units=32, activation='relu', kernel_initializer='uniform'))
		# model.add(Dropout(0.15))
		model.add(Dense(units=16, activation='relu', kernel_initializer='uniform'))
		# model.add(Dropout(0.15))
		model.add(Dense(units=3, activation='softmax', kernel_initializer='uniform'))
		opt = Adam(self.learning_rate)
		model.compile(loss='mse', optimizer=opt)
		if os.path.isfile(weights):
			model.load_weights(weights)
		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def replay_new(self, memory):
		if len(memory) > 1000:
			minibatch = random.sample(memory, 1000)
		else:
			minibatch = memory
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				prediction = self.model.predict(next_state.reshape((1, 180)))
				target = reward + self.gamma*np.amax(prediction[0])
			target_f = self.model.predict(np.array(state).reshape((1, 180)))
			target_f[0][np.argmax(action)] = target #Possile Error, confused!
			self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
			
	def train_short_memory(self, state, action, reward, next_state, done):
		target = reward
		if not done:
			prediction = self.model.predict(next_state.reshape((1, 180)))
			target = reward + self.gamma * np.amax(prediction[0])
		target_f = self.model.predict(state.reshape((1, 180)))
		target_f[0][np.argmax(action)] = target
		self.model.fit(state.reshape((1, 180)), target_f, epochs=1, verbose=0)

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
		return np.array(list(self.laser_data.ranges))

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
		record = 0
		start_time = time.time()
		while not (self.error or self.finish):
			agent.epsilon = 200 - counter_games
			state_old = self.get_state()
			# if(randint(0, 300)/2 < agent.epsilon):
			# 	# final_move = to_categorical(randint(0, 2), num_classes=3)
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
			if(randint(0, 150) < agent.epsilon):
				final_move = to_categorical(randint(0, 2), num_classes=3)
				print("I still come here!")
			else:
				prediction = agent.model.predict(state_old.reshape((1, 180)))
				print(prediction)
				final_move = to_categorical(np.argmax(prediction[0]), num_classes = 3)
			self.do_move(final_move)
			state_new = self.get_state()
			reward = agent.set_reward(state_new, self.error, self.finish, (time.time()-start_time))
			agent.train_short_memory(state_old, final_move, reward, state_new, self.error)
			agent.remember(state_old, final_move, reward, state_new, self.error)
			if self.finish:
				game_score = (time.time() - start_time) + 100000/(time.time() - start_time)
			if self.error:
				game_score = (time.time() - start_time) - 30
			else:
				game_score = (time.time() - start_time)
			record = max(game_score, record)
		agent.replay_new(agent.memory)
		print('Game', counter_games, '    Score:', game_score)
		agent.counter_plot.append(game_score)
		with open('counter_plot', 'wb') as fp:
			pickle.dump(agent.counter_plot, fp)
		with open('memory', 'wb') as fp:
			pickle.dump(agent.memory, fp)

		counter_games += 1
		f = open('game_number.txt','w')
		f.write('{}'.format(counter_games))
		f.close()
		agent.model.save_weights('weights.h5')

rospy.init_node('uml_race_solver_rl')
solver = RaceSolver()
solver.run()