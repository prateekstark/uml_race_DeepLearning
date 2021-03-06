#!/usr/bin/env python
import rospy
import math
from utils import *
from MCTS import *
from DQN import DQN
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
		self.model = self.state_predictor_NN('temp_files_MCTS/state_predictor.h5')
		self.rollout_model = self.rollout_NN('temp_files_MCTS/rollout_predictor.h5')
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
		
		try:
			with open('temp_files_MCTS/counter_plot.pickle', 'rb') as fp:
				self.counter_plot = pickle.load(fp)
		except:
			self.counter_plot = []
		
		try:
			f = open('temp_files_MCTS/game_number.txt', 'r')
			self.game_number = int(f.readline())
			f.close()
		except:
			self.game_number = 0

		rospy.sleep(1.0)
		rospy.loginfo("all objects created...")
		
	def state_predictor_NN(self, weights=None):
		model = Sequential()
		model.add(Dense(180, input_dim=183, activation='relu'))
		model.add(Dense(120, activation='relu'))
		model.add(Dense(120, activation='relu'))
		model.add(Dense(120, activation='relu'))
		model.add(Dense(120, activation='relu'))
		model.add(Dense(120, activation='relu'))
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
		model.add(Dense(units=3, activation='softmax'))
		model.compile(loss='mse', optimizer='adam')
		if os.path.isfile(weights):
			model.load_weights(weights)
		return model

	def fit_save_NN(self, X, y):
		self.model.fit(X, y, epochs=1, verbose=1, batch_size=1)
		self.model.save_weights('temp_files_MCTS/state_predictor.h5')
	
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
		start_time = time.time()
		rate = rospy.Rate(self.rate)
		state_training_X = []
		state_training_y = []
		dt = 1.0/self.rate
		while not (self.error or self.finish):
			state_old = self.get_state()
			mcts = MCTS(state_old, dt, model, (time.time() - start_time), self.rollout_model)
			prediction = mcts.predict_move()
			self.do_move(prediction)
			print(prediction)
			print("Number of rollout in one MCTS: " + str(mcts.num_rollout))
			state_training_X.append(np.append(state_old, prediction, axis=0))
			rate.sleep()
			state_new = self.get_state()
			state_training_y.append(state_new)
			print(min(state_new))
			# print(state_new - state_old)
		game_score = time.time() - start_time
		print('Game', self.game_number, '    Time Elapsed:', game_score)
		self.counter_plot.append(game_score)
		with open('temp_files_MCTS/counter_plot.pickle', 'wb') as fp:
			pickle.dump(self.counter_plot, fp)
		self.game_number += 1
		f = open('temp_files_MCTS/game_number.txt','w')
		f.write('{}'.format(self.game_number))
		f.close()
		state_training_X = np.array(state_training_X)
		state_training_y = np.array(state_training_y)
		self.fit_save_NN(state_training_X, state_training_y)
rospy.init_node('uml_race_solver_MCTS')
solver = RaceSolver()
solver.run()

