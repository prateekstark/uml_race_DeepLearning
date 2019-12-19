#!/usr/bin/env python
import rospy
import math
from utils import *
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
import tensorflow as tf
from keras.regularizers import l2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class RaceSolver(object):
	def __init__(self):
		rospy.loginfo("Initialising solver node..")
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
		try:
			f = open('temp_files/game_number.txt', 'r')
			counter_games = int(f.readline())
			f.close()
		except:
			counter_games = 0
		agent = DQN()
		agent.learning_rate = get_learning_rate(counter_games)
		agent.epsilon = get_agent_epsilon(counter_games)
		try:
			f = open('temp_files/record.txt', 'r')
			record = int(f.readline())
			f.close()
		except:
			record = 0
		start_time = time.time()
		alpha = 0.01
		beta_dash = 1 - alpha
		beta = 200 - counter_games
		rate = rospy.Rate(10)
		next_state_data = []
		print("The current learning rate is: " + str(agent.learning_rate))
		print("The current epsilon value is: " + str(agent.epsilon))
		print("The current memory length is: " + str(len(agent.memory)))
		while not (self.error or self.finish):
			state_old = self.get_state()
			triplet = []
			triplet.append(state_old)
			coin_toss = randint(0, 100)/100.0
			if(coin_toss < agent.epsilon):
				final_move = self.random_move()
			else:
				prediction = agent.model.predict(state_old.reshape((1, 180)))
				print(prediction)
				final_move = to_categorical(np.argmax(prediction[0]), num_classes = 3)
			if(final_move[0] == 1):
				triplet.append(-1)
			elif(final_move[1] == 1):
				triplet.append(1)
			else:
				triplet.append(0)
			self.do_move(final_move)
			time_1 = time.time()
			rate.sleep()
			state_new = self.get_state()
			time_2 = time.time()
			triplet.append(time_2 - time_1)
			triplet.append(state_new)
			reward = agent.get_reward(state_new, self.error, self.finish, (time.time()-start_time))
			agent.SGD_fit(state_old, final_move, reward, state_new, self.error, self.finish)
			agent.write_memory(state_old, final_move, reward, state_new, self.error, self.finish)
			next_state_data.append(triplet)
		game_score = time.time() - start_time
		record = max(game_score, record)
		agent.experience_replay(agent.memory)
		print('Game', counter_games, '    Time Elapsed:', game_score)
		agent.counter_plot.append(game_score)
		with open('temp_files/counter_plot', 'wb') as fp:
			pickle.dump(agent.counter_plot, fp)
		with open('temp_files/memory', 'wb') as fp:
			pickle.dump(agent.memory, fp)
		with open('temp_files/next_state_train.pickle', 'wb') as fp:
			pickle.dump(next_state_data, fp)
		counter_games += 1
		f = open('temp_files/game_number.txt','w')
		f.write('{}'.format(counter_games))
		f.close()
		f = open('temp_files/record.txt','w')
		f.write('{}'.format(int(record)))
		f.close()
		agent.model.save_weights('temp_files/weights.h5')
rospy.init_node('uml_race_solver_DQN')
solver = RaceSolver()
solver.run()