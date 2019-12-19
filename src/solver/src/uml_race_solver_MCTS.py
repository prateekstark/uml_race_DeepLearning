#!/usr/bin/env python
import rospy
import math
from utils import *
from random import randint
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import random
import numpy as np
from std_msgs.msg import String
import time
import os
import rosgraph
import pickle
from MCTS import *

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
			with open('temp_files/counter_plot', 'rb') as fp:
				self.counter_plot = pickle.load(fp)
		except:
			self.counter_plot = []
		try:
			f = open('temp_files/game_number.txt', 'r')
			counter_games = int(f.readline())
			f.close()
		except:
			counter_games = 0
		try:
			f = open('temp_files/record.txt', 'r')
			record = int(f.readline())
			f.close()
		except:
			record = 0
		start_time = time.time()
		beta = 200 - counter_games
		rate = rospy.Rate(10)
		while not (self.error or self.finish):
			state_old = self.get_state()
			node = MCTS(state_old, 0.1)
			prediction = node.predict_move()
			print(prediction)
			final_move = to_categorical(np.argmax(prediction[0]), num_classes = 3)
			self.do_move(final_move)
			rate.sleep()
			state_new = self.get_state()
		game_score = time.time() - start_time
		record = max(game_score, record)
		print('Game', counter_games, '    Time Elapsed:', game_score)
		with open('temp_files/counter_plot', 'wb') as fp:
			pickle.dump(counter_plot, fp)
		counter_games += 1
		f = open('temp_files/game_number.txt','w')
		f.write('{}'.format(counter_games))
		f.close()
		f = open('temp_files/record.txt','w')
		f.write('{}'.format(int(record)))
		f.close()
rospy.init_node('uml_race_solver_MCTS')
solver = RaceSolver()
solver.run()


