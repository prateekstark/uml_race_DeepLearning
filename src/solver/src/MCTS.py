#!/usr/bin/env python
import rospy
import math
from random import randint
import random
import numpy as np
import pandas as pd
import time
import os
import pickle
from keras.utils import to_categorical

class Node(object):
	def __init__(self, ls, start_time):
		self.parent = None
		self.laser_data = ls
		self.time_elapsed = start_time
		self.Q = 0
		self.num_visits = 0
		self.unvisited_children = []
		self.visited_children = []

	def get_parent(self):
		return self.parent

	def is_terminal(self):
		if(np.amin(self.laser_data) < -0.8):
			return True
		else:
			return False

	def best_child(self):
		max_ucb = float('-inf')
		index = 0
		best_child = None
		for i in range(len(self.visited_children)):
			child = self.visited_children[i]
			child_ucb = child.get_ucb()
			if(child_ucb >= max_ucb):
				max_ucb = child_ucb
				best_child = child
				index = i

		return best_child, index

	def final_best_child(self):
		max_visit = 0
		index = 0
		best = None
		for i in range(len(self.visited_children)):
			child = self.visited_children[i]
			child_visit = child.num_visits
			if(child_visit >= max_visit):
				max_visit = child_visit
				best = child
				index = i
		return best, index

	def get_ucb(self):
		c = 0.2
		if(self.num_visits == 0):
			return float('inf')

		else:
			final_answer = (self.Q/(self.num_visits + 0.0)) + c*math.sqrt((math.log(self.parent.num_visits))/(self.num_visits+0.0))
		return final_answer

	def fully_expanded(self):
		if(len(self.unvisited_children) == 0 and len(self.visited_children) == 3):
			return True
		else:
			return False

	def pick_unvisited_children(self):
		index = random.randint(0, len(self.unvisited_children) - 1)
		return self.unvisited_children[index], index

	def update_score(self, result):
		self.Q += result
	
	def is_no_children(self):
		if(len(self.unvisited_children) == 0 and len(self.visited_children) == 0):
			return True
		return False

class MCTS(object):
	def __init__(self, root_ls, r, predictor, root_time, rollout):
		self.root_node = Node(root_ls, root_time)
		self.dt = r
		self.predictor = predictor
		self.root_time = root_time
		self.num_rollout = 0
		self.gamma = 0.9
		self.rollout_NN = rollout
		self.correction_list = []

	def predict_move(self):
		start_time = time.time()
		temp_root = self.root_node
		while(time.time() - start_time < 0.1):
			node_to_explore = self.traverse(temp_root)
			playout_result = self.playout(node_to_explore)
			self.backpropagate(node_to_explore, playout_result)
		if(len(temp_root.visited_children) == 3):
			print(str(temp_root.visited_children[0].num_visits) + " " + str(temp_root.visited_children[0].get_ucb()))
			print(str(temp_root.visited_children[1].num_visits) + " " + str(temp_root.visited_children[1].get_ucb()))
			print(str(temp_root.visited_children[2].num_visits) + " " + str(temp_root.visited_children[2].get_ucb()))
		_, index = temp_root.best_child()
		action = np.zeros(3)
		action[index] = 1
		return action

	def expand_node(self, node):
		zero_action = np.zeros(3)
		children_list = []
		for i in range(3):
			temp_action = zero_action
			temp_action[i] = 1
			a = np.append(node.laser_data, temp_action, axis=0)
			temp_ls = self.predictor.predict(a.reshape((1, 183)))[0]
			temp_node = Node(temp_ls, node.time_elapsed + self.dt)
			temp_node.parent = node
			children_list.append(temp_node)
		node.unvisited_children = children_list

	def traverse(self, node):
		while(node.fully_expanded()):
			node, _ = node.best_child()
			if(node.is_terminal()):
				return node
		if(node.is_no_children()):
			self.expand_node(node)
		child_node, index = node.pick_unvisited_children()
		print(child_node.time_elapsed - self.root_node.time_elapsed)
		node.visited_children.append(child_node)
		node.unvisited_children.pop(index)
		return child_node

	def lookahead_one_step(self, node):
		random_action = to_categorical(randint(0, 2), num_classes=3)
		input_data = np.append(node.laser_data, random_action, axis=0)
		temp_ls = self.predictor.predict(input_data.reshape((1, 183)))[0]
		temp_node = Node(temp_ls, node.time_elapsed + self.dt)
		temp_node.parent = node
		return temp_node

	def playout(self, node):
		temp_node = node
		for i in range(2):
			if(temp_node.is_terminal()):
				reward = -2*pow(self.gamma, i)
				return reward
			temp_node = self.lookahead_one_step(temp_node)
		return self.rollout(temp_node)
	
	def rollout(self, node):
		self.num_rollout += 1
		# reward = self.rollout_NN.predict(node.laser_data.reshape((1, 180)))
		# self.correction_list.append((node.laser_data, reward))
		reward = np.amin(node.laser_data)
		# print("reward is " + str(reward))
		return reward
    
	def backpropagate(self, node, result):
		node.num_visits += 1
		if(self.is_root(node)):
			return
		node.update_score(result)
		self.backpropagate(node.parent, result)

	def is_root(self, node):
		if(np.array_equal(self.root_node.laser_data, node.laser_data)):
			return True
		return False