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

class Node(object):
	def __init__(self, ls, start_time):
		self.children = []
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
		if(np.amin(self.laser_data) < 0.5):
			return True
		else:
			return False

	def get_best_child(self):
		max_num_visits = 0
		best_child = None
		for child in self.children:
			if(child.num_visits > max_num_visits):
				max_num_visits = child.num_visits
				best_child = child		
		return best_child

	def get_ucb(self):
		c = 2
		if(self.num_visits == 0):
			return float('-inf')
		else:
			final_answer = (self.Q / self.num_visits) + c * math.sqrt((math.log(self.parent.num_visits)) / self.num_visits)
		return final_answer

	def fully_expanded(self):
		if(len(self.unvisited_children) == 0):
			return True
		else:
			return False

	def pick_unvisited_children(self):
		index = random.randint(0, len(self.unvisited_children) - 1)
		return self.unvisited_children[index], index

	def update_score(self, result):
		self.Q += result

class MCTS(object):
	def __init__(self, root_ls, r):
		self.root_node = Node(root_ls)
		self.path = []
		self.rate = r

	def predict_move(self):
		start_time = time.time()
		temp_root = self.root_node
		while(time.time() - start_time < 0.1):
			if(temp_root.is_terminal()):
				raise RuntimeError("terminal node detected")
			else:
				node_to_explore = self.traverse(temp_root)
				playout_result = self.simulate_random_playout(node_to_explore)
				self.backpropagate(node_to_explore, playout_result)
		index = temp_root.get_best_child()
		action = np.zeroes(3)
		action[index] = 1
		return action

	def expand_node(self, node, predictor, dt):
		zero_action = np.zeroes(3)
		children_list = []
		for i in range(3):
			temp_action = zero_action
			temp_action[i] = 1
			a = np.append(node.laser_data, temp_action, axis=0)
			temp_ls = predictor.predict(a)
			temp_node = Node(temp_ls, self.time_elapsed + dt)
			temp_node.parent = node
			children_list.append(temp_node)
		node.children = children_list
		node.unvisited_children = children_list

	def traverse(self, node):
		if not node.fully_expanded():
			child_node, index = node.pick_random_univisted_children()
			node.visited_children.append(child_node)
			del node.unvisited_children[index]
		while node.fully_expanded():
			node = best_node(node)
		return node

	def rollout(self, rollout_network, node):
		return(rollout_network.predict(node.laser_data))
    
	def backpropagate(self, node, result):
		if(self.is_root(node)):
			return
		node.update_score(result)  
		self.backpropagate(node.parent)

	def is_root(self, node):
		if(np.array_equal(self.root_node.laser_data, node.laser_data)):
			return True
		return False

