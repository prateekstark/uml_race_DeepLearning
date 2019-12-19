import math
import numpy as np

def normalize(v):
	mean_val = 2.5
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
