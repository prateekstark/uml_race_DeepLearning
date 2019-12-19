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

def next_state():
	with open('memory', 'rb') as fp:
		memory = pickle.load(fp, encoding='latin1')
		print(memory)
next_state()