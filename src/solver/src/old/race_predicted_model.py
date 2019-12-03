#!/usr/bin/env python
import roslib; roslib.load_manifest('uml_race')
import math
# from sklearn.externals import joblib
# from sklearn.preprocessing import MinMaxScaler
import rospy
import numpy as np
import math
import csv
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from keras.models import load_model

class RaceSolver(object):
    def __init__(self, model1, model2):
        rospy.loginfo("Initialising solver node..")
        self.laserscan_sub = rospy.Subscriber('/robot/base_scan', LaserScan, self.laser_cb, queue_size=10)
        self.velocity_pub = rospy.Publisher('/robot/cmd_vel', Twist, latch=True, queue_size=10)
        self.model1 = model1
        self.model2 = model2
        # self.scaler = scaler
        self.laser_data = None
        rospy.sleep(1.0)
        rospy.loginfo("all objects created")

    def laser_cb(self, data):
        self.laser_data = data

    def do_work(self):
        velocity = Twist()
        l = len(self.laser_data.ranges)
        a = self.laser_data.ranges[0]
        b = self.laser_data.ranges[l/4]
        c = self.laser_data.ranges[l/2]
        d = self.laser_data.ranges[3*l/4]
        e = self.laser_data.ranges[l-1]
        f = self.laser_data.ranges[l/3]
        g = self.laser_data.ranges[l/6]
        h = self.laser_data.ranges[2*l/3]
        i = self.laser_data.ranges[5*l/6]
        X = [[a, g, b, f, c, h, d, i, e]]
        X = np.array(X)
        # y = model.predict(X)
        pred_vel = model1.predict(X)
        pred_ang_vel = model2.predict(X)
        # print y
        # y = scaler.inverse_transform(y)
        pred_vel = pred_vel*5
        pred_ang_vel = pred_ang_vel*45*math.pi
        # print y.
        if(pred_vel > 5):
            pred_vel = 5
        if(pred_ang_vel < 0.1):
            pred_ang_vel = 0
        if(pred_vel < 0.1):
            pred_vel = 0
        velocity.linear.x = pred_vel
        velocity.angular.z = pred_ang_vel
        print "Input ->" + str(a) + " " + str(b) + " " + str(c) + " " + str(d) + " " + str(e) + " "
        print "speed: " + str(pred_vel) + " angular_velocity: " + str(pred_ang_vel)
        self.velocity_pub.publish(velocity)

    def run(self):
        r = rospy.Rate(100)
        while not rospy.is_shutdown():
            self.do_work()
            r.sleep()
        exit()


model1 = load_model('uml_race_pred_velocity1.h5')
model2 = load_model('uml_race_pred_angular_velocity1.h5')
# scaler = joblib.load('scaler.save')
rospy.init_node('uml_solver')
solver = RaceSolver(model1, model2)
solver.run()
