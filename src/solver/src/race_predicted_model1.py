#!/usr/bin/env python
import roslib; roslib.load_manifest('uml_race')
import math
import rospy
import numpy as np
import math
import csv
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from keras.models import load_model
class RaceSolver(object):
    def __init__(self, model):
        rospy.loginfo("Initialising solver node..")
        self.laserscan_sub = rospy.Subscriber('/robot/base_scan', LaserScan, self.laser_cb, queue_size=10)
        self.velocity_pub = rospy.Publisher('/robot/cmd_vel', Twist, latch=True, queue_size=10)
        self.model = model
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
        y = model.predict(X)
        y[0][0] = y[0][0]*5
        y[0][1] = y[0][1]*45*math.pi
        if(y[0][0] > 5):
            y[0][0] = 5
        if(y[0][1] < 0.1):
            y[0][1] = 0
        velocity.linear.x = y[0][0]
        velocity.angular.z = y[0][1]
        print "Input ->" + str(a) + " " + str(b) + " " + str(c) + " " + str(d) + " " + str(e) + " "
        print "speed: " + str(y[0][0]) + " angular_velocity: " + str(y[0][1])
        self.velocity_pub.publish(velocity)

    def run(self):
        r = rospy.Rate(2)
        while not rospy.is_shutdown():
            self.do_work()
            r.sleep()
        exit()


model = load_model('uml_race_pred.h5')
rospy.init_node('uml_solver')
solver = RaceSolver(model)
solver.run()
