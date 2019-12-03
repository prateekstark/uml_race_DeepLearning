#!/usr/bin/env python
import roslib; roslib.load_manifest('uml_race')
import rospy
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from math import sqrt
import os
import roslaunch
import time
def toS(t):
    return float(t.secs)+float(t.nsecs) / 1000000000.0

def quit(reason):
    print(reason)
    rospy.sleep(1.0)
    rospy.signal_shutdown(reason)

def distance(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    return sqrt(dx*dx + dy*dy)

class Referee(object):
    def __init__(self):
        self.max_speed = 5.0
        self.goal_x = -25.0
        self.goal_y = -14.0
        self.goal_e =   2.0
        self.start_time = None
        self.isError = False
        self.isFinish = False

    def got_cmd_vel(self, msg):
        if(msg.linear.y > 0 or msg.linear.z > 0):
            rospy.Publisher('/robot/error', String, queue_size=10, latch=True).publish('1')
            self.isError = True
            quit("Error : Move in bad direction")
        if(msg.linear.x > self.max_speed):
            rospy.Publisher('/robot/error', String, queue_size=10, latch=True).publish('1')
            self.isError = True
            quit("Error : speed limit exceeded")
        if(self.start_time == None and msg.linear.x != 0):
            self.start_time = rospy.Time.now()
            print("Start moving at %s" % toS(self.start_time))

    def got_odom(self, msg):
        d = distance(msg.pose.pose.position.x, msg.pose.pose.position.y, self.goal_x, self.goal_y)
        if self.start_time != None and d < self.goal_e:
            duration = rospy.Time.now() - self.start_time
            rospy.Publisher('/robot/finish', String, queue_size=10, latch=True).publish('1')
            self.isFinish = True
            quit("Finished in %fs" % toS(duration))

    def detect_collision(self, msg):
        laser_data = msg.ranges
        if(min(laser_data) < 0.5):
            rospy.Publisher('/robot/error', String, queue_size=10, latch=True).publish('1')
            self.isError = True
            quit("Error : Collision Detected")

    def main(self):
        rospy.init_node('referee')
        rospy.Publisher('/robot/error', String, queue_size=10, latch=True).publish('0')
        rospy.Publisher('/robot/finish', String, queue_size=10, latch=True).publish('0')
        self.isFinish = False
        self.isError = False 
        while not rospy.is_shutdown():
            while not (self.isError or self.isFinish):
                rospy.Subscriber('/robot/base_scan', LaserScan, self.detect_collision, queue_size=10)
                rospy.Subscriber('cmd_vel', Twist, self.got_cmd_vel)
                rospy.Subscriber('base_pose_ground_truth', Odometry, self.got_odom)
                if (self.isError or self.isFinish):
                    break
ref = Referee()
ref.main()