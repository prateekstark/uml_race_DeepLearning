#!/usr/bin/env python
import roslib; roslib.load_manifest('uml_race')
import rospy
import math
import csv
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
class RaceSolver(object):
    def __init__(self, outFileWriter):
        rospy.loginfo("Initialising solver node..")
        self.laserscan_sub = rospy.Subscriber('/robot/base_scan', LaserScan, self.laser_cb, queue_size=1)
        self.velocity_pub = rospy.Publisher('/robot/cmd_vel', Twist, latch=True, queue_size=1)
        self.laser_data = None
        self.outFileWriter = outFileWriter
        rospy.sleep(1.0)
        rospy.loginfo("all objects created")

    def laser_cb(self, data):
        self.laser_data = data

    def do_work(self, outFileWriter):
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
        m = max(a,c,e,b,d)
        if(m==a):
            velocity.angular.z = -180*math.pi/2
            velocity.linear.x = 1
        if(m==b):
            velocity.angular.z = -180*math.pi/4
            velocity.linear.x = 1
        if(m==c):
            velocity.linear.x = 5
        if(m==d):
            velocity.angular.z = 180*math.pi/4
            velocity.linear.x = 1
        if(m==e):
            velocity.angular.z = 180*math.pi/2
            velocity.linear.x = 1
        self.outFileWriter.writerow([a, g, b, f, c, h, d, i, e, velocity.linear.x, velocity.angular.z])
        self.velocity_pub.publish(velocity)

    def run(self):
        r = rospy.Rate(60)
        while not rospy.is_shutdown():
            self.do_work(outFileWriter)
            r.sleep()
        exit()


outFile = open('output_file.csv', 'w+')
outFileWriter = csv.writer(outFile)
outFileWriter.writerow(['0 degree', '30 degree', '45 degree', '60 degree', '90 degree', '120 degree','135 degree', '150 degree', '180 degree', 'velocity', 'angular_velocity'])
rospy.init_node('uml_solver')
solver = RaceSolver(outFileWriter)
solver.run()
