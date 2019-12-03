#!/usr/bin/env python
# import roslib; roslib.load_manifest('uml_race')
import rospy
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class RaceSolver(object):
	def __init__(self):
		rospy.loginfo("Initialising solver node..")
		self.laserscan_sub = rospy.Subscriber('/robot/base_scan', LaserScan, self.laser_cb, queue_size=1) #xxx
		self.velocity_pub = rospy.Publisher('/robot/cmd_vel', Twist, queue_size=1, latch=True)
		self.laser_data= None
		rospy.sleep(1.0)
		rospy.loginfo("all objects created...")

	def laser_cb(self, data):
		self.laser_data = data


	def do_work(self):
		velocity = Twist()

		l = len(self.laser_data.ranges)
		a = self.laser_data.ranges[0]
		b = self.laser_data.ranges[60]
		c = self.laser_data.ranges[90]
		d = self.laser_data.ranges[120]
		e = self.laser_data.ranges[179]
		m = max(a,c,e,b,d)
		if(m==a):
			velocity.angular.z = -180*math.pi 								#xxx
			velocity.linear.x = 5
		if(m==b):
			velocity.angular.z = -180*math.pi								#xxx
			velocity.linear.x = 5
		if(m==c):
			velocity.linear.x = 5
		if(m==d):
			velocity.angular.z = 180*math.pi
			velocity.linear.x = 5
		if(m==e):
			velocity.angular.z = 180*math.pi
			velocity.linear.x = 5

		self.velocity_pub.publish(velocity)

	def run(self, epochs):
		count = 0
		# while(count < epochs):
		r = rospy.Rate(60)
		while not rospy.is_shutdown():
			self.do_work()
				# r.sleep()
			# rospy.wait(5.0)
			# count += 1
		# return None
#if __name__ =='__main__':
rospy.init_node('uml_solver')
solver = RaceSolver()
solver.run(3)
