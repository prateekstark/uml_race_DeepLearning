import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import numpy as np
def finish_callback(msg):
    print(msg)

def laser_cb(data):
	print(np.array((list(data.ranges))).shape)
rospy.init_node('subscriber')
finish_pub = rospy.Subscriber('/robot/finish', String, finish_callback)
laserscan_sub = rospy.Subscriber('/robot/base_scan', LaserScan, laser_cb, queue_size=1)

rospy.spin()