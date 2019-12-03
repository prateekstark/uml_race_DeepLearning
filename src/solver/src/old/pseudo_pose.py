import rospy
from geometry_msgs.msg import Pose

rospy.init_node('pseudo_pose')
pose_pub = rospy.Publisher('/robot/base_pose_ground_truth', Pose, queue_size=1, latch=True)
rospy.sleep(1.0)
count = 0
while(count < 100):
	init_pose = Pose()
	init_pose.position.x = -24.6
	init_pose.position.y = -7.8
	pose_pub.publish(init_pose)
	count += 1
