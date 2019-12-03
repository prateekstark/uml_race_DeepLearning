import roslaunch
import rospy
from std_msgs.msg import String
import time
isFinish = False
isError = False

class StageController(object):
	def __init__(self):
		rospy.init_node('stage_controller', anonymous=True)
		self.isFinish = False
		self.isError = False
		
	def error_cb(self, data):
		if('1' in str(data)):
			self.isError = True

	def finish_cb(self, data):
		if('1' in str(data)):
			self.isFinish = True

	def run(self):
		while not rospy.is_shutdown():
			print("It reached here!")
			self.isFinish = False
			self.isError = False
			uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
			roslaunch.configure_logging(uuid)
			launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/stark/race_ws/src/uml_race/launch/racetrack.launch"])
			launch.start()
			while not (self.isFinish or self.isError):
				rospy.Subscriber('/robot/error', String, self.error_cb, queue_size=10)
				rospy.Subscriber('/robot/finish', String, self.finish_cb, queue_size=10)		
			print('Something Happened')
			print("Error Message: " + str(self,isError))
			print("Finish Message: " + str(self.isFinish))
			launch.shutdown()
			time.sleep(2.0)

controller = StageController()
controller.run()