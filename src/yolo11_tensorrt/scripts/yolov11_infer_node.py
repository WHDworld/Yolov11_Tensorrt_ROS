#!/usr/bin/env python3

import rospy
import numpy as np
from yolov11_infer import *


def main():
	rospy.init_node("yolov11_infer_node")
	YoloDetector()
	rospy.spin()

if __name__=="__main__":
	main()
	
