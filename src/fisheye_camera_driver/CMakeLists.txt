cmake_minimum_required(VERSION 3.0.2)
project(fisheye_camera_driver)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS cv_bridge rospy sensor_msgs
)

catkin_install_python(PROGRAMS
  scripts/rtsp_fisheye_driver.py
  scripts/web_image_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
