cmake_minimum_required(VERSION 3.5)
project(state_web)

find_package(catkin REQUIRED COMPONENTS
  auv_control
  cv_bridge
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
  tf
)

catkin_package(
  CATKIN_DEPENDS
    auv_control
    cv_bridge
    geometry_msgs
    rospy
    sensor_msgs
    std_msgs
    tf
)

catkin_install_python(PROGRAMS
  scripts/state_web_core.py
  scripts/state_web_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch www
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES README.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_state_web_core.py)
endif()
