#!/bin/bash

azure_kinect_command="cd ~/workspace/xwj/kinect_ws && source install/setup.bash && ros2 launch azure_kinect_ros_driver driver.launch.py"
xarm_moveit_command="cd ~/ros2_ws && source install/setup.bash && ros2 launch xarm_moveit_config xarm6_moveit_realmove.launch.py robot_ip:=192.168.1.225 add_gripper:=true"
easy_handeye_command="cd ~/workspace/xwj/handeye_ws && source install/setup.bash && ros2 launch easy_handeye2 handeye_calibrate.launch.py"
aruco_topic_echo_command="ros2 topic echo /aruco_single/pose"

gnome-terminal --tab --title="Azure Kinect" -- bash -c "$azure_kinect_command; bash"
gnome-terminal --tab --title="XArm Moveit" -- bash -c "$xarm_moveit_command; bash"
gnome-terminal --tab --title="Easy Handeye" -- bash -c "$easy_handeye_command; bash"
gnome-terminal --tab --title="Aruco Topic Echo" -- bash -c "$aruco_topic_echo_command; bash"