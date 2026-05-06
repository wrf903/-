#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
source /home/lunatico/workspace1/turtlebot4_ws/install/setup.bash
exec ros2 launch leo_rover_description orchard_sim.launch.py launch_drive:=true
