#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /home/lunatico/workspace1/turtlebot4_ws/install/setup.bash
export PYTHONPATH="/home/lunatico/workspace1/turtlebot4_ws/venv/lib/python3.12/site-packages:$PYTHONPATH"
exec ros2 launch leo_rover_description orchard_sim.launch.py
