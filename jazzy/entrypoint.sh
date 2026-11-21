#!/usr/bin/env bash
set -e

# ROS 2 environment
source /opt/ros/jazzy/setup.bash

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export TURTLEBOT3_MODEL=${TURTLEBOT3_MODEL:-burger}

echo "[entrypoint] ROS 2 Jazzy ready"
exec "$@"
