#!/usr/bin/env python3
import csv
import math
import os
import signal
import sys
from datetime import datetime

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_from_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def clamp(value, low, high):
    return max(low, min(high, value))


class ClickToDrive(Node):
    def __init__(self, output_path):
        super().__init__('click_to_drive')
        self.output_path = output_path
        self.pose = None
        self.goal = None
        self.goal_index = 0
        self.arrival_logged = False

        self.goal_tolerance = 0.35
        self.max_linear = 0.24
        self.min_rolling_linear = 0.08
        self.max_angular = 0.35
        self.k_linear = 0.35
        self.k_angular = 0.75
        self.slow_heading = math.radians(75.0)
        self.stop_heading = math.radians(145.0)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.file = open(output_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'event', 'index', 'wall_time', 'sim_sec',
            'goal_x', 'goal_y', 'robot_x', 'robot_y', 'distance'
        ])
        self.file.flush()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(PointStamped, '/clicked_point', self.clicked_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, 10)
        self.create_subscription(Odometry, '/odom_truth', self.odom_callback, 20)
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            f'Click-to-drive ready. Use RViz 2D Goal Pose or Publish Point. '
            f'Logging goals to {output_path}'
        )

    def now_sim(self):
        now = self.get_clock().now()
        return now.nanoseconds * 1e-9

    def write_event(self, event, distance=''):
        rx = self.pose[0] if self.pose else ''
        ry = self.pose[1] if self.pose else ''
        gx = self.goal[0] if self.goal else ''
        gy = self.goal[1] if self.goal else ''
        self.writer.writerow([
            event,
            self.goal_index,
            datetime.now().isoformat(timespec='seconds'),
            f'{self.now_sim():.3f}',
            f'{gx:.6f}' if gx != '' else '',
            f'{gy:.6f}' if gy != '' else '',
            f'{rx:.6f}' if rx != '' else '',
            f'{ry:.6f}' if ry != '' else '',
            f'{distance:.6f}' if isinstance(distance, float) else distance,
        ])
        self.file.flush()

    def clicked_callback(self, msg):
        self.set_goal(float(msg.point.x), float(msg.point.y), msg.header.frame_id or 'unknown')

    def goal_pose_callback(self, msg):
        self.set_goal(
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            msg.header.frame_id or 'unknown',
        )

    def set_goal(self, x, y, frame_id):
        self.goal_index += 1
        self.goal = (x, y)
        self.arrival_logged = False
        self.write_event('goal_received')
        self.get_logger().info(
            f'Goal #{self.goal_index}: x={self.goal[0]:.3f}, y={self.goal[1]:.3f}, '
            f'frame={frame_id}'
        )

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        self.pose = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            yaw_from_quat(q),
        )

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def control_loop(self):
        if self.pose is None or self.goal is None:
            return

        rx, ry, yaw = self.pose
        gx, gy = self.goal
        dx = gx - rx
        dy = gy - ry
        distance = math.hypot(dx, dy)

        if distance <= self.goal_tolerance:
            self.publish_stop()
            if not self.arrival_logged:
                self.write_event('arrived', distance)
                self.get_logger().info(
                    f'Arrived at goal #{self.goal_index}: distance={distance:.2f} m'
                )
                self.arrival_logged = True
            return

        target_yaw = math.atan2(dy, dx)
        yaw_error = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))

        msg = Twist()
        msg.angular.z = clamp(self.k_angular * yaw_error, -self.max_angular, self.max_angular)

        if abs(yaw_error) > self.stop_heading:
            msg.linear.x = 0.0
        else:
            heading_scale = max(0.38, 1.0 - abs(yaw_error) / self.slow_heading)
            target_linear = min(self.max_linear, self.k_linear * distance) * heading_scale
            if distance > 1.0:
                target_linear = max(self.min_rolling_linear, target_linear)
            msg.linear.x = target_linear

        self.cmd_pub.publish(msg)

    def close(self):
        self.publish_stop()
        self.file.flush()
        self.file.close()


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'logs/click_to_drive_goals.csv'
    rclpy.init()
    node = ClickToDrive(output_path)
    stop = {'requested': False}

    def request_stop(signum, frame):
        stop['requested'] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        while rclpy.ok() and not stop['requested']:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
