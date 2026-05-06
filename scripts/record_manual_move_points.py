#!/usr/bin/env python3
import csv
import math
import os
import signal
import sys
from datetime import datetime

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class ManualMoveRecorder(Node):
    def __init__(self, output_path):
        super().__init__('manual_move_recorder')
        self.output_path = output_path
        self.min_distance = 0.25
        self.last_record = None
        self.count = 0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.file = open(output_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['index', 'wall_time', 'sim_sec', 'x', 'y', 'z', 'yaw_rad'])
        self.file.flush()

        self.create_subscription(Odometry, '/odom_truth', self.odom_callback, 20)
        self.get_logger().info(
            f'Recording manual move points to {output_path} '
            f'(distance threshold {self.min_distance:.2f} m)'
        )

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        point = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            float(msg.pose.pose.position.z),
            yaw,
        )

        if self.last_record is not None:
            dist = math.hypot(point[0] - self.last_record[0], point[1] - self.last_record[1])
            if dist < self.min_distance:
                return

        self.count += 1
        sim_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.writer.writerow([
            self.count,
            datetime.now().isoformat(timespec='seconds'),
            f'{sim_sec:.3f}',
            f'{point[0]:.6f}',
            f'{point[1]:.6f}',
            f'{point[2]:.6f}',
            f'{point[3]:.6f}',
        ])
        self.file.flush()
        self.last_record = point
        self.get_logger().info(
            f'#{self.count}: x={point[0]:.3f}, y={point[1]:.3f}, '
            f'z={point[2]:.3f}, yaw={point[3]:.3f}'
        )

    def close(self):
        self.file.flush()
        self.file.close()


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'logs/manual_move_points.csv'
    rclpy.init()
    node = ManualMoveRecorder(output_path)

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
