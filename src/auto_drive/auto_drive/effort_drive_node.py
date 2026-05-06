#!/usr/bin/env python3
"""
ROS2-control effort bridge for the Leo rover.

This node replaces the Gazebo DiffDrive velocity actuator with a torque-level
wheel driver.  It subscribes to the existing /cmd_vel from auto_drive_node and
converts it to four wheel effort commands for JointGroupEffortController.
When the rover is commanded forward but odometry shows no progress on a slope,
it overrides /cmd_vel with a short torque recovery sequence:
  unload -> high-torque climb with wiggle -> retreat/turn escape if needed.
"""

import math
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray


class EffortDriveNode(Node):
    def __init__(self):
        super().__init__('effort_drive_node')

        self.cmd_vel_topic = self.declare_parameter('cmd_vel_topic', '/cmd_vel').value
        self.odom_topic = self.declare_parameter('odom_topic', '/odom_truth').value
        self.imu_topic = self.declare_parameter('imu_topic', '/imu').value
        self.joint_state_topic = self.declare_parameter('joint_state_topic', '/joint_states').value
        self.effort_topic = self.declare_parameter(
            'effort_topic', '/wheel_effort_controller/commands'
        ).value

        self.left_joints = list(self.declare_parameter(
            'left_joints', ['left_front_wheel_joint', 'left_rear_wheel_joint']
        ).value)
        self.right_joints = list(self.declare_parameter(
            'right_joints', ['right_front_wheel_joint', 'right_rear_wheel_joint']
        ).value)
        self.joint_order = self.left_joints + self.right_joints

        self.wheel_radius = float(self.declare_parameter('wheel_radius', 0.0855).value)
        self.wheel_separation = float(self.declare_parameter('wheel_separation', 0.50).value)
        self.angular_effort_sign = float(self.declare_parameter('angular_effort_sign', 1.0).value)
        self.max_wheel_speed = float(self.declare_parameter('max_wheel_speed', 90.0).value)

        # Velocity-to-effort servo.  Keep gains modest; the recovery phases add
        # torque only when needed, which avoids constant bouncing on rough tiles.
        self.kp_velocity = float(self.declare_parameter('kp_velocity', 8.5).value)
        self.kd_velocity = float(self.declare_parameter('kd_velocity', 0.18).value)
        self.feedforward_effort = float(self.declare_parameter('feedforward_effort', 1.8).value)
        self.static_effort = float(self.declare_parameter('static_effort', 3.2).value)
        self.normal_effort_limit = float(self.declare_parameter('normal_effort_limit', 65.0).value)
        self.recovery_effort_limit = float(self.declare_parameter('recovery_effort_limit', 180.0).value)
        self.emergency_effort_limit = float(self.declare_parameter('emergency_effort_limit', 260.0).value)
        self.command_timeout = float(self.declare_parameter('command_timeout', 0.45).value)
        self.publish_rate = float(self.declare_parameter('publish_rate', 80.0).value)

        # Slope / no-progress detection.
        self.slope_pitch_threshold = float(self.declare_parameter('slope_pitch_threshold', 0.13).value)
        self.stall_timeout = float(self.declare_parameter('stall_timeout', 1.3).value)
        self.min_command_linear = float(self.declare_parameter('min_command_linear', 0.22).value)
        self.max_forward_progress_speed = float(self.declare_parameter('max_forward_progress_speed', 0.035).value)
        self.traction_recovery_enabled = bool(self.declare_parameter('traction_recovery_enabled', True).value)
        self.traction_recovery_max_angular = float(self.declare_parameter('traction_recovery_max_angular', 0.25).value)
        self.turn_recovery_enabled = bool(self.declare_parameter('turn_recovery_enabled', True).value)
        self.min_command_angular = float(self.declare_parameter('min_command_angular', 0.45).value)
        self.max_turn_progress_speed = float(self.declare_parameter('max_turn_progress_speed', 0.08).value)
        self.turn_stall_timeout = float(self.declare_parameter('turn_stall_timeout', 1.2).value)
        self.turn_unstick_time = float(self.declare_parameter('turn_unstick_time', 1.0).value)
        self.turn_unstick_effort = float(self.declare_parameter('turn_unstick_effort', 82.0).value)
        self.turn_unstick_limit = float(self.declare_parameter('turn_unstick_limit', 95.0).value)
        self.success_turn_speed = float(self.declare_parameter('success_turn_speed', 0.18).value)
        self.success_forward_speed = float(self.declare_parameter('success_forward_speed', 0.12).value)
        self.success_progress_distance = float(self.declare_parameter('success_progress_distance', 0.18).value)
        self.max_boost_attempts = int(self.declare_parameter('max_boost_attempts', 2).value)
        self.boost_cooldown = float(self.declare_parameter('boost_cooldown', 2.5).value)

        # Recovery effort sequence.
        self.unload_time = float(self.declare_parameter('unload_time', 0.65).value)
        self.climb_time = float(self.declare_parameter('climb_time', 4.2).value)
        self.escape_reverse_time = float(self.declare_parameter('escape_reverse_time', 2.2).value)
        self.escape_turn_time = float(self.declare_parameter('escape_turn_time', 1.9).value)
        self.unload_effort = float(self.declare_parameter('unload_effort', 95.0).value)
        self.climb_effort = float(self.declare_parameter('climb_effort', 165.0).value)
        self.wiggle_effort = float(self.declare_parameter('wiggle_effort', 42.0).value)
        self.wiggle_period = float(self.declare_parameter('wiggle_period', 0.32).value)
        self.escape_effort = float(self.declare_parameter('escape_effort', 130.0).value)
        self.turn_effort = float(self.declare_parameter('turn_effort', 140.0).value)

        # In case the URDF wheel axes differ on a local machine, these signs can
        # be flipped from launch without touching code.
        self.left_effort_sign = float(self.declare_parameter('left_effort_sign', -1.0).value)
        self.right_effort_sign = float(self.declare_parameter('right_effort_sign', 1.0).value)
        # Map raw joint velocity to a logical wheel speed where forward vehicle
        # motion is positive on both sides. With the current URDF axes, forward
        # driving normally appears as left raw velocity < 0 and right raw velocity > 0.
        self.left_velocity_sign = float(self.declare_parameter('left_velocity_sign', -1.0).value)
        self.right_velocity_sign = float(self.declare_parameter('right_velocity_sign', 1.0).value)
        self.auto_angular_sign = bool(self.declare_parameter('auto_angular_sign', True).value)
        self.angular_probe_start: Optional[float] = None

        self.cmd: Twist = Twist()
        self.last_cmd_time: Optional[float] = None
        self.last_odom_time: Optional[float] = None
        self.last_odom_pose: Optional[Tuple[float, float, float]] = None
        self.odom_pose: Optional[Tuple[float, float, float]] = None
        self.forward_speed = 0.0
        self.angular_speed = 0.0
        self.pitch = 0.0
        self.joint_velocities: Dict[str, float] = {name: 0.0 for name in self.joint_order}
        self.prev_wheel_errors: Dict[str, float] = {name: 0.0 for name in self.joint_order}
        self.prev_error_time: Optional[float] = None

        self.stall_start_time: Optional[float] = None
        self.turn_stall_start_time: Optional[float] = None
        self.stall_reason = 'slope'
        self.recovery_phase: Optional[str] = None
        self.recovery_phase_start: Optional[float] = None
        self.recovery_start_pose: Optional[Tuple[float, float, float]] = None
        self.recovery_attempts = 0
        self.next_boost_allowed_time = 0.0
        self.escape_turn_sign = 1.0
        self.last_phase_log = ''

        self.effort_pub = self.create_publisher(Float64MultiArray, self.effort_topic, 10)
        self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 20)
        self.create_subscription(Imu, self.imu_topic, self.imu_callback, 20)
        self.create_subscription(JointState, self.joint_state_topic, self.joint_state_callback, 20)
        period = 1.0 / max(5.0, self.publish_rate)
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            'Effort drive ready: /cmd_vel -> %s, joints=%s, normal_limit=%.1f, '
            'recovery_limit=%.1f, effort_signs=(%.0f,%.0f), velocity_signs=(%.0f,%.0f), angular_sign=%.0f. Disable Gazebo DiffDrive when using this node.'
            % (self.effort_topic, self.joint_order, self.normal_effort_limit, self.recovery_effort_limit, self.left_effort_sign, self.right_effort_sign, self.left_velocity_sign, self.right_velocity_sign, self.angular_effort_sign)
        )

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def cmd_callback(self, msg: Twist):
        self.cmd = msg
        self.last_cmd_time = self.now_seconds()

    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        now = self.now_seconds()

        # Prefer twist if available, but keep a pose-difference fallback because
        # some truth odom relays zero out twist.
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        if abs(vx) + abs(vy) > 1e-6:
            self.forward_speed = vx * math.cos(yaw) + vy * math.sin(yaw)
        elif self.last_odom_pose is not None and self.last_odom_time is not None:
            dt = max(1e-6, now - self.last_odom_time)
            dx = pose[0] - self.last_odom_pose[0]
            dy = pose[1] - self.last_odom_pose[1]
            self.forward_speed = (dx * math.cos(yaw) + dy * math.sin(yaw)) / dt
        self.angular_speed = msg.twist.twist.angular.z

        self.last_odom_pose = pose
        self.last_odom_time = now
        self.odom_pose = pose

    def imu_callback(self, msg: Imu):
        q = msg.orientation
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        sinp = max(-1.0, min(1.0, sinp))
        self.pitch = math.asin(sinp)

    def joint_state_callback(self, msg: JointState):
        for name, vel in zip(msg.name, msg.velocity):
            if name in self.joint_velocities:
                self.joint_velocities[name] = float(vel)

    def command_is_fresh(self, now: float) -> bool:
        return self.last_cmd_time is not None and now - self.last_cmd_time <= self.command_timeout

    def desired_wheel_speeds(self, cmd: Twist) -> Tuple[float, float]:
        angular = self.angular_effort_sign * cmd.angular.z
        left = (cmd.linear.x - 0.5 * angular * self.wheel_separation) / self.wheel_radius
        right = (cmd.linear.x + 0.5 * angular * self.wheel_separation) / self.wheel_radius
        left = max(-self.max_wheel_speed, min(self.max_wheel_speed, left))
        right = max(-self.max_wheel_speed, min(self.max_wheel_speed, right))
        return left, right

    def clip(self, value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def publish_efforts(self, left_effort: float, right_effort: float, limit: float):
        left = self.clip(left_effort, limit) * self.left_effort_sign
        right = self.clip(right_effort, limit) * self.right_effort_sign
        msg = Float64MultiArray()
        msg.data = [left, left, right, right]
        self.effort_pub.publish(msg)

    def publish_zero(self):
        self.publish_efforts(0.0, 0.0, self.normal_effort_limit)
        self.prev_error_time = None
        for key in self.prev_wheel_errors:
            self.prev_wheel_errors[key] = 0.0

    def normal_efforts_from_cmd(self, cmd: Twist, now: float) -> Tuple[float, float]:
        left_target, right_target = self.desired_wheel_speeds(cmd)
        dt = 0.0 if self.prev_error_time is None else max(1e-4, now - self.prev_error_time)
        self.prev_error_time = now

        efforts = {}
        for joint_name, target in [
            (self.left_joints[0], left_target),
            (self.left_joints[1], left_target),
            (self.right_joints[0], right_target),
            (self.right_joints[1], right_target),
        ]:
            raw_measured = self.joint_velocities.get(joint_name, 0.0)
            velocity_sign = self.left_velocity_sign if joint_name in self.left_joints else self.right_velocity_sign
            measured = velocity_sign * raw_measured
            error = target - measured
            d_error = 0.0 if dt <= 0.0 else (error - self.prev_wheel_errors[joint_name]) / dt
            self.prev_wheel_errors[joint_name] = error
            effort = self.kp_velocity * error + self.kd_velocity * d_error
            if abs(target) > 0.25:
                effort += math.copysign(self.static_effort + self.feedforward_effort * abs(target), target)
            efforts[joint_name] = effort

        left_effort = 0.5 * (efforts[self.left_joints[0]] + efforts[self.left_joints[1]])
        right_effort = 0.5 * (efforts[self.right_joints[0]] + efforts[self.right_joints[1]])
        return left_effort, right_effort

    def in_drive_stall(self, now: float) -> bool:
        if not self.command_is_fresh(now):
            self.stall_start_time = None
            return False
        if self.cmd.linear.x < self.min_command_linear:
            self.stall_start_time = None
            return False
        if abs(self.forward_speed) > self.max_forward_progress_speed:
            self.stall_start_time = None
            return False
        if self.recovery_phase is not None or now < self.next_boost_allowed_time:
            return False

        slope_stall = abs(self.pitch) >= self.slope_pitch_threshold
        traction_stall = (
            self.traction_recovery_enabled
            and abs(self.cmd.angular.z) <= self.traction_recovery_max_angular
        )
        if not slope_stall and not traction_stall:
            self.stall_start_time = None
            return False

        self.stall_reason = 'slope' if slope_stall else 'traction'
        if self.stall_start_time is None:
            self.stall_start_time = now
            return False
        return now - self.stall_start_time >= self.stall_timeout

    def in_turn_stall(self, now: float) -> bool:
        if not self.turn_recovery_enabled or not self.command_is_fresh(now):
            self.turn_stall_start_time = None
            return False
        if abs(self.cmd.angular.z) < self.min_command_angular:
            self.turn_stall_start_time = None
            return False
        if abs(self.angular_speed) > self.max_turn_progress_speed:
            self.turn_stall_start_time = None
            return False
        if self.recovery_phase is not None or now < self.next_boost_allowed_time:
            return False
        self.stall_reason = 'turn'
        if self.turn_stall_start_time is None:
            self.turn_stall_start_time = now
            return False
        return now - self.turn_stall_start_time >= self.turn_stall_timeout

    def start_recovery(self, now: float):
        self.recovery_attempts += 1
        if self.recovery_attempts <= self.max_boost_attempts:
            self.recovery_phase = 'unload'
            self.recovery_phase_start = now
            self.recovery_start_pose = self.odom_pose
            self.last_phase_log = ''
            self.get_logger().warn(
                'Torque %s recovery %d/%d: pitch=%.2f rad, forward_speed=%.3f m/s. '
                'Applying wheel effort unload -> climb/wiggle.'
                % (
                    self.stall_reason,
                    self.recovery_attempts,
                    self.max_boost_attempts,
                    self.pitch,
                    self.forward_speed,
                )
            )
        else:
            self.recovery_phase = 'escape_reverse'
            self.recovery_phase_start = now
            self.recovery_start_pose = self.odom_pose
            self.escape_turn_sign *= -1.0
            self.last_phase_log = ''
            self.get_logger().warn(
                'Torque boost failed repeatedly. Retreating and rotating away from slope.'
            )

    def start_turn_recovery(self, now: float):
        self.recovery_phase = 'turn_unstick'
        self.recovery_phase_start = now
        self.recovery_start_pose = self.odom_pose
        self.last_phase_log = ''
        self.turn_stall_start_time = None
        self.get_logger().warn(
            'Torque turn recovery: cmd_angular=%.2f rad/s, observed_yaw=%.3f rad/s. '
            'Applying short differential wheel effort pulse.'
            % (self.cmd.angular.z, self.angular_speed)
        )

    def recovery_progress_distance(self) -> float:
        if self.recovery_start_pose is None or self.odom_pose is None:
            return 0.0
        return math.hypot(
            self.odom_pose[0] - self.recovery_start_pose[0],
            self.odom_pose[1] - self.recovery_start_pose[1],
        )

    def finish_recovery(self, now: float, reason: str):
        self.get_logger().info('Torque recovery finished: %s.' % reason)
        self.recovery_phase = None
        self.recovery_phase_start = None
        self.recovery_start_pose = None
        self.stall_start_time = None
        self.next_boost_allowed_time = now + self.boost_cooldown

    def recovery_command(self, now: float) -> Optional[Tuple[float, float, float, str]]:
        if self.recovery_phase is None or self.recovery_phase_start is None:
            return None

        elapsed = now - self.recovery_phase_start
        progress = self.recovery_progress_distance()
        phase = self.recovery_phase
        if phase == 'turn_unstick' and abs(self.angular_speed) >= self.success_turn_speed:
            self.finish_recovery(now, 'yaw speed %.2f rad/s' % self.angular_speed)
            return None
        if progress >= self.success_progress_distance or self.forward_speed >= self.success_forward_speed:
            self.finish_recovery(now, 'progress %.2f m, forward %.2f m/s' % (progress, self.forward_speed))
            return None
        if phase == 'turn_unstick':
            if elapsed >= self.turn_unstick_time:
                self.finish_recovery(now, 'turn pulse elapsed')
                return None
            sign = 1.0 if self.cmd.angular.z >= 0.0 else -1.0
            return sign * self.turn_unstick_effort, -sign * self.turn_unstick_effort, self.turn_unstick_limit, 'turn_unstick'

        if phase == 'unload':
            if elapsed >= self.unload_time:
                self.recovery_phase = 'climb'
                self.recovery_phase_start = now
                self.last_phase_log = ''
                return self.recovery_command(now)
            return -self.unload_effort, -self.unload_effort, self.recovery_effort_limit, 'unload'

        if phase == 'climb':
            if elapsed >= self.climb_time:
                self.next_boost_allowed_time = now + self.boost_cooldown
                self.recovery_phase = None
                self.recovery_phase_start = None
                self.stall_start_time = now
                self.get_logger().warn(
                    'Torque climb attempt ended with progress %.3f m. Will retry or escape.' % progress
                )
                return None
            wiggle = self.wiggle_effort if math.sin(2.0 * math.pi * elapsed / max(0.05, self.wiggle_period)) >= 0.0 else -self.wiggle_effort
            left = self.climb_effort + wiggle
            right = self.climb_effort - wiggle
            return left, right, self.recovery_effort_limit, 'climb_wiggle'

        if phase == 'escape_reverse':
            if elapsed >= self.escape_reverse_time:
                self.recovery_phase = 'escape_turn'
                self.recovery_phase_start = now
                self.last_phase_log = ''
                return self.recovery_command(now)
            return -self.escape_effort, -self.escape_effort, self.emergency_effort_limit, 'escape_reverse'

        if phase == 'escape_turn':
            if elapsed >= self.escape_turn_time:
                self.recovery_attempts = 0
                self.finish_recovery(now, 'escape completed')
                return None
            sign = 1.0 if self.escape_turn_sign >= 0.0 else -1.0
            return -sign * self.turn_effort, sign * self.turn_effort, self.emergency_effort_limit, 'escape_turn'

        self.finish_recovery(now, 'unknown phase')
        return None


    def maybe_autoflip_angular_sign(self, now: float):
        if not self.auto_angular_sign or not self.command_is_fresh(now):
            self.angular_probe_start = None
            return
        # MeshScanNode rotates in place when the heading error is large.  This
        # gives a safe low-speed opportunity to verify whether positive cmd.z
        # produces positive odom yaw.  If not, flip only the angular mapping.
        if abs(self.cmd.angular.z) < 0.28 or abs(self.cmd.linear.x) > 0.04:
            self.angular_probe_start = None
            return
        if self.angular_probe_start is None:
            self.angular_probe_start = now
            return
        if now - self.angular_probe_start < 0.75:
            return
        if abs(self.angular_speed) < 0.06:
            return
        if self.cmd.angular.z * self.angular_speed < 0.0:
            self.angular_effort_sign *= -1.0
            self.angular_probe_start = now
            self.get_logger().warn(
                'Auto-flipped angular_effort_sign to %.0f because cmd.z=%.2f produced odom yaw_rate=%.2f.'
                % (self.angular_effort_sign, self.cmd.angular.z, self.angular_speed)
            )

    def timer_callback(self):
        now = self.now_seconds()
        self.maybe_autoflip_angular_sign(now)

        if self.in_drive_stall(now):
            self.start_recovery(now)
        elif self.in_turn_stall(now):
            self.start_turn_recovery(now)

        recovery = self.recovery_command(now)
        if recovery is not None:
            left, right, limit, phase_name = recovery
            if phase_name != self.last_phase_log:
                self.get_logger().warn(
                    'Wheel effort recovery phase=%s left=%.1f right=%.1f limit=%.1f'
                    % (phase_name, left, right, limit)
                )
                self.last_phase_log = phase_name
            self.publish_efforts(left, right, limit)
            return

        if not self.command_is_fresh(now):
            self.publish_zero()
            return

        if abs(self.cmd.linear.x) < 1e-4 and abs(self.cmd.angular.z) < 1e-4:
            self.publish_zero()
            return

        if abs(self.angular_speed) > 2.2 and abs(self.cmd.angular.z) < 0.9:
            self.get_logger().warn('Unexpected yaw rate %.2f rad/s for cmd.z %.2f; cutting wheel effort briefly.' % (self.angular_speed, self.cmd.angular.z))
            self.publish_zero()
            return

        left, right = self.normal_efforts_from_cmd(self.cmd, now)
        self.publish_efforts(left, right, self.normal_effort_limit)


def main(args=None):
    rclpy.init(args=args)
    node = EffortDriveNode()
    try:
        rclpy.spin(node)
    finally:
        node.publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
