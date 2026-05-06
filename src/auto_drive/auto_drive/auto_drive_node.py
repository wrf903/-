#!/usr/bin/env python3
"""
自主驾驶节点：支持视觉巡线、边界触发巡航和弓字形覆盖路径扫描。
默认 mode=line_follow。切换为 waypoint 模式时启用弓字形 Pure Pursuit。
"""
import heapq
import itertools
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import math


class ADNode(Node):
    def __init__(self):
        super().__init__('auto_drive_node')

        self.mode = self.declare_parameter('mode', 'line_follow').value
        self.enabled = self.declare_parameter('enabled', True).value
        self.linear_speed = self.declare_parameter('linear_speed', 0.8).value
        self.min_linear_speed = self.declare_parameter(
            'min_linear_speed', min(0.24, self.linear_speed)
        ).value
        self.max_angular_speed = self.declare_parameter('max_angular_speed', 0.5).value
        self.lookahead = self.declare_parameter('lookahead', 1.5).value
        self.waypoint_radius = self.declare_parameter('waypoint_radius', 0.45).value
        self.path_progress_window = self.declare_parameter(
            'path_progress_window', 12
        ).value
        self.row_spacing = self.declare_parameter('row_spacing', 1.5).value
        self.turn_radius = self.declare_parameter(
            'turn_radius', self.row_spacing / 2.0
        ).value
        self.path_resolution = self.declare_parameter('path_resolution', 0.35).value
        self.coverage_strategy = self.declare_parameter(
            'coverage_strategy', 'sparse_scan'
        ).value
        self.scan_pattern = self.declare_parameter(
            'scan_pattern', 'four_intersections'
        ).value
        self.scan_extent = self.declare_parameter('scan_extent', 3.0).value
        self.sparse_linear_speed = self.declare_parameter(
            'sparse_linear_speed', min(0.45, self.linear_speed)
        ).value
        self.sparse_turn_linear_speed = self.declare_parameter(
            'sparse_turn_linear_speed', 0.14
        ).value
        self.sparse_max_angular_speed = self.declare_parameter(
            'sparse_max_angular_speed', min(0.45, self.max_angular_speed)
        ).value
        self.boundary_escape_speed = self.declare_parameter(
            'boundary_escape_speed', 0.18
        ).value
        self.boundary_turn_linear_speed = self.declare_parameter(
            'boundary_turn_linear_speed', min(0.45, self.linear_speed)
        ).value
        self.boundary_turn_angular_speed = self.declare_parameter(
            'boundary_turn_angular_speed', min(0.8, self.max_angular_speed)
        ).value
        self.boundary_turn_angle = self.declare_parameter(
            'boundary_turn_angle', math.pi / 2.0
        ).value
        self.boundary_turn_completion_tolerance = self.declare_parameter(
            'boundary_turn_completion_tolerance', 0.06
        ).value
        self.boundary_turn_stall_timeout = self.declare_parameter(
            'boundary_turn_stall_timeout', 2.0
        ).value
        self.boundary_turn_min_yaw_rate = self.declare_parameter(
            'boundary_turn_min_yaw_rate', 0.18
        ).value
        self.boundary_turn_slow_linear_speed = self.declare_parameter(
            'boundary_turn_slow_linear_speed', 0.12
        ).value
        self.boundary_turn_rate_grace = self.declare_parameter(
            'boundary_turn_rate_grace', 0.6
        ).value
        self.boundary_turn_slow_safety_margin = self.declare_parameter(
            'boundary_turn_slow_safety_margin', 2.0
        ).value
        self.boundary_turn_trigger_margin = self.declare_parameter(
            'boundary_turn_trigger_margin', 1.0
        ).value
        self.boundary_turn_rearm_margin = self.declare_parameter(
            'boundary_turn_rearm_margin', 1.2
        ).value
        self.boundary_turn_shrink_per_turn = self.declare_parameter(
            'boundary_turn_shrink_per_turn', 0.0
        ).value
        self.boundary_turn_min_extent = self.declare_parameter(
            'boundary_turn_min_extent', 1.0
        ).value
        self.start_x = self.declare_parameter('start_x', 0.75).value
        self.start_y = self.declare_parameter('start_y', 1.0).value
        self.start_yaw = self.declare_parameter('start_yaw', 0.0).value
        self.force_initial_forward_waypoint = self.declare_parameter(
            'force_initial_forward_waypoint', False
        ).value
        self.initial_forward_distance = self.declare_parameter(
            'initial_forward_distance', max(1.8, self.lookahead + 0.4)
        ).value
        self.startup_straight_distance = self.declare_parameter(
            'startup_straight_distance', 0.8
        ).value
        self.startup_blend_distance = self.declare_parameter(
            'startup_blend_distance', 2.2
        ).value
        self.startup_linear_speed = self.declare_parameter(
            'startup_linear_speed', 0.42
        ).value
        self.startup_max_angular_speed = self.declare_parameter(
            'startup_max_angular_speed', 0.16
        ).value
        self.spin_scan_turns = self.declare_parameter('spin_scan_turns', 1.0).value
        self.spin_scan_angular_speed = self.declare_parameter(
            'spin_scan_angular_speed', 0.35
        ).value
        self.scan_spin_turns = self.declare_parameter('scan_spin_turns', 1.0).value
        self.scan_spin_angular_speed = self.declare_parameter(
            'scan_spin_angular_speed', 0.35
        ).value
        self.odom_topic = self.declare_parameter('odom_topic', '/odom').value
        self.field_x_min = self.declare_parameter('field_x_min', -7.0).value
        self.field_x_max = self.declare_parameter('field_x_max', 7.0).value
        self.field_y_min = self.declare_parameter('field_y_min', -7.0).value
        self.field_y_max = self.declare_parameter('field_y_max', 7.0).value
        self.field_guard_margin = self.declare_parameter(
            'field_guard_margin', 0.0
        ).value
        self.field_guard_lookahead = self.declare_parameter(
            'field_guard_lookahead', 1.2
        ).value
        self.field_guard_max_angular_speed = self.declare_parameter(
            'field_guard_max_angular_speed', min(0.45, self.max_angular_speed)
        ).value
        self.field_guard_speed = self.declare_parameter(
            'field_guard_speed', min(0.55, self.linear_speed)
        ).value
        self.terrain_x_min = self.declare_parameter('terrain_x_min', -10.0).value
        self.terrain_x_max = self.declare_parameter('terrain_x_max', 10.0).value
        self.terrain_y_min = self.declare_parameter('terrain_y_min', -10.0).value
        self.terrain_y_max = self.declare_parameter('terrain_y_max', 10.0).value
        self.terrain_edge_margin = self.declare_parameter(
            'terrain_edge_margin', 0.5
        ).value
        self.safety_x_min = self.declare_parameter(
            'safety_x_min', self.terrain_x_min + self.terrain_edge_margin
        ).value
        self.safety_x_max = self.declare_parameter(
            'safety_x_max', self.terrain_x_max - self.terrain_edge_margin
        ).value
        self.safety_y_min = self.declare_parameter(
            'safety_y_min', self.terrain_y_min + self.terrain_edge_margin
        ).value
        self.safety_y_max = self.declare_parameter(
            'safety_y_max', self.terrain_y_max - self.terrain_edge_margin
        ).value
        # Boundary intervention distance. This is only for true edge guidance;
        # distant slowing can stall the rover before it reaches the next waypoint.
        self.brake_dist = self.declare_parameter('brake_dist', 2.0).value
        self.boundary_stop_margin = self.declare_parameter(
            'boundary_stop_margin', 0.5
        ).value
        # Independent work-area guard.  The safety box protects the terrain
        # edge; this guard protects the requested scanning/working field.
        # It uses the full footprint plus a predicted pose, so it reacts before
        # the robot centre crosses the work boundary.
        self.work_guard_margin = self.declare_parameter(
            'work_guard_margin', 0.85
        ).value
        self.work_guard_lookahead_time = self.declare_parameter(
            'work_guard_lookahead_time', 1.2
        ).value
        self.work_guard_lookahead_dist = self.declare_parameter(
            'work_guard_lookahead_dist', 1.0
        ).value
        self.work_guard_speed = self.declare_parameter(
            'work_guard_speed', min(0.45, self.linear_speed)
        ).value
        self.work_guard_reverse_speed = self.declare_parameter(
            'work_guard_reverse_speed', 0.14
        ).value
        self.footprint_length = self.declare_parameter(
            'footprint_length', 0.78
        ).value
        self.footprint_width = self.declare_parameter(
            'footprint_width', 0.53
        ).value
        self.footprint_safety_margin = self.declare_parameter(
            'footprint_safety_margin', 0.25
        ).value
        self.route_footprint_margin = self.declare_parameter(
            'route_footprint_margin', 0.15
        ).value
        self.max_footprint_height_delta = self.declare_parameter(
            'max_footprint_height_delta', 0.35
        ).value
        self.max_footprint_roughness = self.declare_parameter(
            'max_footprint_roughness', 0.045
        ).value
        self.footprint_step_cost_weight = self.declare_parameter(
            'footprint_step_cost_weight', 18.0
        ).value
        self.boundary_cost_weight = self.declare_parameter(
            'boundary_cost_weight', 3.0
        ).value
        self.auto_optimize = self.declare_parameter('auto_optimize', True).value
        self.slope_avoidance_enabled = self.declare_parameter(
            'slope_avoidance_enabled', True
        ).value
        self.terrain_model_name = self.declare_parameter(
            'terrain_model_name', 'terrain_alpha2_h15'
        ).value
        self.max_route_slope = self.declare_parameter('max_route_slope', 0.20).value
        self.max_downhill_slope = self.declare_parameter(
            'max_downhill_slope', 0.14
        ).value
        self.slope_cost_weight = self.declare_parameter('slope_cost_weight', 4.0).value
        self.downhill_cost_weight = self.declare_parameter(
            'downhill_cost_weight', 12.0
        ).value
        self.downhill_slow_slope = self.declare_parameter(
            'downhill_slow_slope', 0.06
        ).value
        self.slope_min_linear_speed = self.declare_parameter(
            'slope_min_linear_speed', 0.22
        ).value
        # Optional full-attitude odometry is used only to classify slope stalls.
        # Keep the navigation pose source unchanged, because the point-cloud TF
        # pipeline may intentionally publish yaw-only odometry for stability.
        self.attitude_odom_topic = self.declare_parameter(
            'attitude_odom_topic', '/model/leo_rover/odometry'
        ).value
        self.slope_stuck_pitch_threshold = self.declare_parameter(
            'slope_stuck_pitch_threshold', 0.14
        ).value
        self.slope_stuck_grade_threshold = self.declare_parameter(
            'slope_stuck_grade_threshold', 0.14
        ).value
        self.slope_stuck_timeout = self.declare_parameter(
            'slope_stuck_timeout', 2.2
        ).value
        self.slope_stuck_min_command_speed = self.declare_parameter(
            'slope_stuck_min_command_speed', 0.25
        ).value
        self.slope_stuck_max_observed_speed = self.declare_parameter(
            'slope_stuck_max_observed_speed', 0.045
        ).value
        self.engine_boost_enabled = self.declare_parameter(
            'engine_boost_enabled', True
        ).value
        self.engine_boost_reverse_time = self.declare_parameter(
            'engine_boost_reverse_time', 0.55
        ).value
        self.engine_boost_time = self.declare_parameter(
            'engine_boost_time', 2.8
        ).value
        self.engine_boost_speed = self.declare_parameter(
            'engine_boost_speed', 1.65
        ).value
        self.engine_boost_reverse_speed = self.declare_parameter(
            'engine_boost_reverse_speed', 0.26
        ).value
        self.engine_boost_wiggle_angular_speed = self.declare_parameter(
            'engine_boost_wiggle_angular_speed', 0.42
        ).value
        self.engine_boost_wiggle_period = self.declare_parameter(
            'engine_boost_wiggle_period', 0.42
        ).value
        self.engine_boost_boundary_clearance = self.declare_parameter(
            'engine_boost_boundary_clearance', 0.75
        ).value
        self.engine_boost_max_attempts = self.declare_parameter(
            'engine_boost_max_attempts', 2
        ).value
        # Let recovery commands override the predictive work-area guard when
        # the rover is still safely inside the work field.  The previous
        # version detected slope stalls correctly, but the field guard often
        # replaced the boost command immediately because predicted clearance
        # dipped just below the margin.
        self.field_guard_yield_to_recovery = self.declare_parameter(
            'field_guard_yield_to_recovery', True
        ).value
        self.slope_escape_reverse_time = self.declare_parameter(
            'slope_escape_reverse_time', 3.0
        ).value
        self.slope_escape_turn_time = self.declare_parameter(
            'slope_escape_turn_time', 2.4
        ).value
        self.slope_escape_forward_time = self.declare_parameter(
            'slope_escape_forward_time', 3.2
        ).value
        self.slope_escape_reverse_speed = self.declare_parameter(
            'slope_escape_reverse_speed', 0.42
        ).value
        self.slope_escape_forward_speed = self.declare_parameter(
            'slope_escape_forward_speed', 0.46
        ).value
        self.slope_escape_angular_speed = self.declare_parameter(
            'slope_escape_angular_speed', 0.95
        ).value
        self.slope_escape_cooldown = self.declare_parameter(
            'slope_escape_cooldown', 8.0
        ).value
        # Progress watchdog.  A simple "pulse the wheels" command was not
        # enough on the tiled uneven terrain; this state machine backs out,
        # turns away from the nearest boundary, and optionally skips a few
        # unreachable waypoints.
        self.stuck_detection_enabled = self.declare_parameter(
            'stuck_detection_enabled', True
        ).value
        self.stuck_timeout = self.declare_parameter('stuck_timeout', 4.0).value
        self.stuck_min_progress = self.declare_parameter(
            'stuck_min_progress', 0.12
        ).value
        self.stuck_min_yaw_progress = self.declare_parameter(
            'stuck_min_yaw_progress', 0.18
        ).value
        self.stuck_reverse_time = self.declare_parameter(
            'stuck_reverse_time', 1.2
        ).value
        self.stuck_turn_time = self.declare_parameter(
            'stuck_turn_time', 1.6
        ).value
        self.stuck_reverse_speed = self.declare_parameter(
            'stuck_reverse_speed', 0.18
        ).value
        self.stuck_forward_speed = self.declare_parameter(
            'stuck_forward_speed', 0.20
        ).value
        self.stuck_turn_angular_speed = self.declare_parameter(
            'stuck_turn_angular_speed', 0.55
        ).value
        self.stuck_skip_waypoints = self.declare_parameter(
            'stuck_skip_waypoints', 3
        ).value
        self._terrain = None
        if self.slope_avoidance_enabled:
            self._terrain = self._load_collision_terrain()

        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self._opt_client = self.create_client(Trigger, '/pointcloud_optimizer/optimize')
        self._mesh_client = self.create_client(Trigger, '/pointcloud_to_mesh/reconstruct')
        self.bridge = CvBridge()

        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 10
        )

        self.pose = [0.0, 0.0, 0.0]
        self.pose_z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.observed_planar_speed = 0.0
        self.observed_forward_speed = 0.0
        self.observed_vertical_speed = 0.0
        self._last_pose_sample = None
        self._attitude_stamp = None
        self.yaw_speed = 0.0
        self._completed = False
        self._in_place_turn = False
        self._startup_origin = None
        self._startup_done = self.startup_blend_distance <= 0.0
        self._spin_last_yaw = None
        self._spin_accum = 0.0
        self._scan_targets = []
        self._scan_target_idx = 0
        self._scan_spin_last_yaw = None
        self._scan_spin_accum = 0.0
        self._boundary_turning = False
        self._boundary_turn_last_yaw = None
        self._boundary_turn_accum = 0.0
        self._boundary_turn_count = 0
        self._boundary_turn_completed_count = 0
        self._boundary_turn_latched = set()
        self._boundary_turn_last_progress_time = None
        self._boundary_turn_last_progress_accum = 0.0
        self._boundary_turn_start_time = None
        self._boundary_turn_last_update_time = None
        self._boundary_turn_slow_warned = False
        self._stuck_ref_pose = None
        self._stuck_ref_time = None
        self._stuck_recovery_phase = None
        self._stuck_recovery_until = None
        self._stuck_recovery_turn_dir = 1.0
        self._stuck_recovery_count = 0
        self._stuck_recovery_pending_skip = False
        self._stuck_recovery_kind = None
        self._stuck_recovery_start_time = None
        self._engine_boost_attempts = 0
        self._slope_escape_cooldown_until = 0.0
        self._recovery_last_phase_logged = None
        self._field_guard_recovery_override_warned = False

        self.attitude_odom_sub = None
        if self.attitude_odom_topic:
            self.attitude_odom_sub = self.create_subscription(
                Odometry, self.attitude_odom_topic, self.attitude_odom_callback, 10
            )

        if self.mode == 'line_follow':
            self.img_sub = self.create_subscription(
                Image, '/oakd/rgb/preview/image_raw', self.img_callback, 1
            )
        elif self.mode == 'boundary_turn':
            self.waypoints = []
            self.current_wp_idx = 0
            self.get_logger().info(
                f'Boundary-turn mode: work field '
                f'[{self.field_x_min},{self.field_x_max}] x '
                f'[{self.field_y_min},{self.field_y_max}], safety '
                f'[{self.safety_x_min},{self.safety_x_max}] x '
                f'[{self.safety_y_min},{self.safety_y_max}], '
                f'cruise={self.linear_speed:.2f} m/s, '
                f'turn={self.boundary_turn_linear_speed:.2f} m/s + '
                f'{self.boundary_turn_angular_speed:.2f} rad/s.'
            )
        elif self.mode == 'spin_scan':
            self.waypoints = []
            self.current_wp_idx = 0
            self.get_logger().info(
                f'Spin scan mode: turns={self.spin_scan_turns:.2f}, '
                f'angular_speed={self.spin_scan_angular_speed:.2f} rad/s, '
                f'odom_topic={self.odom_topic}'
            )
        else:
            self.waypoints = self._generate_smooth_coverage_path()
            self.current_wp_idx = 0
            path_type = 'slope-aware' if self._terrain is not None else 'straight'
            self.get_logger().info(
                f'Waypoint mode: {len(self.waypoints)} {path_type} waypoints, '
                f'field [{self.field_x_min},{self.field_x_max}] x '
                f'[{self.field_y_min},{self.field_y_max}], '
                f'terrain [{self.terrain_x_min},{self.terrain_x_max}] x '
                f'[{self.terrain_y_min},{self.terrain_y_max}], '
                f'software safety [{self.safety_x_min},{self.safety_x_max}] x '
                f'[{self.safety_y_min},{self.safety_y_max}], '
                f'odom_topic={self.odom_topic}'
            )

        self.timer = self.create_timer(0.05, self.timer_callback)

    def _now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        roll, pitch, yaw = self._rpy_from_quaternion(q.x, q.y, q.z, q.w)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        now = self._now_seconds()
        if self._last_pose_sample is not None:
            last_x, last_y, last_z, last_yaw, last_time = self._last_pose_sample
            dt = max(1e-4, now - last_time)
            dx = x - last_x
            dy = y - last_y
            dz = z - last_z
            self.observed_planar_speed = math.hypot(dx, dy) / dt
            self.observed_forward_speed = (dx * math.cos(yaw) + dy * math.sin(yaw)) / dt
            self.observed_vertical_speed = dz / dt
        self._last_pose_sample = (x, y, z, yaw, now)

        self.pose = [x, y, yaw]
        self.pose_z = z
        # If the navigation odometry already carries full orientation, use it.
        # When /odom_truth is yaw-only, attitude_odom_callback will overwrite
        # these with the full Gazebo model attitude.
        if not self.attitude_odom_topic or self.attitude_odom_topic == self.odom_topic:
            self.roll = roll
            self.pitch = pitch
            self._attitude_stamp = now

    def attitude_odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        self.roll, self.pitch, _ = self._rpy_from_quaternion(q.x, q.y, q.z, q.w)
        self._attitude_stamp = self._now_seconds()

    @staticmethod
    def _rpy_from_quaternion(x, y, z, w):
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr, cosr)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny, cosy)
        return roll, pitch, yaw

    def _generate_smooth_coverage_path(self):
        if self._terrain is not None:
            return self._generate_slope_aware_coverage_path()
        return self._generate_straight_coverage_path()

    def _initial_forward_waypoint(self):
        distance = max(self.initial_forward_distance, self.lookahead + 0.2)
        x = self.start_x + distance * math.cos(self.start_yaw)
        y = self.start_y + distance * math.sin(self.start_yaw)
        return (
            max(self.field_x_min, min(self.field_x_max, x)),
            max(self.field_y_min, min(self.field_y_max, y)),
        )

    def _generate_straight_coverage_path(self):
        """Generate a full-field boustrophedon path with straight row connectors."""
        y_values = []
        y = self.field_y_min
        while y <= self.field_y_max + 1e-6:
            y_values.append(y)
            y += self.row_spacing
        if y_values and y_values[-1] < self.field_y_max - 1e-6:
            y_values.append(self.field_y_max)

        x_min = self.field_x_min
        x_max = self.field_x_max
        waypoints = []
        direction = 1
        current_x = max(x_min, min(x_max, self.start_x))
        current_y = self.start_y
        if self.force_initial_forward_waypoint:
            forward = self._initial_forward_waypoint()
            waypoints.append([forward[0], forward[1]])
            current_x, current_y = forward

        for y in y_values:
            row_start_x = x_min if direction > 0 else x_max
            row_end_x = x_max if direction > 0 else x_min

            self._append_line(waypoints, current_x, current_y, row_start_x, y)
            self._append_line(waypoints, row_start_x, y, row_end_x, y)
            current_x, current_y = row_end_x, y
            direction *= -1

        return waypoints

    def _load_collision_terrain(self):
        """Load the smoothed collision tiles that the rover actually drives on."""
        try:
            orchard_share = get_package_share_directory('orchard_sim')
            model_sdf = os.path.join(
                orchard_share, 'models', self.terrain_model_name, 'model.sdf'
            )
            root = ET.parse(model_sdf).getroot()
        except Exception as exc:
            self.get_logger().warn(
                f'Slope avoidance disabled: failed to load terrain model '
                f'{self.terrain_model_name}: {exc}'
            )
            return None

        samples = []
        for collision in root.findall('.//collision'):
            pose_text = collision.findtext('pose')
            size = collision.find('./geometry/box/size')
            if not pose_text or size is None or size.text is None:
                continue
            try:
                pose_vals = [float(v) for v in pose_text.split()]
                size_vals = [float(v) for v in size.text.split()]
            except ValueError:
                continue
            x = pose_vals[0]
            y = pose_vals[1]
            top_z = pose_vals[2] + size_vals[2] * 0.5
            samples.append((x, y, top_z))

        if not samples:
            self.get_logger().warn(
                f'Slope avoidance disabled: no collision height samples in {model_sdf}'
            )
            return None

        xs = sorted({round(x, 6) for x, _, _ in samples})
        ys = sorted({round(y, 6) for _, y, _ in samples})
        heights = {}
        for x, y, z in samples:
            heights[(round(x, 6), round(y, 6))] = z

        self.get_logger().info(
            f'Loaded slope map from {self.terrain_model_name}: '
            f'{len(xs)}x{len(ys)} cells, z={min(heights.values()):.2f}..'
            f'{max(heights.values()):.2f} m, max_route_slope={self.max_route_slope:.2f}, '
            f'max_downhill_slope={self.max_downhill_slope:.2f}'
        )
        return {'xs': xs, 'ys': ys, 'heights': heights}

    def _generate_slope_aware_coverage_path(self):
        if self.coverage_strategy == 'sparse_scan':
            return self._generate_sparse_scan_path()

        y_values = []
        y = self.field_y_min
        while y <= self.field_y_max + 1e-6:
            y_values.append(y)
            y += self.row_spacing
        if y_values and y_values[-1] < self.field_y_max - 1e-6:
            y_values.append(self.field_y_max)

        waypoints = []
        current = (self.start_x, self.start_y)
        x_min = self.field_x_min
        x_max = self.field_x_max
        relaxed_segments = 0

        if self.force_initial_forward_waypoint:
            forward = self._initial_forward_waypoint()
            self._extend_waypoints(waypoints, [forward])
            current = forward
            dot = (
                (forward[0] - self.start_x) * math.cos(self.start_yaw)
                + (forward[1] - self.start_y) * math.sin(self.start_yaw)
            )
            self.get_logger().info(
                f'Forced initial forward waypoint: ({forward[0]:.2f}, '
                f'{forward[1]:.2f}), forward_dot={dot:.2f}'
            )

        for y in y_values:
            row_options = []
            for row_start_x, row_end_x in ((x_min, x_max), (x_max, x_min)):
                start = (row_start_x, y)
                end = (row_end_x, y)
                connector, connector_cost, connector_relaxed = self._route_between(
                    current, start
                )
                row_path, row_cost, row_relaxed = self._route_between(start, end)
                row_options.append(
                    (
                        connector_cost + row_cost,
                        connector + row_path[1:],
                        end,
                        connector_relaxed + row_relaxed,
                    )
                )

            _, best_path, current, segment_relaxed = min(row_options, key=lambda v: v[0])
            relaxed_segments += segment_relaxed
            self._extend_waypoints(waypoints, best_path)

        self.get_logger().info(
            f'Slope-aware planner produced {len(waypoints)} waypoints; '
            f'relaxed segments={relaxed_segments}'
        )
        return waypoints

    def _generate_sparse_scan_path(self):
        waypoints = []
        current = (self.start_x, self.start_y)
        relaxed_segments = 0

        if self.force_initial_forward_waypoint:
            forward = self._initial_forward_waypoint()
            self._extend_waypoints(waypoints, [forward])
            current = forward
            dot = (
                (forward[0] - self.start_x) * math.cos(self.start_yaw)
                + (forward[1] - self.start_y) * math.sin(self.start_yaw)
            )
            self.get_logger().info(
                f'Forced initial forward waypoint: ({forward[0]:.2f}, '
                f'{forward[1]:.2f}), forward_dot={dot:.2f}'
            )

        scan_points = self._scan_viewpoints()
        scan_points = [
            (
                max(self.field_x_min, min(self.field_x_max, x)),
                max(self.field_y_min, min(self.field_y_max, y)),
            )
            for x, y in scan_points
        ]

        route_cache = {}

        def route_between_cached(a, b):
            key = (round(a[0], 3), round(a[1], 3), round(b[0], 3), round(b[1], 3))
            if key not in route_cache:
                route_cache[key] = self._route_between(a, b, allow_fallback=False)
            return route_cache[key]

        best = None
        for order in itertools.permutations(scan_points):
            cursor = current
            cost = 0.0
            relaxed = 0
            segments = []
            feasible = True
            for target in order:
                path, segment_cost, segment_relaxed = route_between_cached(cursor, target)
                if not path:
                    feasible = False
                    break
                cost += segment_cost
                relaxed += segment_relaxed
                segments.append(path)
                cursor = target
            if feasible and (best is None or cost < best[0]):
                best = (cost, relaxed, segments, order)

        if best is None:
            self.get_logger().warn('Sparse scan planner found no safe route; falling back to center scan only.')
            center_path, _, _ = self._route_between(current, (0.0, 0.0), allow_fallback=False)
            if center_path:
                self._extend_waypoints(waypoints, center_path)
                return waypoints
            self.get_logger().warn('No safe center route; falling back to coverage rows.')
            strategy = self.coverage_strategy
            self.coverage_strategy = 'rows'
            try:
                return self._generate_slope_aware_coverage_path()
            finally:
                self.coverage_strategy = strategy

        cost, relaxed_segments, segments, order = best
        self._scan_targets = list(order)
        for segment in segments:
            self._extend_waypoints(waypoints, segment)

        length = self._path_length(waypoints)
        self.get_logger().info(
            f'Sparse scan planner produced {len(waypoints)} waypoints, '
            f'length={length:.1f} m, scan_points={len(scan_points)}, '
            f'relaxed segments={relaxed_segments}, objective_cost={cost:.1f}'
        )
        return waypoints

    def _scan_viewpoints(self):
        if self.scan_pattern == 'four_intersections':
            x_step = (self.field_x_max - self.field_x_min) / 3.0
            y_step = (self.field_y_max - self.field_y_min) / 3.0
            x_values = [self.field_x_min + x_step, self.field_x_min + 2.0 * x_step]
            y_values = [self.field_y_min + y_step, self.field_y_min + 2.0 * y_step]
            return [(x, y) for y in y_values for x in x_values]

        extent = min(
            self.scan_extent,
            max(0.0, self.field_x_max - self.field_x_min) * 0.45,
            max(0.0, self.field_y_max - self.field_y_min) * 0.45,
        )
        return [
            (0.0, 0.0),
            (-extent, 0.0),
            (extent, 0.0),
            (0.0, -extent),
            (0.0, extent),
        ]

    def _path_length(self, points):
        return sum(
            math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
            for i in range(1, len(points))
        )

    def _extend_waypoints(self, waypoints, points):
        for point in points:
            if not waypoints or math.hypot(
                point[0] - waypoints[-1][0], point[1] - waypoints[-1][1]
            ) > self.path_resolution * 0.5:
                waypoints.append([point[0], point[1]])

    def _route_between(self, start, goal, allow_fallback=True):
        attempts = [
            (self.max_route_slope, self.max_downhill_slope),
            (self.max_route_slope + 0.04, self.max_downhill_slope + 0.04),
            (self.max_route_slope + 0.08, self.max_downhill_slope + 0.08),
            (self.max_route_slope + 0.14, self.max_downhill_slope + 0.14),
        ]
        for idx, (max_slope, max_downhill) in enumerate(attempts):
            path, cost = self._astar_route(start, goal, max_slope, max_downhill)
            if path:
                return path, cost, 1 if idx else 0

        self.get_logger().warn(
            f'No slope-safe route from {start} to {goal}; falling back to straight segment.'
        )
        if not allow_fallback:
            return None, float('inf'), 1
        path = []
        self._append_line(path, start[0], start[1], goal[0], goal[1])
        return path, 1e6 + math.hypot(goal[0] - start[0], goal[1] - start[1]), 1

    def _astar_route(self, start, goal, max_slope, max_downhill):
        xs = self._terrain['xs']
        ys = self._terrain['ys']
        start_idx = self._nearest_feasible_terrain_index(start)
        goal_idx = self._nearest_feasible_terrain_index(goal)
        if start_idx is None or goal_idx is None:
            return None, float('inf')
        if start_idx == goal_idx:
            return [self._grid_point(start_idx)], 0.0

        open_heap = [(0.0, start_idx)]
        came_from = {}
        best_cost = {start_idx: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal_idx:
                return self._reconstruct_route(came_from, current), best_cost[current]

            for nxt in self._terrain_neighbors(current):
                x0, y0 = xs[current[0]], ys[current[1]]
                x1, y1 = xs[nxt[0]], ys[nxt[1]]
                if not self._point_in_field(x1, y1):
                    continue
                step_cost = self._terrain_edge_cost(
                    current, nxt, max_slope, max_downhill
                )
                if step_cost is None:
                    continue
                new_cost = best_cost[current] + step_cost
                if new_cost < best_cost.get(nxt, float('inf')):
                    best_cost[nxt] = new_cost
                    priority = new_cost + math.hypot(
                        xs[goal_idx[0]] - x1, ys[goal_idx[1]] - y1
                    )
                    came_from[nxt] = current
                    heapq.heappush(open_heap, (priority, nxt))

        return None, float('inf')

    def _nearest_feasible_terrain_index(self, point):
        candidates = []
        for ix, x in enumerate(self._terrain['xs']):
            if not (self.field_x_min - 1e-6 <= x <= self.field_x_max + 1e-6):
                continue
            for iy, y in enumerate(self._terrain['ys']):
                if not (self.field_y_min - 1e-6 <= y <= self.field_y_max + 1e-6):
                    continue
                node = (ix, iy)
                if self._footprint_cost_at(node, 0.0) is None:
                    continue
                candidates.append((math.hypot(x - point[0], y - point[1]), node))
        if not candidates:
            return None
        return min(candidates, key=lambda v: v[0])[1]

    def _nearest_terrain_index(self, point):
        return (
            self._nearest_index(
                self._terrain['xs'], point[0], self.field_x_min, self.field_x_max
            ),
            self._nearest_index(
                self._terrain['ys'], point[1], self.field_y_min, self.field_y_max
            ),
        )

    def _nearest_index(self, values, value, min_value, max_value):
        candidates = [
            i for i, sample in enumerate(values)
            if min_value - 1e-6 <= sample <= max_value + 1e-6
        ]
        if not candidates:
            candidates = range(len(values))
        idx = min(candidates, key=lambda i: abs(values[i] - value))
        return idx

    def _terrain_neighbors(self, node):
        ix, iy = node
        max_ix = len(self._terrain['xs']) - 1
        max_iy = len(self._terrain['ys']) - 1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = ix + dx
                ny = iy + dy
                if 0 <= nx <= max_ix and 0 <= ny <= max_iy:
                    yield (nx, ny)

    def _terrain_height_at_index(self, node):
        x = self._terrain['xs'][node[0]]
        y = self._terrain['ys'][node[1]]
        return self._terrain['heights'][(x, y)]

    def _height_at(self, x, y):
        if self._terrain is None:
            return None
        return self._terrain_height_at_index(self._nearest_terrain_index((x, y)))

    def _terrain_edge_cost(self, node, nxt, max_slope, max_downhill):
        x0, y0 = self._grid_point(node)
        x1, y1 = self._grid_point(nxt)
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist <= 1e-6:
            return None
        dz = self._terrain_height_at_index(nxt) - self._terrain_height_at_index(node)
        grade = dz / dist
        downhill = max(0.0, -grade)
        if abs(grade) > max_slope or downhill > max_downhill:
            return None
        yaw = math.atan2(y1 - y0, x1 - x0)
        current_footprint_cost = self._footprint_cost_at(node, yaw)
        next_footprint_cost = self._footprint_cost_at(nxt, yaw)
        if current_footprint_cost is None or next_footprint_cost is None:
            return None
        return dist * (
            1.0
            + self.slope_cost_weight * abs(grade)
            + self.downhill_cost_weight * downhill
            + current_footprint_cost
            + next_footprint_cost
        )

    def _footprint_cost_at(self, node, yaw):
        samples = self._footprint_sample_indices(node, yaw)
        if samples is None:
            return None
        heights = [self._terrain_height_at_index(sample) for sample in samples]
        height_delta = max(heights) - min(heights)
        if height_delta > self.max_footprint_height_delta:
            return None
        roughness = self._footprint_roughness(samples, heights)
        if roughness > self.max_footprint_roughness:
            return None

        bounds = self._footprint_bounds_at(self._grid_point(node), yaw, self.route_footprint_margin)
        boundary_dist = min(
            bounds[0] - self.safety_x_min,
            self.safety_x_max - bounds[1],
            bounds[2] - self.safety_y_min,
            self.safety_y_max - bounds[3],
        )
        if boundary_dist <= 0.0:
            return None

        step_cost = self.footprint_step_cost_weight * (height_delta + 4.0 * roughness)
        boundary_cost = self.boundary_cost_weight / max(0.2, boundary_dist)
        return step_cost + boundary_cost

    def _footprint_roughness(self, samples, heights):
        rows = []
        for node in samples:
            x, y = self._grid_point(node)
            rows.append((x, y, 1.0))
        coeff, *_ = np.linalg.lstsq(np.array(rows), np.array(heights), rcond=None)
        residuals = np.array(heights) - np.array(rows).dot(coeff)
        return float(np.max(np.abs(residuals)))

    def _footprint_sample_indices(self, node, yaw):
        cx, cy = self._grid_point(node)
        half_l = self.footprint_length / 2.0 + self.route_footprint_margin
        half_w = self.footprint_width / 2.0 + self.route_footprint_margin
        local_samples = [
            (0.0, 0.0),
            (half_l, half_w),
            (half_l, -half_w),
            (-half_l, half_w),
            (-half_l, -half_w),
            (half_l, 0.0),
            (-half_l, 0.0),
            (0.0, half_w),
            (0.0, -half_w),
        ]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        sample_nodes = []
        for lx, ly in local_samples:
            x = cx + lx * cos_yaw - ly * sin_yaw
            y = cy + lx * sin_yaw + ly * cos_yaw
            if not self._point_in_safety_bounds(x, y):
                return None
            sample_nodes.append(self._nearest_terrain_index_global((x, y)))
        return sample_nodes

    def _nearest_terrain_index_global(self, point):
        return (
            min(range(len(self._terrain['xs'])), key=lambda i: abs(self._terrain['xs'][i] - point[0])),
            min(range(len(self._terrain['ys'])), key=lambda i: abs(self._terrain['ys'][i] - point[1])),
        )

    def _point_in_safety_bounds(self, x, y):
        return (
            self.safety_x_min <= x <= self.safety_x_max
            and self.safety_y_min <= y <= self.safety_y_max
        )

    def _grid_point(self, node):
        return (self._terrain['xs'][node[0]], self._terrain['ys'][node[1]])

    def _point_in_field(self, x, y):
        return (
            self.field_x_min - 1e-6 <= x <= self.field_x_max + 1e-6
            and self.field_y_min - 1e-6 <= y <= self.field_y_max + 1e-6
        )

    def _footprint_bounds_at(self, center, yaw, margin):
        px, py = center
        half_l = self.footprint_length / 2.0 + margin
        half_w = self.footprint_width / 2.0 + margin
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        xs = []
        ys = []
        for lx in (-half_l, half_l):
            for ly in (-half_w, half_w):
                xs.append(px + lx * cos_yaw - ly * sin_yaw)
                ys.append(py + lx * sin_yaw + ly * cos_yaw)
        return min(xs), max(xs), min(ys), max(ys)

    def _reconstruct_route(self, came_from, current):
        route = [self._grid_point(current)]
        while current in came_from:
            current = came_from[current]
            route.append(self._grid_point(current))
        route.reverse()
        return route

    def _append_line(self, waypoints, x0, y0, x1, y1):
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(dist / self.path_resolution))
        for i in range(steps + 1):
            t = i / steps
            point = [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]
            if not waypoints or math.hypot(
                point[0] - waypoints[-1][0], point[1] - waypoints[-1][1]
            ) > self.path_resolution * 0.5:
                waypoints.append(point)

    def img_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([45, 255, 255])
        mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
        mask = mask[220:, :]
        moments = cv2.moments(mask)
        try:
            x_center = moments['m10'] / moments['m00']
            self.yaw_speed = (160 - x_center) * 0.08
        except ZeroDivisionError:
            pass

    def _pure_pursuit(self):
        """
        Pure Pursuit 控制器。
        大角度误差时低速转弯，避免在坡面上长时间原地打转。
        """
        if self._completed or self.current_wp_idx >= len(self.waypoints):
            if not self._completed:
                self.get_logger().info('Coverage complete!')
                self._completed = True
            return 0.0, 0.0

        self._advance_path_index()
        if self._completed:
            return 0.0, 0.0

        target = self._lookahead_target()
        dx = target[0] - self.pose[0]
        dy = target[1] - self.pose[1]
        dist = math.sqrt(dx * dx + dy * dy)

        target_angle = math.atan2(dy, dx)
        angle_error = target_angle - self.pose[2]
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        abs_err = abs(angle_error)
        sparse_mode = self.coverage_strategy == 'sparse_scan'
        speed_cap = min(self.linear_speed, self.sparse_linear_speed) if sparse_mode else self.linear_speed
        turn_speed = min(speed_cap, self.sparse_turn_linear_speed) if sparse_mode else min(self.min_linear_speed, self.linear_speed)
        angular_cap = min(self.max_angular_speed, self.sparse_max_angular_speed) if sparse_mode else self.max_angular_speed

        if abs_err > math.radians(100.0):
            angular = math.copysign(angular_cap, angle_error)
            if abs_err > math.radians(135.0):
                # Keep rolling while correcting heading. In-place turns on the
                # uneven tile terrain tend to dig the rover into contact edges.
                return turn_speed, angular * 0.55
            return turn_speed, angular

        curvature = 2.0 * math.sin(angle_error) / self.lookahead
        curvature = max(-1.0, min(1.0, curvature))
        if sparse_mode:
            turn_scale = max(0.35, 1.0 - 0.75 * min(1.0, abs_err / math.radians(90.0)))
            linear = max(self.sparse_turn_linear_speed, speed_cap * turn_scale)
            if dist < self.lookahead:
                linear = max(
                    self.sparse_turn_linear_speed,
                    linear * max(0.35, dist / max(1e-6, self.lookahead)),
                )
        else:
            linear = self.linear_speed
        angular = curvature * linear
        angular = max(-angular_cap, min(angular_cap, angular))
        return linear, angular

    def _spin_scan_command(self):
        msg = Twist()
        if self._completed:
            return msg

        yaw = self.pose[2]
        if self._spin_last_yaw is None:
            self._spin_last_yaw = yaw
            self.get_logger().info('Spin scan started.')

        delta = math.atan2(
            math.sin(yaw - self._spin_last_yaw),
            math.cos(yaw - self._spin_last_yaw),
        )
        self._spin_accum += abs(delta)
        self._spin_last_yaw = yaw

        target = 2.0 * math.pi * max(0.0, self.spin_scan_turns)
        if self._spin_accum >= target:
            self._completed = True
            self.get_logger().info(
                f'Spin scan complete: rotated {self._spin_accum:.2f} rad.'
            )
            return msg

        msg.angular.z = self.spin_scan_angular_speed
        return msg

    def _scan_target_spin_command(self):
        if (
            self.coverage_strategy != 'sparse_scan'
            or self.scan_spin_turns <= 0.0
            or self._scan_target_idx >= len(self._scan_targets)
        ):
            return None

        target = self._scan_targets[self._scan_target_idx]
        if math.hypot(target[0] - self.pose[0], target[1] - self.pose[1]) > self.waypoint_radius:
            return None

        msg = Twist()
        yaw = self.pose[2]
        if self._scan_spin_last_yaw is None:
            self._scan_spin_last_yaw = yaw
            self._scan_spin_accum = 0.0
            self.get_logger().info(
                f'Scan spin {self._scan_target_idx + 1}/{len(self._scan_targets)} '
                f'at ({target[0]:.2f}, {target[1]:.2f}) started.'
            )

        delta = math.atan2(
            math.sin(yaw - self._scan_spin_last_yaw),
            math.cos(yaw - self._scan_spin_last_yaw),
        )
        self._scan_spin_accum += abs(delta)
        self._scan_spin_last_yaw = yaw

        target_rotation = 2.0 * math.pi * max(0.0, self.scan_spin_turns)
        if self._scan_spin_accum >= target_rotation:
            self.get_logger().info(
                f'Scan spin {self._scan_target_idx + 1}/{len(self._scan_targets)} '
                f'complete: rotated {self._scan_spin_accum:.2f} rad.'
            )
            self._scan_target_idx += 1
            self._scan_spin_last_yaw = None
            self._scan_spin_accum = 0.0
            return msg

        msg.angular.z = self.scan_spin_angular_speed
        return msg

    def _startup_command(self):
        if (
            self.mode == 'line_follow'
            or self._completed
            or self._startup_done
            or self.startup_blend_distance <= 0.0
        ):
            if self.startup_blend_distance <= 0.0:
                self._startup_done = True
            return None

        px, py = self.pose[0], self.pose[1]
        if self._startup_origin is None:
            self._startup_origin = (px, py)

        traveled = math.hypot(px - self._startup_origin[0], py - self._startup_origin[1])
        if traveled >= self.startup_blend_distance:
            self._startup_done = True
            self.get_logger().info(
                f'Startup forward run complete: traveled {traveled:.2f} m.'
            )
            return None

        target = self.waypoints[min(self.current_wp_idx, len(self.waypoints) - 1)]
        target_angle = math.atan2(target[1] - py, target[0] - px)
        angle_error = math.atan2(
            math.sin(target_angle - self.pose[2]),
            math.cos(target_angle - self.pose[2]),
        )

        msg = Twist()
        msg.linear.x = min(self.startup_linear_speed, self.linear_speed)
        if traveled < self.startup_straight_distance:
            msg.angular.z = 0.0
        else:
            blend_span = max(
                1e-6, self.startup_blend_distance - self.startup_straight_distance
            )
            blend = max(0.0, min(1.0, (traveled - self.startup_straight_distance) / blend_span))
            cap = self.startup_max_angular_speed * blend
            msg.angular.z = max(-cap, min(cap, angle_error * 0.8))

        if not getattr(self, '_startup_logged', False):
            self.get_logger().info(
                f'Startup forward run: straight {self.startup_straight_distance:.2f} m, '
                f'blend until {self.startup_blend_distance:.2f} m.'
            )
            self._startup_logged = True
        return msg

    def _advance_path_index(self):
        while self.current_wp_idx < len(self.waypoints):
            wp = self.waypoints[self.current_wp_idx]
            if math.hypot(wp[0] - self.pose[0], wp[1] - self.pose[1]) >= self.waypoint_radius:
                break
            self.current_wp_idx += 1

        if self.current_wp_idx >= len(self.waypoints):
            if not self._completed:
                self.get_logger().info('Coverage complete!')
                self._completed = True
            return

        search_end = min(len(self.waypoints), self.current_wp_idx + self.path_progress_window)
        closest_idx = min(
            range(self.current_wp_idx, search_end),
            key=lambda i: math.hypot(
                self.waypoints[i][0] - self.pose[0],
                self.waypoints[i][1] - self.pose[1],
            ),
        )
        if closest_idx > self.current_wp_idx:
            self.current_wp_idx = closest_idx

    def _lookahead_target(self):
        target_idx = self.current_wp_idx
        while target_idx + 1 < len(self.waypoints):
            wp = self.waypoints[target_idx]
            if math.hypot(wp[0] - self.pose[0], wp[1] - self.pose[1]) >= self.lookahead:
                break
            target_idx += 1
        return self.waypoints[target_idx]

    def _boundary_distances(self):
        bounds = self._footprint_bounds()
        return {
            'x_min': bounds[0] - self.safety_x_min,
            'x_max': self.safety_x_max - bounds[1],
            'y_min': bounds[2] - self.safety_y_min,
            'y_max': self.safety_y_max - bounds[3],
        }

    def _footprint_bounds(self):
        """Axis-aligned world bounds for the rover footprint plus safety margin."""
        px, py, yaw = self.pose
        half_l = self.footprint_length / 2.0 + self.footprint_safety_margin
        half_w = self.footprint_width / 2.0 + self.footprint_safety_margin
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        xs = []
        ys = []
        for lx in (-half_l, half_l):
            for ly in (-half_w, half_w):
                xs.append(px + lx * cos_yaw - ly * sin_yaw)
                ys.append(py + lx * sin_yaw + ly * cos_yaw)
        return min(xs), max(xs), min(ys), max(ys)

    def _moving_toward_nearest_boundary(self, linear):
        if abs(linear) < 1e-6:
            return False

        distances = self._boundary_distances()
        nearest = min(distances, key=distances.get)
        vx = linear * math.cos(self.pose[2])
        vy = linear * math.sin(self.pose[2])

        if nearest == 'x_min':
            return vx < 0.0
        if nearest == 'x_max':
            return vx > 0.0
        if nearest == 'y_min':
            return vy < 0.0
        return vy > 0.0

    def _inward_boundary_heading(self):
        px, py = self.pose[0], self.pose[1]
        margin = max(self.boundary_stop_margin, self.field_guard_lookahead)
        target_x = max(
            self.safety_x_min + margin,
            min(self.safety_x_max - margin, px),
        )
        target_y = max(
            self.safety_y_min + margin,
            min(self.safety_y_max - margin, py),
        )
        return math.atan2(target_y - py, target_x - px)

    def _field_distances(self):
        """Footprint clearance to the requested work/scanning field."""
        bounds = self._footprint_bounds()
        return {
            'x_min': bounds[0] - self.field_x_min,
            'x_max': self.field_x_max - bounds[1],
            'y_min': bounds[2] - self.field_y_min,
            'y_max': self.field_y_max - bounds[3],
        }

    def _predicted_field_distances(self, linear):
        """Predict work-field clearance after one reaction horizon."""
        travel = max(
            0.0,
            min(
                max(0.0, self.work_guard_lookahead_dist),
                abs(linear) * max(0.0, self.work_guard_lookahead_time),
            ),
        )
        direction = 1.0 if linear >= 0.0 else -1.0
        px = self.pose[0] + direction * travel * math.cos(self.pose[2])
        py = self.pose[1] + direction * travel * math.sin(self.pose[2])
        bounds = self._footprint_bounds_at(
            (px, py), self.pose[2], self.footprint_safety_margin
        )
        return {
            'x_min': bounds[0] - self.field_x_min,
            'x_max': self.field_x_max - bounds[1],
            'y_min': bounds[2] - self.field_y_min,
            'y_max': self.field_y_max - bounds[3],
        }

    def _moving_outward_from_distances(self, distances, linear):
        if abs(linear) < 1e-6:
            return False
        nearest = min(distances, key=distances.get)
        vx = linear * math.cos(self.pose[2])
        vy = linear * math.sin(self.pose[2])
        if nearest == 'x_min':
            return vx < 0.0
        if nearest == 'x_max':
            return vx > 0.0
        if nearest == 'y_min':
            return vy < 0.0
        return vy > 0.0

    def _field_inward_heading(self):
        px, py = self.pose[0], self.pose[1]
        margin = max(
            self.work_guard_margin,
            self.footprint_length * 0.5 + self.footprint_safety_margin,
            self.footprint_width * 0.5 + self.footprint_safety_margin,
        )
        target_x = max(self.field_x_min + margin, min(self.field_x_max - margin, px))
        target_y = max(self.field_y_min + margin, min(self.field_y_max - margin, py))
        if math.hypot(target_x - px, target_y - py) > 1e-4:
            return math.atan2(target_y - py, target_x - px)

        distances = self._field_distances()
        nearest = min(distances, key=distances.get)
        if nearest == 'x_min':
            return 0.0
        if nearest == 'x_max':
            return math.pi
        if nearest == 'y_min':
            return math.pi / 2.0
        return -math.pi / 2.0

    def _field_violation(self):
        return max(-min(self._field_distances().values()), 0.0)

    def _field_guard_can_yield_to_recovery(self, msg, current_distances):
        """Allow traction recovery to keep its command when it is still safe.

        The work guard is predictive. During engine boost it may see a future
        footprint just inside the configured margin and overwrite the high
        command before the rover ever moves. We only yield when the current
        footprint is inside the work area and the real terrain safety boundary
        is not close; hard boundary protection still runs afterwards.
        """
        if not self.field_guard_yield_to_recovery:
            return False
        phase = self._stuck_recovery_phase
        recovery_phases = (
            'engine_reverse', 'engine_boost',
            'slope_retreat', 'slope_turn', 'slope_escape',
        )
        if phase not in recovery_phases:
            return False

        min_current = min(current_distances.values())
        min_safety = min(self._boundary_distances().values())
        if min_current <= 0.15 or min_safety <= max(0.0, self.boundary_stop_margin):
            return False

        moving_outward_now = self._moving_outward_from_distances(
            current_distances, msg.linear.x
        )
        if moving_outward_now and min_current <= max(0.0, self.work_guard_margin):
            return False
        return True

    def _apply_field_guard(self, msg):
        """Predictive virtual fence for the work area, independent of terrain safety."""
        if self._completed:
            self._field_guard_warned = False
            self._field_guard_recovery_override_warned = False
            return

        current_distances = self._field_distances()
        if self._field_guard_can_yield_to_recovery(msg, current_distances):
            self._field_guard_warned = False
            if not self._field_guard_recovery_override_warned:
                self.get_logger().warn(
                    'Work-area guard yielding to traction recovery: current '
                    f'clearance={min(current_distances.values()):.2f} m, '
                    f'phase={self._stuck_recovery_phase}, command '
                    f'v={msg.linear.x:.2f}, w={msg.angular.z:.2f}.'
                )
                self._field_guard_recovery_override_warned = True
            return
        self._field_guard_recovery_override_warned = False

        predicted_distances = self._predicted_field_distances(msg.linear.x)
        min_current = min(current_distances.values())
        min_predicted = min(predicted_distances.values())
        trigger_margin = max(0.0, self.work_guard_margin)
        moving_outward = (
            self._moving_outward_from_distances(current_distances, msg.linear.x)
            or self._moving_outward_from_distances(predicted_distances, msg.linear.x)
        )

        if (
            min_current > trigger_margin
            and min_predicted > trigger_margin
        ):
            self._field_guard_warned = False
            return

        if not moving_outward and min_current > 0.0 and min_predicted > 0.0:
            self._field_guard_warned = False
            return

        heading = self._field_inward_heading()
        angle_error = math.atan2(
            math.sin(heading - self.pose[2]),
            math.cos(heading - self.pose[2]),
        )
        angular = max(
            -self.field_guard_max_angular_speed,
            min(self.field_guard_max_angular_speed, angle_error * 0.9),
        )

        if abs(angle_error) > math.radians(105.0) and msg.linear.x > 0.0:
            # The safe direction is behind the rover.  Reversing a little is
            # safer than continuing outward while waiting for yaw to change.
            msg.linear.x = -max(0.05, self.work_guard_reverse_speed)
        elif msg.linear.x >= 0.0:
            speed_scale = max(0.25, min(1.0, max(min_current, 0.0) / max(1e-6, trigger_margin)))
            msg.linear.x = min(msg.linear.x, max(0.08, self.work_guard_speed * speed_scale))
        else:
            msg.linear.x = max(msg.linear.x, -max(0.05, self.work_guard_reverse_speed))
        msg.angular.z = angular

        if not getattr(self, '_field_guard_warned', False):
            self.get_logger().warn(
                f'Work-area guard active: current clearance={min_current:.2f} m, '
                f'predicted clearance={min_predicted:.2f} m. Steering inside '
                f'[{self.field_x_min},{self.field_x_max}] x '
                f'[{self.field_y_min},{self.field_y_max}].'
            )
            self._field_guard_warned = True

    def _work_boundary_distances(self):
        x_min, x_max, y_min, y_max = self._active_work_bounds()
        bounds = self._footprint_bounds()
        return {
            'x_min': bounds[0] - x_min,
            'x_max': x_max - bounds[1],
            'y_min': bounds[2] - y_min,
            'y_max': y_max - bounds[3],
        }

    def _active_work_bounds(self):
        shrink = max(0.0, self.boundary_turn_shrink_per_turn)
        turn_offset = max(0, self._boundary_turn_completed_count) * shrink
        min_extent = max(0.0, self.boundary_turn_min_extent)

        x_half = max(
            min_extent,
            min(abs(self.field_x_min), abs(self.field_x_max)) - turn_offset,
        )
        y_half = max(
            min_extent,
            min(abs(self.field_y_min), abs(self.field_y_max)) - turn_offset,
        )
        return -x_half, x_half, -y_half, y_half

    def _work_boundary_turn_trigger(self):
        distances = self._work_boundary_distances()
        margin = max(0.0, self.boundary_turn_trigger_margin)
        rearm_margin = max(margin, self.boundary_turn_rearm_margin)
        for side, dist in distances.items():
            if dist > rearm_margin:
                self._boundary_turn_latched.discard(side)

        vx = math.cos(self.pose[2])
        vy = math.sin(self.pose[2])

        candidates = []
        if (
            'x_min' not in self._boundary_turn_latched
            and distances['x_min'] <= margin
            and vx < -0.05
        ):
            candidates.append(('x_min', distances['x_min']))
        if (
            'x_max' not in self._boundary_turn_latched
            and distances['x_max'] <= margin
            and vx > 0.05
        ):
            candidates.append(('x_max', distances['x_max']))
        if (
            'y_min' not in self._boundary_turn_latched
            and distances['y_min'] <= margin
            and vy < -0.05
        ):
            candidates.append(('y_min', distances['y_min']))
        if (
            'y_max' not in self._boundary_turn_latched
            and distances['y_max'] <= margin
            and vy > 0.05
        ):
            candidates.append(('y_max', distances['y_max']))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1])[0]

    def _boundary_turn_command(self):
        msg = Twist()
        if self._completed:
            return msg

        if not self._boundary_turning:
            trigger = self._work_boundary_turn_trigger()
            if trigger is None:
                msg.linear.x = self.linear_speed
                return msg

            distances = self._work_boundary_distances()
            x_min, x_max, y_min, y_max = self._active_work_bounds()
            self._boundary_turning = True
            self._boundary_turn_last_yaw = self.pose[2]
            self._boundary_turn_accum = 0.0
            self._boundary_turn_count += 1
            self._boundary_turn_latched.add(trigger)
            now = self._now_seconds()
            self._boundary_turn_start_time = now
            self._boundary_turn_last_update_time = now
            self._boundary_turn_last_progress_time = now
            self._boundary_turn_last_progress_accum = 0.0
            self._boundary_turn_slow_warned = False
            self.get_logger().info(
                f'Work boundary {trigger} touched '
                f'({distances[trigger]:.2f} m). Starting left arc turn '
                f'{self._boundary_turn_count}; active field '
                f'[{x_min:.1f},{x_max:.1f}] x [{y_min:.1f},{y_max:.1f}].'
            )

        yaw = self.pose[2]
        if self._boundary_turn_last_yaw is None:
            self._boundary_turn_last_yaw = yaw
        delta = math.atan2(
            math.sin(yaw - self._boundary_turn_last_yaw),
            math.cos(yaw - self._boundary_turn_last_yaw),
        )
        yaw_progress = max(0.0, delta)
        now = self._now_seconds()
        dt = 0.0
        if self._boundary_turn_last_update_time is not None:
            dt = max(0.0, now - self._boundary_turn_last_update_time)
        yaw_rate = yaw_progress / dt if dt > 1e-3 else 0.0
        self._boundary_turn_accum += yaw_progress
        self._boundary_turn_last_yaw = yaw
        self._boundary_turn_last_update_time = now
        if self._boundary_turn_accum - self._boundary_turn_last_progress_accum > 0.01:
            self._boundary_turn_last_progress_time = now
            self._boundary_turn_last_progress_accum = self._boundary_turn_accum

        target_angle = max(0.0, self.boundary_turn_angle)
        completion_angle = max(
            0.0,
            target_angle - max(0.0, self.boundary_turn_completion_tolerance),
        )
        stalled_near_target = (
            target_angle - self._boundary_turn_accum
            <= max(0.08, self.boundary_turn_completion_tolerance * 2.0)
            and self._boundary_turn_last_progress_time is not None
            and now - self._boundary_turn_last_progress_time
            >= max(0.0, self.boundary_turn_stall_timeout)
        )
        if self._boundary_turn_accum >= completion_angle or stalled_near_target:
            self._boundary_turning = False
            self._boundary_turn_last_yaw = None
            self._boundary_turn_accum = 0.0
            self._boundary_turn_last_progress_time = None
            self._boundary_turn_last_progress_accum = 0.0
            self._boundary_turn_start_time = None
            self._boundary_turn_last_update_time = None
            self._boundary_turn_slow_warned = False
            self._boundary_turn_completed_count = max(
                self._boundary_turn_completed_count,
                self._boundary_turn_count,
            )
            msg.linear.x = self.linear_speed
            self.get_logger().info(
                f'Left arc turn {self._boundary_turn_count} complete.'
            )
            return msg

        turn_elapsed = (
            now - self._boundary_turn_start_time
            if self._boundary_turn_start_time is not None
            else 0.0
        )
        slow_yaw = (
            turn_elapsed >= max(0.0, self.boundary_turn_rate_grace)
            and yaw_rate < max(0.0, self.boundary_turn_min_yaw_rate)
            and min(self._boundary_distances().values())
            <= max(0.0, self.boundary_turn_slow_safety_margin)
        )
        if slow_yaw:
            msg.linear.x = max(0.0, self.boundary_turn_slow_linear_speed)
            if not self._boundary_turn_slow_warned:
                self.get_logger().warn(
                    f'Left arc turn yaw rate low ({yaw_rate:.2f} rad/s); '
                    f'limiting linear speed to {msg.linear.x:.2f} m/s.'
                )
                self._boundary_turn_slow_warned = True
        else:
            msg.linear.x = max(0.05, self.boundary_turn_linear_speed)
            self._boundary_turn_slow_warned = False
        msg.angular.z = abs(self.boundary_turn_angular_speed)
        return msg

    def _apply_boundary_guard(self, msg):
        """提前把车辆引导回安全区，避免靠惯性冲出地形。"""
        min_dist = min(self._boundary_distances().values())
        moving_outward = self._moving_toward_nearest_boundary(msg.linear.x)

        if min_dist <= self.boundary_stop_margin and (moving_outward or min_dist <= 0.0):
            heading = self._inward_boundary_heading()
            angle_error = math.atan2(
                math.sin(heading - self.pose[2]),
                math.cos(heading - self.pose[2]),
            )
            angular_cap = min(self.max_angular_speed, max(self.sparse_max_angular_speed, 0.75))
            guide_speed = max(0.12, self.boundary_escape_speed)
            if abs(angle_error) > math.radians(100.0):
                guide_speed = max(0.12, guide_speed * 0.5)
            msg.linear.x = max(0.12, min(max(0.0, msg.linear.x), guide_speed))
            msg.angular.z = max(
                -angular_cap,
                min(angular_cap, angle_error * 0.8),
            )
            if not getattr(self, '_boundary_brake_warned', False):
                self.get_logger().warn(
                    f'Near safety boundary ({min_dist:.2f} m). '
                    'Steering back into safe area.'
                )
                self._boundary_brake_warned = True
            return

        self._boundary_brake_warned = False

        if 0.0 < min_dist < min(self.brake_dist, self.boundary_stop_margin) and moving_outward:
            scale = max(0.6, min(1.0, min_dist / max(1e-6, self.boundary_stop_margin)))
            msg.linear.x *= scale

    def _apply_slope_speed_limit(self, msg):
        if self._terrain is None or msg.linear.x <= 0.05 or self._completed:
            return

        target = self._lookahead_target()
        current_z = self._height_at(self.pose[0], self.pose[1])
        target_z = self._height_at(target[0], target[1])
        if current_z is None or target_z is None:
            return

        dist = math.hypot(target[0] - self.pose[0], target[1] - self.pose[1])
        if dist <= 1e-6:
            return

        grade = (target_z - current_z) / dist
        downhill = max(0.0, -grade)
        if downhill <= self.downhill_slow_slope:
            self._downhill_limit_warned = False
            return

        denom = max(1e-6, self.max_downhill_slope - self.downhill_slow_slope)
        severity = max(0.0, min(1.0, (downhill - self.downhill_slow_slope) / denom))
        target_speed = max(
            self.slope_min_linear_speed,
            msg.linear.x * (1.0 - 0.75 * severity),
        )
        if target_speed < msg.linear.x:
            msg.linear.x = target_speed
            msg.angular.z *= max(0.5, 1.0 - 0.35 * severity)
            if not getattr(self, '_downhill_limit_warned', False):
                self.get_logger().warn(
                    f'Downhill grade {downhill:.2f}; limiting speed to '
                    f'{msg.linear.x:.2f} m/s.'
                )
                self._downhill_limit_warned = True

    def _reset_stuck_watchdog(self):
        self._stuck_ref_pose = tuple(self.pose)
        self._stuck_ref_time = self._now_seconds()

    def _grade_ahead(self, distance=None):
        """Estimate terrain grade in the current heading, if a terrain map is loaded."""
        if self._terrain is None:
            return None
        distance = max(0.25, distance if distance is not None else self.lookahead)
        x0, y0 = self.pose[0], self.pose[1]
        x1 = x0 + distance * math.cos(self.pose[2])
        y1 = y0 + distance * math.sin(self.pose[2])
        z0 = self._height_at(x0, y0)
        z1 = self._height_at(x1, y1)
        if z0 is None or z1 is None:
            return None
        return (z1 - z0) / distance

    def _near_guarded_boundary(self, clearance=None):
        margin = max(0.0, clearance if clearance is not None else self.engine_boost_boundary_clearance)
        try:
            return min(self._field_distances().values()) <= margin or min(self._boundary_distances().values()) <= margin
        except Exception:
            return False

    def _slope_stuck_score(self, commanded_linear):
        """Return (is_slope_stall, reason) using pitch, terrain grade and observed speed."""
        if self._now_seconds() < self._slope_escape_cooldown_until:
            return False, 'slope escape cooldown'
        if commanded_linear <= max(0.05, self.slope_stuck_min_command_speed):
            return False, 'command too small'
        if self._near_guarded_boundary(self.engine_boost_boundary_clearance):
            return False, 'near boundary'

        pitch_abs = abs(self.pitch)
        grade = self._grade_ahead()
        grade_abs = abs(grade) if grade is not None else 0.0
        observed = abs(self.observed_forward_speed)
        slope_like = (
            pitch_abs >= max(0.0, self.slope_stuck_pitch_threshold)
            or grade_abs >= max(0.0, self.slope_stuck_grade_threshold)
        )
        low_speed = observed <= max(0.0, self.slope_stuck_max_observed_speed)
        if slope_like and low_speed:
            detail = f'pitch={pitch_abs:.2f} rad'
            if grade is not None:
                detail += f', grade={grade:.2f}'
            detail += f', observed_forward={observed:.3f} m/s'
            return True, detail
        return False, f'pitch={pitch_abs:.2f}, grade={grade_abs:.2f}, observed_forward={observed:.3f}'

    def _classify_stuck(self, msg, translation, yaw_delta, elapsed):
        if self._near_guarded_boundary(self.engine_boost_boundary_clearance):
            return 'boundary_guard', 'too close to work/safety boundary for engine boost'

        slope_stall, slope_reason = self._slope_stuck_score(msg.linear.x)
        if self.engine_boost_enabled and slope_stall and elapsed >= max(0.1, self.slope_stuck_timeout):
            return 'slope_climb', slope_reason

        if abs(msg.angular.z) > 0.08 and yaw_delta < self.stuck_min_yaw_progress:
            return 'turn_stall', f'yaw changed only {yaw_delta:.3f} rad'

        if abs(msg.linear.x) > 0.08 and translation < self.stuck_min_progress:
            return 'traction_stall', f'translation only {translation:.3f} m'

        return 'unknown_stall', f'translation={translation:.3f} m, yaw={yaw_delta:.3f} rad'

    def _choose_stuck_turn_direction(self):
        # Prefer a turn that points toward the work-field interior.  Away from
        # the boundary, alternate directions so repeated recoveries do not dig
        # the same wheel into the same tile edge.
        heading = self._field_inward_heading()
        angle_error = math.atan2(
            math.sin(heading - self.pose[2]),
            math.cos(heading - self.pose[2]),
        )
        if abs(angle_error) > math.radians(8.0):
            return 1.0 if angle_error > 0.0 else -1.0
        return 1.0 if self._stuck_recovery_count % 2 == 0 else -1.0

    def _start_stuck_recovery(self, kind, reason, translation, yaw_delta, elapsed):
        now = self._now_seconds()
        self._stuck_recovery_count += 1
        self._stuck_recovery_kind = kind
        self._stuck_recovery_start_time = now
        self._stuck_recovery_turn_dir = self._choose_stuck_turn_direction()

        if kind == 'slope_climb' and self._engine_boost_attempts < max(0, int(self.engine_boost_max_attempts)):
            self._engine_boost_attempts += 1
            self._stuck_recovery_phase = 'engine_reverse'
            self._stuck_recovery_until = now + max(0.0, self.engine_boost_reverse_time)
            self._stuck_recovery_pending_skip = False
            self.get_logger().warn(
                f'Slope-climb stall {self._engine_boost_attempts}/'
                f'{int(self.engine_boost_max_attempts)}: {reason}. '
                'Using engine boost: unload wheels, then low-speed high-command wiggle climb.'
            )
            return

        # After boost attempts fail, do not keep pushing into the same slope.
        # Retreat farther, rotate away from the uphill heading, and move out of
        # the local trap before normal scanning resumes.
        if kind == 'slope_climb':
            self._stuck_recovery_phase = 'slope_retreat'
            self._stuck_recovery_until = now + max(0.0, self.slope_escape_reverse_time)
            self._stuck_recovery_pending_skip = True
            self.get_logger().warn(
                f'Slope escape {self._stuck_recovery_count}: boost failed after '
                f'{int(self.engine_boost_max_attempts)} attempt(s); pose changed only '
                f'{translation:.3f} m and {yaw_delta:.3f} rad in {elapsed:.1f}s. '
                'Retreating, rotating away from the slope, and bypassing this heading.'
            )
            return

        # General fallback for non-slope stalls.
        self._stuck_recovery_phase = 'reverse'
        self._stuck_recovery_until = now + max(0.0, self.stuck_reverse_time)
        self._stuck_recovery_pending_skip = kind not in ('boundary_guard',)
        self.get_logger().warn(
            f'Stuck recovery {self._stuck_recovery_count} [{kind}]: pose changed only '
            f'{translation:.3f} m and {yaw_delta:.3f} rad in {elapsed:.1f}s; '
            f'{reason}. Backing out and taking a rolling escape arc.'
        )

    def _finish_stuck_recovery(self):
        finishing_kind = self._stuck_recovery_kind
        finishing_phase = self._stuck_recovery_phase
        if finishing_kind == 'slope_climb' or finishing_phase in ('slope_retreat', 'slope_turn', 'slope_escape'):
            self._slope_escape_cooldown_until = self._now_seconds() + max(0.0, self.slope_escape_cooldown)
            self._engine_boost_attempts = 0
        if self._stuck_recovery_pending_skip:
            if self.mode not in ('line_follow', 'boundary_turn', 'spin_scan') and hasattr(self, 'waypoints'):
                old_idx = self.current_wp_idx
                self.current_wp_idx = min(
                    len(self.waypoints),
                    self.current_wp_idx + max(0, int(self.stuck_skip_waypoints)),
                )
                if self.current_wp_idx != old_idx:
                    self.get_logger().warn(
                        f'Stuck recovery skipped waypoints {old_idx}->{self.current_wp_idx}.'
                    )
            elif self.mode == 'boundary_turn':
                # Drop the old arc-turn state; the next timer tick will either
                # re-arm a cleaner turn or continue straight if already safe.
                self._boundary_turning = False
                self._boundary_turn_last_yaw = None
                self._boundary_turn_accum = 0.0
                self._boundary_turn_last_progress_time = None
                self._boundary_turn_last_progress_accum = 0.0
                self._boundary_turn_start_time = None
                self._boundary_turn_last_update_time = None

        self._stuck_recovery_phase = None
        self._stuck_recovery_until = None
        self._stuck_recovery_pending_skip = False
        self._stuck_recovery_kind = None
        self._stuck_recovery_start_time = None
        self._recovery_last_phase_logged = None
        self._field_guard_recovery_override_warned = False
        self._reset_stuck_watchdog()

    def _stuck_recovery_command(self):
        now = self._now_seconds()
        if self._stuck_recovery_phase == 'engine_reverse' and now >= self._stuck_recovery_until:
            self._stuck_recovery_phase = 'engine_boost'
            self._stuck_recovery_start_time = now
            self._stuck_recovery_until = now + max(0.0, self.engine_boost_time)
        elif self._stuck_recovery_phase == 'engine_boost' and now >= self._stuck_recovery_until:
            self._finish_stuck_recovery()
            return None
        elif self._stuck_recovery_phase == 'slope_retreat' and now >= self._stuck_recovery_until:
            self._stuck_recovery_phase = 'slope_turn'
            self._stuck_recovery_start_time = now
            self._stuck_recovery_until = now + max(0.0, self.slope_escape_turn_time)
        elif self._stuck_recovery_phase == 'slope_turn' and now >= self._stuck_recovery_until:
            self._stuck_recovery_phase = 'slope_escape'
            self._stuck_recovery_start_time = now
            self._stuck_recovery_until = now + max(0.0, self.slope_escape_forward_time)
        elif self._stuck_recovery_phase == 'slope_escape' and now >= self._stuck_recovery_until:
            self._finish_stuck_recovery()
            return None
        elif self._stuck_recovery_phase == 'reverse' and now >= self._stuck_recovery_until:
            self._stuck_recovery_phase = 'turn'
            self._stuck_recovery_until = now + max(0.0, self.stuck_turn_time)
        elif self._stuck_recovery_phase == 'turn' and now >= self._stuck_recovery_until:
            self._finish_stuck_recovery()
            return None

        msg = Twist()
        turn_dir = 1.0 if self._stuck_recovery_turn_dir >= 0.0 else -1.0
        if self._stuck_recovery_phase == 'engine_reverse':
            # Briefly unload the tire/terrain contact before the climb pulse.
            msg.linear.x = -abs(self.engine_boost_reverse_speed)
            msg.angular.z = -0.25 * abs(self.engine_boost_wiggle_angular_speed) * turn_dir
        elif self._stuck_recovery_phase == 'engine_boost':
            # DiffDrive accepts velocity commands, not effort.  This is an
            # engine-equivalent command: high requested wheel speed with a slow
            # left-right wiggle, so contact points change instead of digging in.
            elapsed = max(0.0, now - (self._stuck_recovery_start_time or now))
            period = max(0.08, self.engine_boost_wiggle_period)
            wiggle_sign = 1.0 if int(elapsed / period) % 2 == 0 else -1.0
            # Ramp up over a short interval to avoid tipping the vehicle.
            ramp = min(1.0, elapsed / max(0.25, 0.35 * self.engine_boost_time))
            msg.linear.x = max(self.stuck_forward_speed, abs(self.engine_boost_speed) * ramp)
            msg.angular.z = wiggle_sign * turn_dir * abs(self.engine_boost_wiggle_angular_speed)
        elif self._stuck_recovery_phase == 'slope_retreat':
            msg.linear.x = -abs(self.slope_escape_reverse_speed)
            msg.angular.z = -0.35 * abs(self.slope_escape_angular_speed) * turn_dir
        elif self._stuck_recovery_phase == 'slope_turn':
            msg.linear.x = 0.08
            msg.angular.z = abs(self.slope_escape_angular_speed) * turn_dir
        elif self._stuck_recovery_phase == 'slope_escape':
            msg.linear.x = abs(self.slope_escape_forward_speed)
            msg.angular.z = 0.28 * abs(self.slope_escape_angular_speed) * turn_dir
        elif self._stuck_recovery_phase == 'reverse':
            msg.linear.x = -abs(self.stuck_reverse_speed)
            msg.angular.z = 0.35 * abs(self.stuck_turn_angular_speed) * turn_dir
        elif self._stuck_recovery_phase == 'turn':
            # A slow rolling arc works better on uneven collision tiles than a
            # pure in-place spin, which often pins the rover against an edge.
            msg.linear.x = abs(self.stuck_forward_speed)
            msg.angular.z = abs(self.stuck_turn_angular_speed) * turn_dir
        return msg

    def _apply_stuck_recovery(self, msg):
        if not self.stuck_detection_enabled or self._completed:
            return msg

        if self._stuck_recovery_phase is not None:
            recovery_msg = self._stuck_recovery_command()
            if recovery_msg is not None:
                phase = self._stuck_recovery_phase
                if self._recovery_last_phase_logged != phase:
                    self.get_logger().warn(
                        f'Recovery command phase={phase}: '
                        f'v={recovery_msg.linear.x:.2f} m/s, '
                        f'w={recovery_msg.angular.z:.2f} rad/s.'
                    )
                    self._recovery_last_phase_logged = phase
                return recovery_msg
            self._recovery_last_phase_logged = None
            return msg

        commanded_motion = abs(msg.linear.x) > 0.05 or abs(msg.angular.z) > 0.05
        if not commanded_motion:
            self._reset_stuck_watchdog()
            return msg

        now = self._now_seconds()
        if self._stuck_ref_pose is None or self._stuck_ref_time is None or now < self._stuck_ref_time:
            self._reset_stuck_watchdog()
            return msg

        elapsed = now - self._stuck_ref_time
        early_slope_check = elapsed >= max(0.1, self.slope_stuck_timeout)
        normal_watchdog = elapsed >= max(0.1, self.stuck_timeout)
        if not early_slope_check and not normal_watchdog:
            return msg

        dx = self.pose[0] - self._stuck_ref_pose[0]
        dy = self.pose[1] - self._stuck_ref_pose[1]
        translation = math.hypot(dx, dy)
        yaw_delta = abs(math.atan2(
            math.sin(self.pose[2] - self._stuck_ref_pose[2]),
            math.cos(self.pose[2] - self._stuck_ref_pose[2]),
        ))

        slope_stall, _ = self._slope_stuck_score(msg.linear.x)
        if not normal_watchdog and not slope_stall:
            return msg

        if translation >= self.stuck_min_progress or yaw_delta >= self.stuck_min_yaw_progress:
            self._reset_stuck_watchdog()
            # Successful movement means the previous engine boost worked; allow
            # future boosts again if another hill is reached later.
            self._engine_boost_attempts = 0
            return msg

        kind, reason = self._classify_stuck(msg, translation, yaw_delta, elapsed)
        self._start_stuck_recovery(kind, reason, translation, yaw_delta, elapsed)
        recovery_msg = self._stuck_recovery_command()
        return recovery_msg if recovery_msg is not None else msg

    def timer_callback(self):
        if not self.enabled:
            return

        # Safety boundary: warn immediately if outside safe zone, then let the
        # boundary guard steer the rover back in. Hard-stopping at the edge can
        # leave the rover stranded on uneven terrain.
        distances = self._boundary_distances()
        if min(distances.values()) < 0.0:
            if not getattr(self, '_boundary_warned', False):
                bounds = self._footprint_bounds()
                self.get_logger().error(
                    f'OUT OF BOUNDS: footprint x=[{bounds[0]:.2f},{bounds[1]:.2f}] '
                    f'y=[{bounds[2]:.2f},{bounds[3]:.2f}] exceeds safety '
                    f'[{self.safety_x_min},{self.safety_x_max}] x '
                    f'[{self.safety_y_min},{self.safety_y_max}]. Steering back in.'
                )
                self._boundary_warned = True
        else:
            self._boundary_warned = False

        msg = Twist()
        if self.mode == 'spin_scan':
            msg = self._spin_scan_command()
        elif self.mode == 'boundary_turn':
            msg = self._boundary_turn_command()
        elif self.mode == 'line_follow':
            msg.angular.z = self.yaw_speed
            msg.linear.x = self.linear_speed
        else:
            scan_spin_msg = self._scan_target_spin_command()
            startup_msg = None if scan_spin_msg is not None else self._startup_command()
            if scan_spin_msg is not None:
                msg = scan_spin_msg
            if startup_msg is not None:
                msg = startup_msg
            elif scan_spin_msg is None:
                linear, angular = self._pure_pursuit()
                msg.linear.x = linear
                msg.angular.z = angular
                self._apply_slope_speed_limit(msg)

        msg = self._apply_stuck_recovery(msg)

        # Work-field guard must also protect boundary_turn mode; otherwise the
        # rover can be several metres outside the requested work area before
        # the terrain-edge safety box intervenes.
        self._apply_field_guard(msg)
        self._apply_boundary_guard(msg)

        self.vel_pub.publish(msg)

        if self.mode != 'line_follow' and self._completed:
            self.timer.cancel()
            self.get_logger().info('Auto-drive timer cancelled, vehicle idle.')
            if self.auto_optimize and not getattr(self, '_pipeline_triggered', False):
                self._pipeline_triggered = True
                self._post_drive_pipeline()


    def _post_drive_pipeline(self):
        self.get_logger().info('Drive complete — auto-triggering optimize → reconstruct...')
        fut = self._opt_client.call_async(Trigger.Request())
        fut.add_done_callback(self._on_optimize_done)

    def _on_optimize_done(self, future):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f'Optimize call failed: {e}')
            return
        if resp.success:
            self.get_logger().info(f'Optimize OK: {resp.message}. Triggering reconstruct...')
            fut2 = self._mesh_client.call_async(Trigger.Request())
            fut2.add_done_callback(self._on_mesh_done)
        else:
            self.get_logger().error(f'Optimize failed: {resp.message}')

    def _on_mesh_done(self, future):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f'Reconstruct call failed: {e}')
            return
        self.get_logger().info(f'Reconstruct result: {resp.message}')


def main(args=None):
    rclpy.init(args=args)
    node = ADNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
