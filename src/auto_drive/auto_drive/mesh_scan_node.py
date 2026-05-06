#!/usr/bin/env python3
"""Traversability-first mesh scan route follower.

The rover mapping task is point-cloud coverage, not aggressive terrain traversal.
This node therefore treats the mesh as a height map, plans only through cells that
are likely traversable by the rover footprint, and uses progress monitoring to
skip/replan around segments that still fail in Gazebo.
"""
from __future__ import annotations

import csv
import heapq
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

Point2 = Tuple[float, float]
GridIndex = Tuple[int, int]


@dataclass
class RouteProjection:
    index: int
    distance: float
    progress: float
    point: Point2



@dataclass
class BlockedZone:
    x: float
    y: float
    radius: float
    until: float
    reason: str = ''


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class TerrainRoutePlanner:
    def __init__(
        self,
        height_source_path: str,
        height_source_label: str,
        grid_resolution: float,
        drive_x_min: float,
        drive_x_max: float,
        drive_y_min: float,
        drive_y_max: float,
        boundary_cost_weight: float,
        slope_cost_weight: float,
        max_edge_grade: float,
        max_edge_height_delta: float,
        strict_grade_limit: bool = True,
        footprint_length: float = 0.70,
        footprint_width: float = 0.57,
        footprint_margin: float = 0.12,
        max_footprint_height_delta: float = 0.22,
        max_footprint_roughness: float = 0.065,
    ):
        self.height_source_path = height_source_path
        self.height_source_label = height_source_label
        self.grid_resolution = grid_resolution
        self.drive_x_min = drive_x_min
        self.drive_x_max = drive_x_max
        self.drive_y_min = drive_y_min
        self.drive_y_max = drive_y_max
        self.boundary_cost_weight = boundary_cost_weight
        self.slope_cost_weight = slope_cost_weight
        self.max_edge_grade = max_edge_grade
        self.max_edge_height_delta = max_edge_height_delta
        self.strict_grade_limit = strict_grade_limit
        self.footprint_length = footprint_length
        self.footprint_width = footprint_width
        self.footprint_margin = footprint_margin
        self.max_footprint_height_delta = max_footprint_height_delta
        self.max_footprint_roughness = max_footprint_roughness
        self.mesh_xs, self.mesh_ys, self.mesh_z = self._load_height_grid(height_source_path)
        self.grid_xs = np.arange(drive_x_min, drive_x_max + 0.5 * grid_resolution, grid_resolution)
        self.grid_ys = np.arange(drive_y_min, drive_y_max + 0.5 * grid_resolution, grid_resolution)
        self.grid_heights = np.array(
            [[self.height_at(float(x), float(y)) for x in self.grid_xs] for y in self.grid_ys],
            dtype=float,
        )

    def _load_height_grid(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if path.endswith('.sdf'):
            return self._load_sdf_tile_height_grid(path)
        return self._load_obj_height_grid(path)

    @staticmethod
    def _load_sdf_tile_height_grid(sdf_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load legacy box-tile collision heights from model.sdf.

        This is a fallback for old terrain models.  It deliberately uses the
        collision boxes' top surfaces, not the visual mesh, because Gazebo's
        physics engine drives on collision geometry.
        """
        text = open(sdf_path, 'r', encoding='utf-8', errors='ignore').read()
        pattern = re.compile(
            r'<collision name="tile_(\d+)_(\d+)">\s*'
            r'<pose>([^<]+)</pose>\s*'
            r'<geometry>\s*<box><size>([^<]+)</size></box>',
            re.S,
        )
        records: List[Tuple[int, int, float, float, float]] = []
        for match in pattern.finditer(text):
            ix = int(match.group(1))
            iy = int(match.group(2))
            pose = [float(v) for v in match.group(3).split()]
            size = [float(v) for v in match.group(4).split()]
            x, y, z_mid = pose[:3]
            height = size[2]
            records.append((ix, iy, x, y, z_mid + 0.5 * height))
        if not records:
            raise RuntimeError(f'No tile collision boxes found in SDF: {sdf_path}')
        max_ix = max(r[0] for r in records)
        max_iy = max(r[1] for r in records)
        xs = np.zeros(max_ix + 1, dtype=float)
        ys = np.zeros(max_iy + 1, dtype=float)
        z = np.zeros((max_iy + 1, max_ix + 1), dtype=float)
        for ix, iy, x, y, top in records:
            xs[ix] = x
            ys[iy] = y
            z[iy, ix] = top
        return xs, ys, z

    @staticmethod
    def _load_obj_height_grid(obj_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertices: List[Tuple[float, float, float]] = []
        with open(obj_path, 'r', encoding='utf-8', errors='ignore') as handle:
            for line in handle:
                if line.startswith('v '):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        if not vertices:
            raise RuntimeError(f'No vertices found in terrain OBJ: {obj_path}')
        xs = np.array(sorted({v[0] for v in vertices}), dtype=float)
        ys = np.array(sorted({v[1] for v in vertices}), dtype=float)
        xi = {float(x): i for i, x in enumerate(xs)}
        yi = {float(y): i for i, y in enumerate(ys)}
        z = np.zeros((len(ys), len(xs)), dtype=float)
        for x, y, h in vertices:
            z[yi[float(y)], xi[float(x)]] = h
        return xs, ys, z

    def height_at(self, x: float, y: float) -> float:
        xs, ys, z = self.mesh_xs, self.mesh_ys, self.mesh_z
        x = float(np.clip(x, xs[0], xs[-1]))
        y = float(np.clip(y, ys[0], ys[-1]))
        ix = max(0, min(int(np.searchsorted(xs, x) - 1), len(xs) - 2))
        iy = max(0, min(int(np.searchsorted(ys, y) - 1), len(ys) - 2))
        x0, x1 = xs[ix], xs[ix + 1]
        y0, y1 = ys[iy], ys[iy + 1]
        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        z00, z10 = z[iy, ix], z[iy, ix + 1]
        z01, z11 = z[iy + 1, ix], z[iy + 1, ix + 1]
        return float((1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10 + (1 - tx) * ty * z01 + tx * ty * z11)

    def index_to_xy(self, idx: GridIndex) -> Point2:
        ix, iy = idx
        return float(self.grid_xs[ix]), float(self.grid_ys[iy])

    def xy_to_index(self, xy: Point2) -> GridIndex:
        x, y = xy
        ix = int(round((x - self.grid_xs[0]) / self.grid_resolution))
        iy = int(round((y - self.grid_ys[0]) / self.grid_resolution))
        return max(0, min(ix, len(self.grid_xs) - 1)), max(0, min(iy, len(self.grid_ys) - 1))

    def _inside_index(self, idx: GridIndex) -> bool:
        ix, iy = idx
        return 0 <= ix < len(self.grid_xs) and 0 <= iy < len(self.grid_ys)

    def boundary_clearance(self, x: float, y: float) -> float:
        return min(x - self.drive_x_min, self.drive_x_max - x, y - self.drive_y_min, self.drive_y_max - y)

    def _boundary_penalty(self, x: float, y: float) -> float:
        return max(0.0, 1.8 - self.boundary_clearance(x, y)) ** 2

    def footprint_stats(self, x: float, y: float) -> Tuple[float, float]:
        # Conservative axis-aligned footprint check.  It intentionally treats the
        # footprint as wider than the physical body so the planned path does not
        # rely on a fragile heading over a rough collision patch.
        half_l = 0.5 * self.footprint_length + self.footprint_margin
        half_w = 0.5 * self.footprint_width + self.footprint_margin
        sample_points = [
            (x, y),
            (x - half_l, y - half_w), (x - half_l, y + half_w),
            (x + half_l, y - half_w), (x + half_l, y + half_w),
            (x - half_l, y), (x + half_l, y), (x, y - half_w), (x, y + half_w),
        ]
        heights = [self.height_at(px, py) for px, py in sample_points]
        height_delta = max(heights) - min(heights)
        mean_height = sum(heights) / len(heights)
        roughness = math.sqrt(sum((h - mean_height) ** 2 for h in heights) / len(heights))
        return float(height_delta), float(roughness)

    def is_traversable_xy(self, x: float, y: float, blocked_zones: Optional[List[BlockedZone]] = None) -> bool:
        if self.boundary_clearance(x, y) < 0.80:
            return False
        height_delta, roughness = self.footprint_stats(x, y)
        if height_delta > self.max_footprint_height_delta or roughness > self.max_footprint_roughness:
            return False
        if blocked_zones:
            for zone in blocked_zones:
                if math.hypot(x - zone.x, y - zone.y) <= zone.radius:
                    return False
        return True

    def snap(self, xy: Point2, search_radius_cells: int = 10, blocked_zones: Optional[List[BlockedZone]] = None) -> Point2:
        base = self.xy_to_index(xy)
        best_idx: Optional[GridIndex] = None
        best_score = float('inf')
        for radius in range(search_radius_cells + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    idx = (base[0] + dx, base[1] + dy)
                    if not self._inside_index(idx):
                        continue
                    px, py = self.index_to_xy(idx)
                    if not self.is_traversable_xy(px, py, blocked_zones):
                        continue
                    score = math.hypot(px - xy[0], py - xy[1]) + 0.10 * self._boundary_penalty(px, py)
                    if score < best_score:
                        best_score = score
                        best_idx = idx
            if best_idx is not None and radius >= 1:
                break
        if best_idx is None:
            raise RuntimeError(f'No traversable snap cell near {xy}')
        return self.index_to_xy(best_idx)

    def _edge_cost(self, current: GridIndex, nxt: GridIndex, blocked_zones: Optional[List[BlockedZone]] = None) -> Optional[float]:
        if not self._inside_index(nxt):
            return None
        cx, cy = self.index_to_xy(current)
        nx, ny = self.index_to_xy(nxt)
        if not self.is_traversable_xy(nx, ny, blocked_zones):
            return None
        step = math.hypot(nx - cx, ny - cy)
        if step <= 1e-9:
            return None
        ch = self.grid_heights[current[1], current[0]]
        nh = self.grid_heights[nxt[1], nxt[0]]
        height_delta = abs(nh - ch)
        grade = height_delta / step
        if self.max_edge_height_delta > 0.0 and height_delta > self.max_edge_height_delta:
            return None
        if self.strict_grade_limit and grade > self.max_edge_grade:
            return None
        uphill = max(0.0, nh - ch) / step
        downhill = max(0.0, ch - nh) / step
        slope_penalty = self.slope_cost_weight * (grade + 0.75 * uphill + 0.35 * downhill)
        clearance_penalty = self.boundary_cost_weight * self._boundary_penalty(nx, ny)
        return step * (1.0 + slope_penalty + clearance_penalty)

    def astar(self, start_xy: Point2, goal_xy: Point2, blocked_zones: Optional[List[BlockedZone]] = None) -> List[Point2]:
        start = self.xy_to_index(self.snap(start_xy, blocked_zones=blocked_zones))
        goal = self.xy_to_index(self.snap(goal_xy, blocked_zones=blocked_zones))
        if start == goal:
            return [self.index_to_xy(start)]
        open_heap: List[Tuple[float, GridIndex]] = [(0.0, start)]
        came_from: Dict[GridIndex, Optional[GridIndex]] = {start: None}
        g_score: Dict[GridIndex, float] = {start: 0.0}
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                break
            for ox, oy in offsets:
                nxt = (current[0] + ox, current[1] + oy)
                # Avoid squeezing diagonally through two blocked/high-step side cells.
                if ox != 0 and oy != 0:
                    side_a = (current[0] + ox, current[1])
                    side_b = (current[0], current[1] + oy)
                    if self._edge_cost(current, side_a, blocked_zones) is None:
                        continue
                    if self._edge_cost(current, side_b, blocked_zones) is None:
                        continue
                edge_cost = self._edge_cost(current, nxt, blocked_zones)
                if edge_cost is None:
                    continue
                tentative = g_score[current] + edge_cost
                if tentative < g_score.get(nxt, float('inf')):
                    came_from[nxt] = current
                    g_score[nxt] = tentative
                    nx, ny = self.index_to_xy(nxt)
                    gx, gy = self.index_to_xy(goal)
                    heapq.heappush(open_heap, (tentative + math.hypot(gx - nx, gy - ny), nxt))
        if goal not in came_from:
            raise RuntimeError(
                f'No traversable route from {start_xy} to {goal_xy} using {self.height_source_label}; '
                f'max_edge_grade={self.max_edge_grade:.3f}, max_edge_height_delta={self.max_edge_height_delta:.3f}'
            )
        indices: List[GridIndex] = []
        cur: Optional[GridIndex] = goal
        while cur is not None:
            indices.append(cur)
            cur = came_from[cur]
        return [self.index_to_xy(idx) for idx in reversed(indices)]

    @staticmethod
    def _downsample(route: List[Point2]) -> List[Point2]:
        if len(route) <= 2:
            return route
        # Remove tiny A-B-A loops caused by connecting separately optimized A* segments.
        pruned: List[Point2] = []
        for point in route:
            if len(pruned) >= 2 and math.hypot(point[0] - pruned[-2][0], point[1] - pruned[-2][1]) < 1e-6:
                pruned.pop()
                continue
            if pruned and math.hypot(point[0] - pruned[-1][0], point[1] - pruned[-1][1]) < 1e-6:
                continue
            pruned.append(point)
        out = [pruned[0]]
        for point in pruned[1:]:
            if math.hypot(point[0] - out[-1][0], point[1] - out[-1][1]) >= 0.50:
                out.append(point)
        if out[-1] != pruned[-1]:
            out.append(pruned[-1])
        return out

    def plan_route(self, start_xy: Point2, scan_targets: List[Point2], blocked_zones: Optional[List[BlockedZone]] = None) -> List[Point2]:
        route: List[Point2] = []
        current = self.snap(start_xy, blocked_zones=blocked_zones)
        route.append(current)
        for target in scan_targets:
            snapped = self.snap(target, blocked_zones=blocked_zones)
            segment = self.astar(current, snapped, blocked_zones=blocked_zones)
            if len(segment) > 1:
                route.extend(segment[1:])
            current = snapped
        return self._downsample(route)


class MeshScanNode(Node):
    def __init__(self):
        super().__init__('mesh_scan_node')
        self.odom_topic = self.declare_parameter('odom_topic', '/odom_truth').value
        self.cmd_vel_topic = self.declare_parameter('cmd_vel_topic', '/cmd_vel').value
        self.terrain_model_name = self.declare_parameter('terrain_model_name', 'terrain_alpha2_h15').value
        self.grid_resolution = float(self.declare_parameter('grid_resolution', 0.50).value)
        self.drive_x_min = float(self.declare_parameter('drive_x_min', -7.2).value)
        self.drive_x_max = float(self.declare_parameter('drive_x_max', 7.2).value)
        self.drive_y_min = float(self.declare_parameter('drive_y_min', -7.2).value)
        self.drive_y_max = float(self.declare_parameter('drive_y_max', 7.2).value)
        self.hard_x_min = float(self.declare_parameter('hard_x_min', -8.4).value)
        self.hard_x_max = float(self.declare_parameter('hard_x_max', 8.4).value)
        self.hard_y_min = float(self.declare_parameter('hard_y_min', -8.4).value)
        self.hard_y_max = float(self.declare_parameter('hard_y_max', 8.4).value)
        self.start_x = float(self.declare_parameter('start_x', -5.8).value)
        self.start_y = float(self.declare_parameter('start_y', -5.8).value)
        self.scan_extent = float(self.declare_parameter('scan_extent', 5.8).value)
        self.scan_inner = float(self.declare_parameter('scan_inner', 2.0).value)
        self.cruise_speed = float(self.declare_parameter('cruise_speed', 0.26).value)
        self.rejoin_speed = float(self.declare_parameter('rejoin_speed', 0.14).value)
        self.hard_return_speed = float(self.declare_parameter('hard_return_speed', 0.08).value)
        self.max_angular_speed = float(self.declare_parameter('max_angular_speed', 0.55).value)
        self.rotate_only_error = float(self.declare_parameter('rotate_only_error', 0.82).value)
        self.slowdown_error = float(self.declare_parameter('slowdown_error', 0.45).value)
        self.angular_kp = float(self.declare_parameter('angular_kp', 1.25).value)
        self.lookahead = float(self.declare_parameter('lookahead', 0.95).value)
        self.rejoin_lookahead = float(self.declare_parameter('rejoin_lookahead', 0.65).value)
        self.corridor_radius = float(self.declare_parameter('corridor_radius', 0.90).value)
        self.rejoin_radius = float(self.declare_parameter('rejoin_radius', 1.25).value)
        self.target_radius = float(self.declare_parameter('target_radius', 0.55).value)
        self.scan_catchup_window = int(self.declare_parameter('scan_catchup_window', 4).value)
        self.dwell_time = float(self.declare_parameter('dwell_time', 0.8).value)
        self.publish_rate = float(self.declare_parameter('publish_rate', 20.0).value)
        self.route_csv_path = self.declare_parameter('route_csv_path', '/tmp/mesh_scan_route.csv').value
        self.auto_optimize = bool(self.declare_parameter('auto_optimize', True).value)
        self.auto_reconstruct = bool(self.declare_parameter('auto_reconstruct', True).value)
        self.boundary_cost_weight = float(self.declare_parameter('boundary_cost_weight', 4.0).value)
        self.slope_cost_weight = float(self.declare_parameter('slope_cost_weight', 9.0).value)
        self.max_edge_grade = float(self.declare_parameter('max_edge_grade', 0.18).value)
        self.max_edge_height_delta = float(self.declare_parameter('max_edge_height_delta', 0.085).value)
        self.terrain_source = self.declare_parameter('terrain_source', 'collision').value
        self.strict_grade_limit = bool(self.declare_parameter('strict_grade_limit', True).value)
        self.footprint_length = float(self.declare_parameter('footprint_length', 0.70).value)
        self.footprint_width = float(self.declare_parameter('footprint_width', 0.57).value)
        self.footprint_margin = float(self.declare_parameter('footprint_margin', 0.12).value)
        self.max_footprint_height_delta = float(self.declare_parameter('max_footprint_height_delta', 0.22).value)
        self.max_footprint_roughness = float(self.declare_parameter('max_footprint_roughness', 0.065).value)
        self.progress_watchdog_time = float(self.declare_parameter('progress_watchdog_time', 9.0).value)
        self.progress_watchdog_min_delta = float(self.declare_parameter('progress_watchdog_min_delta', 0.30).value)
        self.blocked_zone_radius = float(self.declare_parameter('blocked_zone_radius', 1.0).value)
        self.blocked_zone_lifetime = float(self.declare_parameter('blocked_zone_lifetime', 90.0).value)
        self.max_replans = int(self.declare_parameter('max_replans', 5).value)
        self.marker_z_offset = float(self.declare_parameter('marker_z_offset', 0.20).value)

        orchard_share = get_package_share_directory('orchard_sim')
        model_dir = os.path.join(orchard_share, 'models', self.terrain_model_name)
        collision_obj_path = os.path.join(model_dir, 'collision_mesh.obj')
        model_sdf_path = os.path.join(model_dir, 'model.sdf')
        visual_obj_path = os.path.join(model_dir, 'mesh.obj')
        package_obj_path = os.path.join(orchard_share, 'orchard_sim', f'{self.terrain_model_name}.obj')
        if self.terrain_source == 'collision' and os.path.exists(collision_obj_path):
            height_source_path = collision_obj_path
            height_source_label = 'smooth_collision_mesh'
        elif self.terrain_source == 'collision' and os.path.exists(model_sdf_path):
            height_source_path = model_sdf_path
            height_source_label = 'sdf_tile_collision'
        elif os.path.exists(visual_obj_path):
            height_source_path = visual_obj_path
            height_source_label = 'visual_mesh_fallback'
        else:
            height_source_path = package_obj_path
            height_source_label = 'package_visual_mesh_fallback'
        self.planner = TerrainRoutePlanner(
            height_source_path, height_source_label, self.grid_resolution,
            self.drive_x_min, self.drive_x_max, self.drive_y_min, self.drive_y_max,
            self.boundary_cost_weight, self.slope_cost_weight, self.max_edge_grade,
            self.max_edge_height_delta, self.strict_grade_limit, self.footprint_length, self.footprint_width,
            self.footprint_margin, self.max_footprint_height_delta, self.max_footprint_roughness,
        )
        self.height_source_path = height_source_path
        self.height_source_label = height_source_label
        self.scan_targets = self._make_scan_targets()
        self.blocked_zones: List[BlockedZone] = []
        self.replan_count = 0
        self.route = self.planner.plan_route((self.start_x, self.start_y), self.scan_targets, self.blocked_zones)
        self.route_s = self._cumulative_lengths(self.route)
        self.total_length = self.route_s[-1] if self.route_s else 0.0
        self._write_route_csv()

        self.pose: Optional[Tuple[float, float, float]] = None
        self.last_progress = 0.0
        self.watchdog_progress = 0.0
        self.watchdog_time = self.now_seconds()
        self.current_scan_index = 0
        self.dwell_until: Optional[float] = None
        self.dwell_scan_index: Optional[int] = None
        self.optimize_called = False
        self.reconstruct_called = False
        self.last_mode_log = ''

        transient_qos = QoSProfile(depth=1)
        transient_qos.reliability = ReliabilityPolicy.RELIABLE
        transient_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.path_pub = self.create_publisher(Path, '/mesh_scan/path', transient_qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/mesh_scan/markers', transient_qos)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 20)
        self.optimize_client = self.create_client(Trigger, '/pointcloud_optimizer/optimize')
        self.reconstruct_client = self.create_client(Trigger, '/pointcloud_to_mesh/reconstruct')
        self.timer = self.create_timer(1.0 / max(2.0, self.publish_rate), self.timer_callback)
        self.marker_timer = self.create_timer(1.0, self.publish_visualization)
        self._publish_path()
        self.publish_visualization()
        self._log_route_audit()

    def _make_scan_targets(self) -> List[Point2]:
        e, m = self.scan_extent, self.scan_inner
        rows = [-e, -m, m, e]
        cols = [-e, -m, m, e]
        targets: List[Point2] = []
        for row_i, y in enumerate(rows):
            ordered_cols = cols if row_i % 2 == 0 else list(reversed(cols))
            targets.extend((x, y) for x in ordered_cols)
        return targets

    @staticmethod
    def _cumulative_lengths(points: List[Point2]) -> List[float]:
        lengths = [0.0]
        total = 0.0
        for a, b in zip(points[:-1], points[1:]):
            total += math.hypot(b[0] - a[0], b[1] - a[1])
            lengths.append(total)
        return lengths

    def _write_route_csv(self):
        os.makedirs(os.path.dirname(self.route_csv_path) or '.', exist_ok=True)
        lengths = self._cumulative_lengths(self.route)
        with open(self.route_csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['index', 'x', 'y', 'z', 's'])
            for i, (x, y) in enumerate(self.route):
                writer.writerow([i, f'{x:.3f}', f'{y:.3f}', f'{self.planner.height_at(x, y):.3f}', f'{lengths[i]:.3f}'])

    def _log_route_audit(self):
        if len(self.route) >= 2:
            first_heading = math.atan2(self.route[1][1] - self.route[0][1], self.route[1][0] - self.route[0][0])
        else:
            first_heading = 0.0
        min_clearance = min(min(x - self.drive_x_min, self.drive_x_max - x, y - self.drive_y_min, self.drive_y_max - y) for x, y in self.route)
        max_grade = 0.0
        max_height_delta = 0.0
        for a, b in zip(self.route[:-1], self.route[1:]):
            step = math.hypot(b[0] - a[0], b[1] - a[1])
            if step > 1e-9:
                delta = abs(self.planner.height_at(*b) - self.planner.height_at(*a))
                max_height_delta = max(max_height_delta, delta)
                max_grade = max(max_grade, delta / step)
        self.get_logger().info(
            'Mesh scan route ready: %d scan targets, %d route waypoints, length %.1f m, source=%s, first=(%.2f, %.2f), second=(%.2f, %.2f), first_heading=%.2f rad, min_drive_clearance=%.2f m, max_route_grade=%.3f, max_edge_dz=%.3f m, max_allowed_dz=%.3f m, csv=%s.'
            % (
                len(self.scan_targets), len(self.route), self.total_length, self.height_source_label,
                self.route[0][0], self.route[0][1],
                self.route[min(1, len(self.route)-1)][0], self.route[min(1, len(self.route)-1)][1],
                first_heading, min_clearance, max_grade, max_height_delta, self.max_edge_height_delta, self.route_csv_path,
            )
        )
        self.get_logger().warn('Route visualization is in RViz on /mesh_scan/path and /mesh_scan/markers. Do not add physical Gazebo route geometry during point-cloud collection because GPU lidar may scan it.')

    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def project_to_route(self, xy: Point2, min_progress: float = 0.0) -> RouteProjection:
        if len(self.route) == 1:
            return RouteProjection(0, math.hypot(xy[0] - self.route[0][0], xy[1] - self.route[0][1]), 0.0, self.route[0])
        best = RouteProjection(0, float('inf'), 0.0, self.route[0])
        min_progress = max(0.0, min_progress)
        for i, (a, b) in enumerate(zip(self.route[:-1], self.route[1:])):
            if self.route_s[i + 1] < min_progress:
                continue
            ax, ay = a
            bx, by = b
            vx, vy = bx - ax, by - ay
            seg_len2 = vx * vx + vy * vy
            if seg_len2 <= 1e-9:
                continue
            t = clamp(((xy[0] - ax) * vx + (xy[1] - ay) * vy) / seg_len2, 0.0, 1.0)
            px, py = ax + t * vx, ay + t * vy
            dist = math.hypot(xy[0] - px, xy[1] - py)
            progress = self.route_s[i] + t * math.sqrt(seg_len2)
            if progress + 1e-6 < min_progress:
                continue
            if dist < best.distance:
                best = RouteProjection(i, dist, progress, (px, py))
        return best

    def point_at_progress(self, progress: float) -> Point2:
        progress = clamp(progress, 0.0, self.total_length)
        for i in range(len(self.route_s) - 1):
            if self.route_s[i] <= progress <= self.route_s[i + 1]:
                seg_len = self.route_s[i + 1] - self.route_s[i]
                if seg_len <= 1e-9:
                    return self.route[i]
                t = (progress - self.route_s[i]) / seg_len
                ax, ay = self.route[i]
                bx, by = self.route[i + 1]
                return ax + t * (bx - ax), ay + t * (by - ay)
        return self.route[-1]

    def nearest_target_reached(self, xy: Point2) -> bool:
        if self.current_scan_index >= len(self.scan_targets):
            return False
        tx, ty = self.planner.snap(self.scan_targets[self.current_scan_index])
        return math.hypot(xy[0] - tx, xy[1] - ty) <= self.target_radius

    def reached_scan_index(self, xy: Point2) -> Optional[int]:
        if self.current_scan_index >= len(self.scan_targets):
            return None
        end = min(len(self.scan_targets), self.current_scan_index + max(1, self.scan_catchup_window))
        best_index: Optional[int] = None
        best_distance = float('inf')
        for i in range(self.current_scan_index, end):
            tx, ty = self.planner.snap(self.scan_targets[i])
            distance = math.hypot(xy[0] - tx, xy[1] - ty)
            if distance <= self.target_radius and distance < best_distance:
                best_index = i
                best_distance = distance
        return best_index

    def hard_boundary_exceeded(self, x: float, y: float) -> bool:
        return x < self.hard_x_min or x > self.hard_x_max or y < self.hard_y_min or y > self.hard_y_max

    def near_hard_boundary(self, x: float, y: float) -> bool:
        return min(x - self.hard_x_min, self.hard_x_max - x, y - self.hard_y_min, self.hard_y_max - y) < 0.75

    def _make_command(self, target: Point2, speed: float, mode: str) -> Twist:
        cmd = Twist()
        if self.pose is None:
            return cmd
        x, y, yaw = self.pose
        desired_heading = math.atan2(target[1] - y, target[0] - x)
        heading_error = normalize_angle(desired_heading - yaw)
        cmd.angular.z = clamp(self.angular_kp * heading_error, -self.max_angular_speed, self.max_angular_speed)
        if abs(heading_error) > self.rotate_only_error:
            cmd.linear.x = 0.0
        else:
            heading_factor = max(0.0, math.cos(heading_error))
            if abs(heading_error) > self.slowdown_error:
                heading_factor *= 0.45
            cmd.linear.x = speed * heading_factor
        if mode != self.last_mode_log:
            self.get_logger().info('Mesh scan control mode=%s, target=(%.2f, %.2f), heading_error=%.2f, cmd=(%.2f, %.2f).' % (mode, target[0], target[1], heading_error, cmd.linear.x, cmd.angular.z))
            self.last_mode_log = mode
        return cmd


    def _remaining_targets(self) -> List[Point2]:
        return self.scan_targets[self.current_scan_index:]

    def _prune_blocked_zones(self, now: float):
        self.blocked_zones = [zone for zone in self.blocked_zones if zone.until > now]

    def replan_from_current_pose(self, reason: str) -> bool:
        if self.pose is None or self.current_scan_index >= len(self.scan_targets):
            return False
        now = self.now_seconds()
        self._prune_blocked_zones(now)
        x, y, _ = self.pose
        self.blocked_zones.append(BlockedZone(x, y, self.blocked_zone_radius, now + self.blocked_zone_lifetime, reason))
        self.replan_count += 1
        if self.replan_count > self.max_replans:
            self.get_logger().error('Mesh scan watchdog exceeded max_replans=%d; stopping to avoid unsafe repeated attempts.' % self.max_replans)
            return False
        try:
            remaining = self._remaining_targets()
            if not remaining:
                return False
            self.route = self.planner.plan_route((x, y), remaining, self.blocked_zones)
            self.route_s = self._cumulative_lengths(self.route)
            self.total_length = self.route_s[-1] if self.route_s else 0.0
            self.last_progress = 0.0
            self.watchdog_progress = 0.0
            self.watchdog_time = now
            self._write_route_csv()
            self._publish_path()
            self.publish_visualization()
            self.get_logger().warn('Progress watchdog replanned from current pose after %s; blocked_zones=%d, remaining_targets=%d.' % (reason, len(self.blocked_zones), len(remaining)))
            self._log_route_audit()
            return True
        except Exception as exc:
            self.get_logger().error('Replan failed after %s: %s. Skipping current scan target and stopping briefly.' % (reason, exc))
            self.current_scan_index = min(self.current_scan_index + 1, len(self.scan_targets))
            return False

    def timer_callback(self):
        if self.pose is None or not self.route:
            self.cmd_pub.publish(Twist())
            return
        x, y, _ = self.pose
        xy = (x, y)
        now = self.now_seconds()

        reached_index = self.reached_scan_index(xy)
        if reached_index is not None:
            if self.dwell_until is None:
                self.dwell_until = now + self.dwell_time
                self.dwell_scan_index = reached_index
                if reached_index > self.current_scan_index:
                    self.get_logger().warn('Scan catch-up: reached target %d/%d while waiting for %d/%d; marking intervening targets complete.' % (reached_index + 1, len(self.scan_targets), self.current_scan_index + 1, len(self.scan_targets)))
                self.get_logger().info('Scan dwell %d/%d at (%.2f, %.2f).' % (reached_index + 1, len(self.scan_targets), x, y))
            if now < self.dwell_until:
                self.cmd_pub.publish(Twist())
                return
            completed_index = self.dwell_scan_index if self.dwell_scan_index is not None else reached_index
            self.current_scan_index = max(self.current_scan_index, completed_index + 1)
            self.dwell_until = None
            self.dwell_scan_index = None

        if self.current_scan_index >= len(self.scan_targets):
            self.cmd_pub.publish(Twist())
            if not self.optimize_called and self.auto_optimize:
                self.optimize_called = True
                if self.optimize_client.service_is_ready():
                    self.optimize_client.call_async(Trigger.Request())
                    self.get_logger().info('Mesh scan complete: triggered pointcloud optimizer.')
            if not self.reconstruct_called and self.auto_reconstruct:
                self.reconstruct_called = True
                if self.reconstruct_client.service_is_ready():
                    self.reconstruct_client.call_async(Trigger.Request())
                    self.get_logger().info('Mesh scan complete: triggered mesh reconstruction.')
            return

        projection = self.project_to_route(xy, min_progress=max(0.0, self.last_progress - 1.0))
        self.last_progress = max(self.last_progress, projection.progress)
        if self.last_progress - self.watchdog_progress >= self.progress_watchdog_min_delta:
            self.watchdog_progress = self.last_progress
            self.watchdog_time = now
        elif now - self.watchdog_time > self.progress_watchdog_time:
            self.get_logger().warn('Progress watchdog: route progress advanced only %.2f m in %.1f s at cross-track %.2f m; replanning around this patch instead of forcing the slope.' % (self.last_progress - self.watchdog_progress, now - self.watchdog_time, projection.distance))
            if self.replan_from_current_pose('no_route_progress'):
                self.cmd_pub.publish(Twist())
                return
            self.watchdog_time = now
            self.cmd_pub.publish(Twist())
            return

        if self.hard_boundary_exceeded(x, y) or self.near_hard_boundary(x, y):
            # Hard safety is intentionally simple: aim toward the safe interior,
            # not toward a route point that may be close to the same boundary.
            safe_x = clamp(x, self.drive_x_min + 1.0, self.drive_x_max - 1.0)
            safe_y = clamp(y, self.drive_y_min + 1.0, self.drive_y_max - 1.0)
            target = (0.65 * safe_x, 0.65 * safe_y)
            self.cmd_pub.publish(self._make_command(target, self.hard_return_speed, 'hard_boundary_return'))
            return
        if projection.distance > self.rejoin_radius:
            target = self.point_at_progress(projection.progress + self.rejoin_lookahead)
            self.cmd_pub.publish(self._make_command(target, self.rejoin_speed, 'route_rejoin'))
            return
        if projection.distance > self.corridor_radius:
            target = self.point_at_progress(projection.progress + self.rejoin_lookahead)
            self.cmd_pub.publish(self._make_command(target, min(self.rejoin_speed, self.cruise_speed), 'corridor_correction'))
            return
        target = self.point_at_progress(max(self.last_progress, projection.progress) + self.lookahead)
        self.cmd_pub.publish(self._make_command(target, self.cruise_speed, 'route_follow'))

    def _publish_path(self):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.route:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = self.planner.height_at(x, y) + self.marker_z_offset
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)

    def _line_marker(self, marker_id: int, ns: str, points: Iterable[Point2], r: float, g: float, b: float, width: float) -> Marker:
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = width
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        marker.lifetime = Duration(seconds=0).to_msg()
        for x, y in points:
            p = Point(x=x, y=y, z=self.planner.height_at(x, y) + self.marker_z_offset)
            marker.points.append(p)
        return marker

    def publish_visualization(self):
        markers = MarkerArray()
        markers.markers.append(self._line_marker(1, 'route', self.route, 0.1, 0.8, 1.0, 0.08))
        drive_box = [(self.drive_x_min, self.drive_y_min), (self.drive_x_max, self.drive_y_min), (self.drive_x_max, self.drive_y_max), (self.drive_x_min, self.drive_y_max), (self.drive_x_min, self.drive_y_min)]
        hard_box = [(self.hard_x_min, self.hard_y_min), (self.hard_x_max, self.hard_y_min), (self.hard_x_max, self.hard_y_max), (self.hard_x_min, self.hard_y_max), (self.hard_x_min, self.hard_y_min)]
        markers.markers.append(self._line_marker(2, 'drive_box', drive_box, 0.2, 1.0, 0.2, 0.05))
        markers.markers.append(self._line_marker(3, 'hard_box', hard_box, 1.0, 0.2, 0.2, 0.05))
        for i, (x, y) in enumerate([self.planner.snap(t) for t in self.scan_targets]):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'scan_targets'
            marker.id = 100 + i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = self.planner.height_at(x, y) + self.marker_z_offset + 0.10
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.28
            marker.color.r = 1.0
            marker.color.g = 0.8 if i >= self.current_scan_index else 0.2
            marker.color.b = 0.1
            marker.color.a = 1.0 if i >= self.current_scan_index else 0.35
            markers.markers.append(marker)
        if self.pose is not None:
            proj = self.project_to_route((self.pose[0], self.pose[1]))
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'route_projection'
            marker.id = 300
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = proj.point[0]
            marker.pose.position.y = proj.point[1]
            marker.pose.position.z = self.planner.height_at(*proj.point) + self.marker_z_offset + 0.18
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.20
            marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.0
            markers.markers.append(marker)
        self.marker_pub.publish(markers)
        self._publish_path()


def main(args=None):
    rclpy.init(args=args)
    node = MeshScanNode()
    try:
        rclpy.spin(node)
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
