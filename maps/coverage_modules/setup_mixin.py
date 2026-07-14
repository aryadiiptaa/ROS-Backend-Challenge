from __future__ import annotations
import math
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose, Spin
from nav2_msgs.msg import CostmapFilterInfo
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path as NavPathMessage
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray
from .types import PixelPoint, Polygon, WorldPoint


class CoverageSetupMixin:

    def __init__(self) -> None:
        super().__init__("robot_ros_coverage_navigator")
        self.package_share = Path(get_package_share_directory("robot_ros_backend"))
        self.coverage_bt_path = str(
            self.package_share / "behavior_trees" / "coverage_through_poses.xml"
        )
        if not Path(self.coverage_bt_path).is_file():
            raise FileNotFoundError(f"behavior tree missing: {self.coverage_bt_path}")
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("line_spacing_m", 0.4)
        self.declare_parameter("safety_margin_m", 0.35)
        self.declare_parameter("minimum_component_pixels", 20)
        self.declare_parameter("turn_radius_m", 0.4)
        self.declare_parameter("turn_steps", 12)
        self.declare_parameter("minimum_segment_length_m", 0.12)
        self.declare_parameter("connector_sample_spacing_m", 0.35)
        self.declare_parameter("scanline_edge_offset_m", 0.2)
        self.declare_parameter("exclusion_margin_m", 0.35)
        self.declare_parameter("nav_simplify_tolerance_m", 0.1)
        self.declare_parameter("max_nav_waypoints", 140)
        self.declare_parameter("start_heading_weight_m", 2.0)
        self.declare_parameter("start_entry_distance_m", 0.55)
        self.declare_parameter("start_entry_min_distance_m", 0.2)
        self.declare_parameter("start_entry_clearance_m", 0.45)
        self.declare_parameter("pre_spin_enabled", True)
        self.declare_parameter("pre_spin_threshold_deg", 35.0)
        self.declare_parameter("transition_pre_spin_threshold_deg", 20.0)
        self.declare_parameter("sharp_turn_split_deg", 35.0)
        self.declare_parameter("sweep_straightness_deg", 8.0)
        self.declare_parameter("astar_clearance_radius_m", 0.3)
        self.declare_parameter("astar_clearance_weight", 4.0)
        self.declare_parameter("pre_spin_time_allowance_s", 20.0)
        self.declare_parameter("chunk_start_tolerance_m", 0.22)
        self.declare_parameter("waypoint_pass_radius_m", 0.18)
        self.declare_parameter("chunk_projection_tolerance_m", 0.18)
        self.declare_parameter("chunk_retry_limit", 1)
        self.declare_parameter("minimum_motion_chunk_length_m", 0.45)
        self.declare_parameter("spin_lookahead_distance_m", 0.65)
        self.declare_parameter("minimum_spin_path_length_m", 0.2)
        self.declare_parameter("mandatory_sweep_min_length_m", 0.25)
        self.declare_parameter("mandatory_corner_angle_deg", 30.0)
        self.declare_parameter("rviz_visualization_enabled", True)
        self.declare_parameter("rviz_area_sample_step_px", 4)
        self.declare_parameter("keepout_filter_enabled", True)
        self.declare_parameter(
            "keepout_filter_info_topic", "/keepout_costmap_filter_info"
        )
        self.declare_parameter("keepout_filter_mask_topic", "/keepout_filter_mask")
        self.declare_parameter("no_progress_watchdog_enabled", False)
        self.declare_parameter("no_progress_timeout_s", 20.0)
        self.declare_parameter("no_progress_min_delta_m", 0.05)
        self.declare_parameter("no_progress_check_period_s", 1.0)
        self.declare_parameter("actual_coverage_tracking_enabled", True)
        self.declare_parameter("coverage_tool_width_m", 0.4)
        self.declare_parameter("coverage_tracking_period_s", 0.2)
        self.declare_parameter("actual_coverage_publish_period_s", 2.0)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.line_spacing_m = float(self.get_parameter("line_spacing_m").value)
        self.safety_margin_m = float(self.get_parameter("safety_margin_m").value)
        self.minimum_component_pixels = int(
            self.get_parameter("minimum_component_pixels").value
        )
        self.turn_radius_m = float(self.get_parameter("turn_radius_m").value)
        self.turn_steps = max(6, int(self.get_parameter("turn_steps").value))
        self.minimum_segment_length_m = float(
            self.get_parameter("minimum_segment_length_m").value
        )
        self.connector_sample_spacing_m = float(
            self.get_parameter("connector_sample_spacing_m").value
        )
        self.scanline_edge_offset_m = float(
            self.get_parameter("scanline_edge_offset_m").value
        )
        self.exclusion_margin_m = float(self.get_parameter("exclusion_margin_m").value)
        self.nav_simplify_tolerance_m = float(
            self.get_parameter("nav_simplify_tolerance_m").value
        )
        self.max_nav_waypoints = max(
            20, int(self.get_parameter("max_nav_waypoints").value)
        )
        self.start_heading_weight_m = max(
            0.0, float(self.get_parameter("start_heading_weight_m").value)
        )
        self.start_entry_distance_m = max(
            0.0, float(self.get_parameter("start_entry_distance_m").value)
        )
        self.start_entry_min_distance_m = max(
            0.0, float(self.get_parameter("start_entry_min_distance_m").value)
        )
        self.start_entry_clearance_m = max(
            self.safety_margin_m,
            float(self.get_parameter("start_entry_clearance_m").value),
        )
        self.pre_spin_enabled = bool(self.get_parameter("pre_spin_enabled").value)
        self.pre_spin_threshold_rad = math.radians(
            max(0.0, float(self.get_parameter("pre_spin_threshold_deg").value))
        )
        self.transition_pre_spin_threshold_rad = math.radians(
            max(
                0.0,
                float(self.get_parameter("transition_pre_spin_threshold_deg").value),
            )
        )
        self.sharp_turn_split_rad = math.radians(
            max(5.0, float(self.get_parameter("sharp_turn_split_deg").value))
        )
        self.sweep_straightness_rad = math.radians(
            max(1.0, float(self.get_parameter("sweep_straightness_deg").value))
        )
        self.astar_clearance_radius_m = max(
            0.0, float(self.get_parameter("astar_clearance_radius_m").value)
        )
        self.astar_clearance_weight = max(
            0.0, float(self.get_parameter("astar_clearance_weight").value)
        )
        self.pre_spin_time_allowance_s = max(
            1.0, float(self.get_parameter("pre_spin_time_allowance_s").value)
        )
        self.chunk_start_tolerance_m = max(
            0.05, float(self.get_parameter("chunk_start_tolerance_m").value)
        )
        self.waypoint_pass_radius_m = max(
            0.05, float(self.get_parameter("waypoint_pass_radius_m").value)
        )
        requested_projection_tolerance = max(
            0.05, float(self.get_parameter("chunk_projection_tolerance_m").value)
        )
        maximum_projection_tolerance = max(0.05, 0.45 * self.line_spacing_m)
        self.chunk_projection_tolerance_m = min(
            requested_projection_tolerance, maximum_projection_tolerance
        )
        self.chunk_retry_limit = max(
            0, int(self.get_parameter("chunk_retry_limit").value)
        )
        self.minimum_motion_chunk_length_m = max(
            0.0, float(self.get_parameter("minimum_motion_chunk_length_m").value)
        )
        maximum_safe_pass_radius = max(0.05, 0.45 * self.line_spacing_m)
        if self.waypoint_pass_radius_m > maximum_safe_pass_radius:
            self.get_logger().warning("waypoint radius clamped")
            self.waypoint_pass_radius_m = maximum_safe_pass_radius
        self.spin_lookahead_distance_m = max(
            0.05, float(self.get_parameter("spin_lookahead_distance_m").value)
        )
        self.minimum_spin_path_length_m = max(
            0.0, float(self.get_parameter("minimum_spin_path_length_m").value)
        )
        self.mandatory_sweep_min_length_m = max(
            0.05, float(self.get_parameter("mandatory_sweep_min_length_m").value)
        )
        self.mandatory_corner_angle_rad = math.radians(
            max(5.0, float(self.get_parameter("mandatory_corner_angle_deg").value))
        )
        self.rviz_visualization_enabled = bool(
            self.get_parameter("rviz_visualization_enabled").value
        )
        self.rviz_area_sample_step_px = max(
            1, int(self.get_parameter("rviz_area_sample_step_px").value)
        )
        self.keepout_filter_enabled = bool(
            self.get_parameter("keepout_filter_enabled").value
        )
        self.keepout_filter_info_topic = str(
            self.get_parameter("keepout_filter_info_topic").value
        ).strip()
        self.keepout_filter_mask_topic = str(
            self.get_parameter("keepout_filter_mask_topic").value
        ).strip()
        self.no_progress_watchdog_enabled = bool(
            self.get_parameter("no_progress_watchdog_enabled").value
        )
        self.no_progress_timeout_s = max(
            5.0, float(self.get_parameter("no_progress_timeout_s").value)
        )
        self.no_progress_min_delta_m = max(
            0.01, float(self.get_parameter("no_progress_min_delta_m").value)
        )
        self.no_progress_check_period_s = max(
            0.2, float(self.get_parameter("no_progress_check_period_s").value)
        )
        self.actual_coverage_tracking_enabled = bool(
            self.get_parameter("actual_coverage_tracking_enabled").value
        )
        self.coverage_tool_width_m = max(
            0.05, float(self.get_parameter("coverage_tool_width_m").value)
        )
        self.coverage_tracking_period_s = max(
            0.05, float(self.get_parameter("coverage_tracking_period_s").value)
        )
        self.actual_coverage_publish_period_s = max(
            0.5, float(self.get_parameter("actual_coverage_publish_period_s").value)
        )
        if self.keepout_filter_enabled:
            if not self.keepout_filter_info_topic:
                raise ValueError("keepout_filter_info_topic empty")
            if not self.keepout_filter_mask_topic:
                raise ValueError("keepout_filter_mask_topic empty")
        if self.line_spacing_m <= 0.0:
            raise ValueError("line_spacing_m <= 0")
        if self.safety_margin_m < 0.0:
            raise ValueError("safety_margin_m < 0")
        if self.turn_radius_m < 0.0:
            raise ValueError("turn_radius_m < 0")
        if self.minimum_segment_length_m <= 0.0:
            raise ValueError("minimum_segment_length_m <= 0")
        if self.connector_sample_spacing_m <= 0.0:
            raise ValueError("connector_sample_spacing_m <= 0")
        if self.scanline_edge_offset_m < 0.0:
            raise ValueError("scanline_edge_offset_m < 0")
        if self.exclusion_margin_m < 0.0:
            raise ValueError("exclusion_margin_m < 0")
        if self.nav_simplify_tolerance_m <= 0.0:
            raise ValueError("nav_simplify_tolerance_m <= 0")
        if self.start_entry_min_distance_m > self.start_entry_distance_m:
            raise ValueError("start_entry_min_distance_m > start_entry_distance_m")
        self.map_image, self.map_path, self.resolution, self.origin = self._load_map()
        self.image_height, self.image_width = self.map_image.shape
        self.coverage_tool_width_m = max(self.resolution, self.coverage_tool_width_m)
        self.display_base = cv2.cvtColor(self.map_image, cv2.COLOR_GRAY2BGR)
        self.free_mask = np.where(self.map_image >= 250, 255, 0).astype(np.uint8)
        self.free_clearance_m = (
            cv2.distanceTransform(self.free_mask, cv2.DIST_L2, 5) * self.resolution
        )
        self.safe_free_mask = self._make_safe_free_mask()
        self.coverage_polygons: List[Polygon] = []
        self.exclusion_polygons: List[Polygon] = []
        self.current_polygon: Polygon = []
        self.draw_mode = "coverage"
        self.path_pixels: List[PixelPoint] = []
        self.path_world: List[WorldPoint] = []
        self.start_entry_pixel: Optional[PixelPoint] = None
        self.initial_approach_world: Optional[WorldPoint] = None
        self.initial_approach_yaw: Optional[float] = None
        self.initial_approach_done = False
        self.nav_path_pixels: List[PixelPoint] = []
        self.nav_path_world: List[WorldPoint] = []
        self.mandatory_path_pixels: List[PixelPoint] = []
        self.mandatory_path_world: List[WorldPoint] = []
        self.robot_yaw = 0.0
        self.cancel_pending = False
        self.actual_coverage_target_mask = np.zeros(
            self.map_image.shape, dtype=np.uint8
        )
        self.actual_coverage_mask = np.zeros(self.map_image.shape, dtype=np.uint8)
        self.actual_coverage_tracking_active = False
        self.actual_coverage_last_pixel: Optional[PixelPoint] = None
        self.actual_coverage_last_publish_ns = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_client = ActionClient(
            self, NavigateThroughPoses, "navigate_through_poses"
        )
        self.approach_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.spin_client = ActionClient(self, Spin, "spin")
        self.active_goal_handle = None
        self.send_goal_future = None
        self.get_result_future = None
        self.current_action_kind: Optional[str] = None
        self.route_chunks: List[List[WorldPoint]] = []
        self.current_chunk_index = 0
        self.current_execution_points: List[WorldPoint] = []
        self.current_chunk_retry_count = 0
        self.pending_spin_angle = 0.0
        self.watchdog_active = False
        self.watchdog_action_kind: Optional[str] = None
        self.watchdog_best_remaining_m = math.inf
        self.watchdog_last_progress_ns = 0
        self.watchdog_cancel_requested = False
        self.watchdog_triggered_kind: Optional[str] = None
        self.running = True
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.rviz_marker_pub = self.create_publisher(
            MarkerArray, "coverage/markers", marker_qos
        )
        self.visual_path_pub = self.create_publisher(
            NavPathMessage, "coverage/visual_path", marker_qos
        )
        self.nav_waypoints_pub = self.create_publisher(
            NavPathMessage, "coverage/nav_waypoints", marker_qos
        )
        self.active_chunk_pub = self.create_publisher(
            NavPathMessage, "coverage/active_chunk", marker_qos
        )
        self.keepout_info_pub = self.create_publisher(
            CostmapFilterInfo, self.keepout_filter_info_topic, marker_qos
        )
        self.keepout_mask_pub = self.create_publisher(
            OccupancyGrid, self.keepout_filter_mask_topic, marker_qos
        )
        self.actual_coverage_pub = self.create_publisher(
            OccupancyGrid, "coverage/actual_coverage", marker_qos
        )
        self.watchdog_timer = self.create_timer(
            self.no_progress_check_period_s, self._no_progress_watchdog_tick
        )
        self.actual_coverage_timer = self.create_timer(
            self.coverage_tracking_period_s, self._actual_coverage_tracking_tick
        )
        self._publish_keepout_filter()
        self._publish_actual_coverage_grid()
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
