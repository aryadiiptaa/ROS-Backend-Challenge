from __future__ import annotations
import math
from typing import List, Optional, Sequence, Tuple
import cv2
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose, Spin
from rclpy.duration import Duration
from .types import WorldPoint


class CoverageMissionMixin:

    def _start_no_progress_watchdog(self, action_kind: str) -> None:
        if not self.no_progress_watchdog_enabled:
            return
        self.watchdog_active = True
        self.watchdog_action_kind = action_kind
        self.watchdog_best_remaining_m = math.inf
        self.watchdog_last_progress_ns = self.get_clock().now().nanoseconds
        self.watchdog_cancel_requested = False
        self.watchdog_triggered_kind = None

    def _observe_no_progress_watchdog(self, distance_remaining_m: float) -> None:
        if not self.watchdog_active:
            return
        remaining = max(0.0, float(distance_remaining_m))
        if (
            not math.isfinite(self.watchdog_best_remaining_m)
            or remaining
            <= self.watchdog_best_remaining_m - self.no_progress_min_delta_m
        ):
            self.watchdog_best_remaining_m = remaining
            self.watchdog_last_progress_ns = self.get_clock().now().nanoseconds

    def _stop_no_progress_watchdog(self, clear_trigger: bool = True) -> None:
        self.watchdog_active = False
        self.watchdog_action_kind = None
        self.watchdog_best_remaining_m = math.inf
        self.watchdog_last_progress_ns = 0
        self.watchdog_cancel_requested = False
        if clear_trigger:
            self.watchdog_triggered_kind = None

    def _consume_watchdog_trigger(self, action_kind: str) -> bool:
        triggered = self.watchdog_triggered_kind == action_kind
        self._stop_no_progress_watchdog(clear_trigger=True)
        return triggered

    def _no_progress_watchdog_tick(self) -> None:
        if (
            not self.no_progress_watchdog_enabled
            or not self.watchdog_active
            or self.watchdog_cancel_requested
            or self.cancel_pending
            or (self.active_goal_handle is None)
        ):
            return
        now_ns = self.get_clock().now().nanoseconds
        elapsed_s = (now_ns - self.watchdog_last_progress_ns) / 1000000000.0
        if elapsed_s < self.no_progress_timeout_s:
            return
        self.watchdog_cancel_requested = True
        self.watchdog_triggered_kind = self.watchdog_action_kind
        self.get_logger().warning("navigation stalled")
        cancel_future = self.active_goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self._watchdog_cancel_response_callback)

    def _watchdog_cancel_response_callback(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"watchdog cancel: {exc}")
            self.watchdog_cancel_requested = False
            self.watchdog_last_progress_ns = self.get_clock().now().nanoseconds
            return
        if response.goals_canceling:
            self.get_logger().warning("stalled action canceled")
            return
        self.get_logger().warning("watchdog cancel rejected")
        self.watchdog_cancel_requested = False
        self.watchdog_triggered_kind = None
        self.watchdog_last_progress_ns = self.get_clock().now().nanoseconds

    @staticmethod
    def _yaw_quaternion(yaw: float) -> Tuple[float, float]:
        return (math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    def _build_pose_messages(
        self, points: Sequence[WorldPoint], single_pose_yaw: Optional[float] = None
    ) -> List[PoseStamped]:
        poses: List[PoseStamped] = []
        if not points:
            return poses
        stamp = self.get_clock().now().to_msg()
        last_yaw = self.robot_yaw if single_pose_yaw is None else float(single_pose_yaw)
        for index, (x, y) in enumerate(points):
            if len(points) == 1:
                tangent_x = 0.0
                tangent_y = 0.0
            elif index == 0:
                next_x, next_y = points[index + 1]
                tangent_x = next_x - x
                tangent_y = next_y - y
            elif index == len(points) - 1:
                previous_x, previous_y = points[index - 1]
                tangent_x = x - previous_x
                tangent_y = y - previous_y
            else:
                previous_x, previous_y = points[index - 1]
                next_x, next_y = points[index + 1]
                tangent_x = next_x - previous_x
                tangent_y = next_y - previous_y
            if math.hypot(tangent_x, tangent_y) > 1e-06:
                last_yaw = math.atan2(tangent_y, tangent_x)
            orientation_z, orientation_w = self._yaw_quaternion(last_yaw)
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = stamp
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = orientation_z
            pose.pose.orientation.w = orientation_w
            poses.append(pose)
        return poses

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _segment_heading(first: WorldPoint, second: WorldPoint) -> Optional[float]:
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        if math.hypot(dx, dy) <= 1e-06:
            return None
        return math.atan2(dy, dx)

    @staticmethod
    def _polyline_length(points: Sequence[WorldPoint]) -> float:
        if len(points) < 2:
            return 0.0
        return sum(
            (math.dist(first, second) for first, second in zip(points[:-1], points[1:]))
        )

    @staticmethod
    def _append_without_duplicate(
        first: Sequence[WorldPoint], second: Sequence[WorldPoint]
    ) -> List[WorldPoint]:
        merged = list(first)
        for point in second:
            if not merged or math.dist(merged[-1], point) > 0.0001:
                merged.append(point)
        return merged

    def _merge_short_motion_chunks(
        self, chunks: Sequence[Sequence[WorldPoint]]
    ) -> Tuple[List[List[WorldPoint]], int]:
        source = [list(chunk) for chunk in chunks if len(chunk) >= 2]
        if not source:
            return ([], 0)
        result: List[List[WorldPoint]] = []
        short_buffer: Optional[List[WorldPoint]] = None
        short_chunk_count = 0
        short_group_count = 0

        def flush_short_buffer() -> None:
            nonlocal short_buffer, short_group_count
            if short_buffer is None:
                return
            result.append(short_buffer)
            short_buffer = None
            short_group_count += 1

        for chunk in source:
            length_m = self._polyline_length(chunk)
            is_short = length_m < self.minimum_motion_chunk_length_m
            if is_short:
                short_chunk_count += 1
                if short_buffer is None:
                    short_buffer = list(chunk)
                else:
                    short_buffer = self._append_without_duplicate(short_buffer, chunk)
                continue
            flush_short_buffer()
            result.append(list(chunk))
        flush_short_buffer()
        merge_count = max(0, short_chunk_count - short_group_count)
        return (result, merge_count)

    def _closest_progress_on_polyline(
        self, points: Sequence[WorldPoint], robot_world: WorldPoint
    ) -> Tuple[float, float]:
        if len(points) < 2:
            return (0.0, math.dist(robot_world, points[0]) if points else math.inf)
        best_progress = 0.0
        best_lateral = math.inf
        cumulative = 0.0
        rx, ry = robot_world
        for first, second in zip(points[:-1], points[1:]):
            ax, ay = first
            bx, by = second
            dx = bx - ax
            dy = by - ay
            length_squared = dx * dx + dy * dy
            segment_length = math.sqrt(length_squared)
            if segment_length <= 1e-09:
                continue
            projection_ratio = ((rx - ax) * dx + (ry - ay) * dy) / length_squared
            projection_ratio = min(1.0, max(0.0, projection_ratio))
            projection = (ax + projection_ratio * dx, ay + projection_ratio * dy)
            lateral = math.dist(robot_world, projection)
            progress = cumulative + projection_ratio * segment_length
            if lateral < best_lateral - 1e-06 or (
                abs(lateral - best_lateral) <= 1e-06 and progress > best_progress
            ):
                best_lateral = lateral
                best_progress = progress
            cumulative += segment_length
        return (best_progress, best_lateral)

    def _chunk_kind_and_turn(self, chunk: Sequence[WorldPoint]) -> Tuple[str, float]:
        internal_turns_deg: List[float] = []
        for first, corner, third in zip(chunk[:-2], chunk[1:-1], chunk[2:]):
            incoming = self._segment_heading(first, corner)
            outgoing = self._segment_heading(corner, third)
            if incoming is None or outgoing is None:
                continue
            internal_turns_deg.append(
                abs(math.degrees(self._normalize_angle(outgoing - incoming)))
            )
        maximum_internal_turn_deg = max(internal_turns_deg, default=0.0)
        transition_length_limit_m = max(0.8, 2.0 * self.line_spacing_m)
        chunk_length_m = self._polyline_length(chunk)
        chunk_kind = (
            "transition"
            if chunk_length_m < transition_length_limit_m
            or maximum_internal_turn_deg > math.degrees(self.sweep_straightness_rad)
            else "sweep"
        )
        return (chunk_kind, maximum_internal_turn_deg)

    def _prepare_chunk_execution_points(
        self, chunk: Sequence[WorldPoint], robot_world: WorldPoint
    ) -> List[WorldPoint]:
        points = list(chunk)
        if len(points) < 2:
            return points
        cumulative = [0.0]
        for first, second in zip(points[:-1], points[1:]):
            cumulative.append(cumulative[-1] + math.dist(first, second))
        first_keep_index = 1 if self.current_chunk_index > 0 else 0
        progress = 0.0
        lateral = math.inf
        if self.current_chunk_retry_count > 0:
            progress, lateral = self._closest_progress_on_polyline(points, robot_world)
            if lateral <= self.chunk_projection_tolerance_m:
                passed_limit = progress + self.waypoint_pass_radius_m
                for index in range(len(points) - 1):
                    if cumulative[index] <= passed_limit:
                        first_keep_index = max(first_keep_index, index + 1)
        first_keep_index = min(first_keep_index, len(points) - 1)
        _, _, _, exclusion_keepout_mask = self._build_masks()
        navigation_mask = cv2.bitwise_and(
            self.safe_free_mask, cv2.bitwise_not(exclusion_keepout_mask)
        )
        robot_pixel = self.world_to_pixel(robot_world)
        requested_keep_index = first_keep_index
        while first_keep_index > 0:
            target_pixel = self.world_to_pixel(points[first_keep_index])
            if (
                self._pixel_inside(robot_pixel)
                and self._pixel_inside(target_pixel)
                and self._line_is_safe(robot_pixel, target_pixel, navigation_mask)
            ):
                break
            first_keep_index -= 1
        if first_keep_index < requested_keep_index:
            pass
        prepared = points[first_keep_index:]
        if len(prepared) < len(points):
            chunk_kind, _ = self._chunk_kind_and_turn(points)
            if chunk_kind == "transition":
                pass
        if first_keep_index > 0:
            if self.current_chunk_retry_count > 0:
                pass
            else:
                pass
        return prepared

    def _point_at_path_distance(
        self, points: Sequence[WorldPoint], distance_m: float
    ) -> Optional[WorldPoint]:
        if not points:
            return None
        if len(points) == 1 or distance_m <= 0.0:
            return points[0]
        remaining = distance_m
        for first, second in zip(points[:-1], points[1:]):
            segment_length = math.dist(first, second)
            if segment_length <= 1e-06:
                continue
            if remaining <= segment_length:
                ratio = remaining / segment_length
                return (
                    first[0] + ratio * (second[0] - first[0]),
                    first[1] + ratio * (second[1] - first[1]),
                )
            remaining -= segment_length
        return points[-1]

    def _split_path_into_chunks(
        self, points: Sequence[WorldPoint]
    ) -> List[List[WorldPoint]]:
        cleaned: List[WorldPoint] = []
        for point in points:
            if not cleaned or math.dist(cleaned[-1], point) > 0.0001:
                cleaned.append(point)
        if len(cleaned) < 2:
            return []
        split_indices = [0]
        for index in range(1, len(cleaned) - 1):
            incoming = self._segment_heading(cleaned[index - 1], cleaned[index])
            outgoing = self._segment_heading(cleaned[index], cleaned[index + 1])
            if incoming is None or outgoing is None:
                continue
            heading_change = abs(self._normalize_angle(outgoing - incoming))
            if heading_change >= self.sharp_turn_split_rad:
                if index - split_indices[-1] >= 1:
                    split_indices.append(index)
        if split_indices[-1] != len(cleaned) - 1:
            split_indices.append(len(cleaned) - 1)
        chunks: List[List[WorldPoint]] = []
        for left, right in zip(split_indices[:-1], split_indices[1:]):
            chunk = cleaned[left : right + 1]
            if len(chunk) >= 2:
                chunks.append(chunk)
        if not chunks:
            chunks = [cleaned]
        merged_chunks, merge_count = self._merge_short_motion_chunks(chunks)
        if merge_count > 0:
            pass
        return merged_chunks

    def _reset_staged_navigation(self) -> None:
        self._stop_no_progress_watchdog()
        if self.actual_coverage_tracking_active:
            self._finish_actual_coverage_tracking()
        self.route_chunks = []
        self.current_chunk_index = 0
        self.current_execution_points = []
        self.current_chunk_retry_count = 0
        self.initial_approach_world = None
        self.initial_approach_yaw = None
        self.initial_approach_done = False
        self.pending_spin_angle = 0.0
        self.current_action_kind = None
        self.send_goal_future = None
        self.get_result_future = None
        self.active_goal_handle = None
        self._publish_path_message(self.active_chunk_pub, [])

    def _start_staged_navigation(self, points: Sequence[WorldPoint]) -> None:
        chunks = self._split_path_into_chunks(points)
        if not chunks:
            self.get_logger().error("insufficient poses")
            return
        self.route_chunks = chunks
        self.current_chunk_index = 0
        self.current_chunk_retry_count = 0
        self.initial_approach_done = False
        first_chunk = chunks[0]
        first_sweep_heading = self._segment_heading(first_chunk[0], first_chunk[1])
        if first_sweep_heading is not None and self.start_entry_pixel is not None:
            self.initial_approach_world = self.pixel_to_world(self.start_entry_pixel)
            self.initial_approach_yaw = first_sweep_heading
        else:
            self.initial_approach_world = None
            self.initial_approach_yaw = None
            self.initial_approach_done = True
        self._begin_current_chunk()

    def _make_pose_stamped(self, point: WorldPoint, yaw: float) -> PoseStamped:
        orientation_z, orientation_w = self._yaw_quaternion(yaw)
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(point[0])
        pose.pose.position.y = float(point[1])
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = orientation_z
        pose.pose.orientation.w = orientation_w
        return pose

    def _send_initial_approach(self, target: WorldPoint, target_yaw: float) -> None:
        if not self.approach_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("navigate_to_pose unavailable")
            self._reset_staged_navigation()
            return
        goal = NavigateToPose.Goal()
        goal.pose = self._make_pose_stamped(target, target_yaw)
        goal.behavior_tree = ""
        self.current_action_kind = "approach"
        self.send_goal_future = self.approach_client.send_goal_async(
            goal, feedback_callback=self._approach_feedback_callback
        )
        self.send_goal_future.add_done_callback(self._approach_goal_response_callback)

    def _approach_goal_response_callback(self, future) -> None:
        self.send_goal_future = None
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"entry send: {exc}")
            self._reset_staged_navigation()
            return
        if not goal_handle.accepted:
            self.get_logger().error("entry rejected")
            self._reset_staged_navigation()
            return
        self.active_goal_handle = goal_handle
        self._start_no_progress_watchdog("approach")
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self._approach_result_callback)

    def _approach_feedback_callback(self, feedback_msg) -> None:
        self._observe_no_progress_watchdog(feedback_msg.feedback.distance_remaining)

    def _approach_result_callback(self, future) -> None:
        self.get_result_future = None
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"entry result: {exc}")
            self._reset_staged_navigation()
            self.cancel_pending = False
            return

        self.active_goal_handle = None
        watchdog_triggered = self._consume_watchdog_trigger("approach")

        if response.status == GoalStatus.STATUS_SUCCEEDED:
            self.initial_approach_done = True
            self.current_action_kind = None
            self._begin_current_chunk()
            return

        if (
            response.status == GoalStatus.STATUS_ABORTED
            and self.initial_approach_world is not None
        ):
            robot_world = self._get_robot_world_position(log_error=False)
            entry_tolerance = max(self.chunk_start_tolerance_m, 0.30)
            if (
                robot_world is not None
                and math.dist(robot_world, self.initial_approach_world)
                <= entry_tolerance
            ):
                self.get_logger().warning("entry accepted near target")
                self.initial_approach_done = True
                self.current_action_kind = None
                self._begin_current_chunk()
                return

        if response.status == GoalStatus.STATUS_CANCELED:
            if watchdog_triggered:
                self.get_logger().warning("entry stalled")
                self.initial_approach_done = True
                self.current_action_kind = None
                self._begin_current_chunk()
                return
            self.get_logger().warning("entry canceled")
        else:
            result = response.result
            self.get_logger().error(
                f"entry failed status={response.status}, "
                f"code={getattr(result, 'error_code', 0)}, "
                f"message={getattr(result, 'error_msg', '')}"
            )

        self._reset_staged_navigation()
        self.cancel_pending = False

    def _current_chunk(self) -> Optional[List[WorldPoint]]:
        if not 0 <= self.current_chunk_index < len(self.route_chunks):
            return None
        return self.route_chunks[self.current_chunk_index]

    def _target_heading_for_chunk(
        self,
        execution_points: Sequence[WorldPoint],
        original_chunk: Sequence[WorldPoint],
        robot_world: WorldPoint,
    ) -> Optional[float]:
        if not execution_points:
            return None
        if len(execution_points) >= 2:
            lookahead_point = self._point_at_path_distance(
                execution_points, self.spin_lookahead_distance_m
            )
            if lookahead_point is not None:
                heading = self._segment_heading(robot_world, lookahead_point)
                if heading is not None:
                    return heading
            heading = self._segment_heading(execution_points[0], execution_points[1])
            if heading is not None:
                return heading
        chunk_kind, _ = self._chunk_kind_and_turn(original_chunk)
        if chunk_kind == "transition":
            return self._segment_heading(robot_world, execution_points[0])
        if len(original_chunk) >= 2:
            return self._segment_heading(original_chunk[-2], original_chunk[-1])
        return None

    def _begin_current_chunk(self) -> None:
        chunk = self._current_chunk()
        if chunk is None:
            self._finish_actual_coverage_tracking()
            self._reset_staged_navigation()
            self.cancel_pending = False
            return
        robot_world = self._get_robot_world_position(log_error=True)
        if robot_world is None:
            self.get_logger().error("robot TF unavailable")
            self._reset_staged_navigation()
            return
        if (
            self.current_chunk_index == 0
            and (not self.initial_approach_done)
            and (self.initial_approach_world is not None)
            and (self.initial_approach_yaw is not None)
        ):
            approach_distance = math.dist(robot_world, self.initial_approach_world)
            if approach_distance > self.chunk_start_tolerance_m:
                self._send_initial_approach(
                    self.initial_approach_world, self.initial_approach_yaw
                )
                return
            self.initial_approach_done = True
        execution_points = self._prepare_chunk_execution_points(chunk, robot_world)
        if not execution_points:
            self.get_logger().error("empty chunk")
            self._reset_staged_navigation()
            return
        self.current_execution_points = execution_points
        target_yaw = self._target_heading_for_chunk(
            execution_points, chunk, robot_world
        )
        if target_yaw is None:
            self.get_logger().error("invalid chunk direction")
            self._reset_staged_navigation()
            return
        heading_error = self._normalize_angle(target_yaw - self.robot_yaw)
        self._publish_active_chunk_visualization(chunk)
        chunk_length_m = self._polyline_length(chunk)
        approach_length_m = math.dist(robot_world, chunk[0])
        effective_path_length_m = chunk_length_m + approach_length_m
        spin_allowed = effective_path_length_m >= self.minimum_spin_path_length_m
        chunk_kind, maximum_internal_turn_deg = self._chunk_kind_and_turn(chunk)
        spin_threshold_rad = (
            self.transition_pre_spin_threshold_rad
            if chunk_kind == "transition"
            else self.pre_spin_threshold_rad
        )
        if chunk_kind == "sweep" and maximum_internal_turn_deg > math.degrees(
            self.sweep_straightness_rad
        ):
            self.get_logger().error("unsafe sweep turn")
            self._reset_staged_navigation()
            return
        if (
            self.pre_spin_enabled
            and spin_allowed
            and (abs(heading_error) >= spin_threshold_rad)
        ):
            self._send_spin_goal(heading_error)
        else:
            self._send_current_chunk()

    def _send_spin_goal(self, relative_yaw: float) -> None:
        if not self.spin_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warning("spin unavailable")
            self._send_current_chunk()
            return
        goal = Spin.Goal()
        goal.target_yaw = float(relative_yaw)
        goal.time_allowance = Duration(seconds=self.pre_spin_time_allowance_s).to_msg()
        self.pending_spin_angle = float(relative_yaw)
        self.current_action_kind = "spin"
        self.send_goal_future = self.spin_client.send_goal_async(goal)
        self.send_goal_future.add_done_callback(self._spin_goal_response_callback)

    def _spin_goal_response_callback(self, future) -> None:
        self.send_goal_future = None
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"spin send: {exc}")
            self._reset_staged_navigation()
            return
        if not goal_handle.accepted:
            self.get_logger().error("spin rejected")
            self._reset_staged_navigation()
            return
        self.active_goal_handle = goal_handle
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self._spin_result_callback)

    def _spin_result_callback(self, future) -> None:
        self.get_result_future = None
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"spin result: {exc}")
            self._reset_staged_navigation()
            self.cancel_pending = False
            return

        self.active_goal_handle = None

        if response.status == GoalStatus.STATUS_SUCCEEDED:
            self.current_action_kind = None
            self.pending_spin_angle = 0.0
            self._send_current_chunk()
            return

        result = response.result
        error_code = int(getattr(result, "error_code", 0))
        collision_ahead = int(getattr(Spin.Result, "COLLISION_AHEAD", 703))

        if (
            response.status == GoalStatus.STATUS_ABORTED
            and error_code == collision_ahead
        ):
            self.get_logger().warning("spin blocked; continuing with path")
            self.current_action_kind = None
            self.pending_spin_angle = 0.0
            self._send_current_chunk()
            return

        if response.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warning("spin canceled")
        else:
            self.get_logger().error(
                f"spin failed code={error_code}, "
                f"message={getattr(result, 'error_msg', '')}"
            )

        self._reset_staged_navigation()
        self.cancel_pending = False

    def _send_current_chunk(self) -> None:
        chunk = self._current_chunk()
        if chunk is None or len(chunk) < 2:
            self.get_logger().error("chunk too short")
            self._reset_staged_navigation()
            return
        if not self.nav_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("navigate_through_poses unavailable")
            self._reset_staged_navigation()
            return
        robot_world = self._get_robot_world_position(log_error=False)
        execution_points = list(self.current_execution_points)
        if not execution_points:
            if robot_world is None:
                execution_points = list(chunk)
            else:
                execution_points = self._prepare_chunk_execution_points(
                    chunk, robot_world
                )
        if not execution_points:
            self.get_logger().error("route empty after pruning")
            self._reset_staged_navigation()
            return
        chunk_kind, _ = self._chunk_kind_and_turn(chunk)
        if len(execution_points) == 1 and chunk_kind == "transition":
            if robot_world is not None:
                final_yaw = self._segment_heading(robot_world, execution_points[0])
            else:
                final_yaw = None
        else:
            final_yaw = self._segment_heading(chunk[-2], chunk[-1])
        if final_yaw is None:
            final_yaw = self.robot_yaw
        goal = NavigateThroughPoses.Goal()
        goal.poses = self._build_pose_messages(
            execution_points, single_pose_yaw=final_yaw
        )
        goal.behavior_tree = self.coverage_bt_path
        self.current_action_kind = "coverage"
        self.send_goal_future = self.nav_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback
        )
        self.send_goal_future.add_done_callback(self._coverage_goal_response_callback)

    def _coverage_goal_response_callback(self, future) -> None:
        self.send_goal_future = None
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"chunk send: {exc}")
            self._reset_staged_navigation()
            return
        if not goal_handle.accepted:
            self.get_logger().error("chunk rejected")
            self._reset_staged_navigation()
            return
        self.active_goal_handle = goal_handle
        self._start_no_progress_watchdog("coverage")
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self._coverage_chunk_result_callback)

    def _feedback_callback(self, feedback_msg) -> None:
        self._observe_no_progress_watchdog(feedback_msg.feedback.distance_remaining)

    def _coverage_chunk_result_callback(self, future) -> None:
        self.get_result_future = None
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"chunk result: {exc}")
            self._reset_staged_navigation()
            self.cancel_pending = False
            return
        self.active_goal_handle = None
        watchdog_triggered = self._consume_watchdog_trigger("coverage")
        if response.status == GoalStatus.STATUS_SUCCEEDED:
            self.current_chunk_index += 1
            self.current_chunk_retry_count = 0
            self.current_execution_points = []
            self.current_action_kind = None
            self._begin_current_chunk()
            return
        if response.status == GoalStatus.STATUS_CANCELED:
            if watchdog_triggered:
                if self.current_chunk_retry_count < self.chunk_retry_limit:
                    self.current_chunk_retry_count += 1
                    self.current_execution_points = []
                    self.current_action_kind = None
                    self.get_logger().warning(
                        f"Chunk stalled; retry {self.current_chunk_retry_count}/{self.chunk_retry_limit}."
                    )
                    self._begin_current_chunk()
                    return
                self.get_logger().error("chunk stalled")
                self._reset_staged_navigation()
                self.cancel_pending = False
                return
            self.get_logger().warning("mission canceled")
            self._reset_staged_navigation()
            self.cancel_pending = False
            return
        if self.current_chunk_retry_count < self.chunk_retry_limit:
            self.current_chunk_retry_count += 1
            self.current_execution_points = []
            self.current_action_kind = None
            self.get_logger().warning(
                f"Chunk failed; retry {self.current_chunk_retry_count}/{self.chunk_retry_limit}."
            )
            self._begin_current_chunk()
            return
        self.get_logger().error("mission failed")
        self._reset_staged_navigation()
        self.cancel_pending = False

    def cancel_current_goal(self) -> None:
        if self.watchdog_cancel_requested:
            self.get_logger().warning("watchdog cancel pending")
            return
        if self.cancel_pending:
            self.get_logger().warning("cancel pending")
            return
        if self.active_goal_handle is None:
            if self.send_goal_future is not None or self.route_chunks:
                self.get_logger().warning("goal pending")
            return
        self.cancel_pending = True
        cancel_future = self.active_goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self._cancel_response_callback)

    def _cancel_response_callback(self, future) -> None:
        try:
            response = future.result()
            if not response.goals_canceling:
                self.cancel_pending = False
                self.get_logger().warning("cancel rejected")
        except Exception as exc:
            self.cancel_pending = False
            self.get_logger().error(f"cancel failed: {exc}")
