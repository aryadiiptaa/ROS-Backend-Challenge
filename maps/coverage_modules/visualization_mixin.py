from __future__ import annotations
from typing import Optional, Sequence
import cv2
import numpy as np
from geometry_msgs.msg import Point
from nav2_msgs.msg import CostmapFilterInfo
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path as NavPathMessage
from visualization_msgs.msg import Marker, MarkerArray
from .types import Polygon, WorldPoint


class CoverageVisualizationMixin:

    def _mask_to_occupancy_grid(self, mask: np.ndarray) -> OccupancyGrid:
        message = OccupancyGrid()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        message.info.map_load_time = message.header.stamp
        message.info.resolution = float(self.resolution)
        message.info.width = int(self.image_width)
        message.info.height = int(self.image_height)
        message.info.origin.position.x = float(self.origin[0])
        message.info.origin.position.y = float(self.origin[1])
        message.info.origin.position.z = 0.0
        message.info.origin.orientation.w = 1.0
        values = np.where(mask > 0, 100, 0).astype(np.int8)
        message.data = np.flipud(values).reshape(-1).tolist()
        return message

    def _publish_keepout_filter(self=False) -> None:
        if not self.keepout_filter_enabled:
            return
        _, _, _, exclusion_keepout_mask = self._build_masks()
        info = CostmapFilterInfo()
        info.header.frame_id = "map"
        info.header.stamp = self.get_clock().now().to_msg()
        info.type = 0
        info.filter_mask_topic = self.keepout_filter_mask_topic
        info.base = 0.0
        info.multiplier = 1.0
        self.keepout_info_pub.publish(info)
        self.keepout_mask_pub.publish(
            self._mask_to_occupancy_grid(exclusion_keepout_mask)
        )

    def _actual_coverage_percentage(self) -> float:
        target = cv2.countNonZero(self.actual_coverage_target_mask)
        if target == 0:
            return 0.0
        covered = cv2.countNonZero(
            cv2.bitwise_and(
                self.actual_coverage_mask,
                self.actual_coverage_target_mask,
            )
        )
        return 100.0 * covered / target

    def _publish_actual_coverage_grid(self) -> None:
        self.actual_coverage_pub.publish(
            self._mask_to_occupancy_grid(self.actual_coverage_mask)
        )

    def _reset_actual_coverage_tracking(self, clear_target: bool) -> None:
        self.actual_coverage_tracking_active = False
        self.actual_coverage_mask.fill(0)
        if clear_target:
            self.actual_coverage_target_mask.fill(0)
        self.actual_coverage_last_pixel = None
        self.actual_coverage_last_publish_ns = 0
        self._publish_actual_coverage_grid()

    def _start_actual_coverage_tracking(self, target_mask: np.ndarray) -> None:
        self.actual_coverage_target_mask = target_mask.copy()
        self.actual_coverage_mask.fill(0)
        self.actual_coverage_last_pixel = None
        now_ns = self.get_clock().now().nanoseconds
        self.actual_coverage_last_publish_ns = now_ns
        self.actual_coverage_tracking_active = (
            self.actual_coverage_tracking_enabled
            and cv2.countNonZero(self.actual_coverage_target_mask) > 0
        )
        self._publish_actual_coverage_grid()

    def _record_actual_coverage_pose(self) -> None:
        robot_world = self._get_robot_world_position(log_error=False)
        if robot_world is None:
            return
        robot_pixel = self.world_to_pixel(robot_world)
        if not self._pixel_inside(robot_pixel):
            return
        width_px = max(1, int(round(self.coverage_tool_width_m / self.resolution)))
        if width_px % 2 == 0:
            width_px += 1
        trail = self.actual_coverage_mask.copy()
        if self.actual_coverage_last_pixel is None:
            cv2.circle(
                trail, robot_pixel, max(1, width_px // 2), 255, -1, lineType=cv2.LINE_8
            )
        else:
            cv2.line(
                trail,
                self.actual_coverage_last_pixel,
                robot_pixel,
                255,
                width_px,
                lineType=cv2.LINE_8,
            )
        self.actual_coverage_mask = cv2.bitwise_and(
            trail, self.actual_coverage_target_mask
        )
        self.actual_coverage_last_pixel = robot_pixel

    def _actual_coverage_tracking_tick(self) -> None:
        if (
            not self.actual_coverage_tracking_active
            or self.current_action_kind != "coverage"
            or self.active_goal_handle is None
        ):
            return
        self._record_actual_coverage_pose()
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.actual_coverage_last_publish_ns >= int(
            self.actual_coverage_publish_period_s * 1000000000
        ):
            self._publish_actual_coverage_grid()
            self.actual_coverage_last_publish_ns = now_ns

    def _finish_actual_coverage_tracking(self) -> None:
        if not self.actual_coverage_tracking_active:
            return
        self._record_actual_coverage_pose()
        self.actual_coverage_tracking_active = False
        self._publish_actual_coverage_grid()

    def _publish_path_message(self, publisher, points: Sequence[WorldPoint]) -> None:
        if not self.rviz_visualization_enabled:
            return
        message = NavPathMessage()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        message.poses = self._build_pose_messages(points)
        publisher.publish(message)

    def _marker_point(self, point: WorldPoint, z: float = 0.03) -> Point:
        output = Point()
        output.x = float(point[0])
        output.y = float(point[1])
        output.z = float(z)
        return output

    def _mask_marker(
        self,
        marker_id: int,
        namespace: str,
        mask: np.ndarray,
        red: float,
        green: float,
        blue: float,
        alpha: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        cell_size = self.resolution * self.rviz_area_sample_step_px * 0.92
        marker.scale.x = cell_size
        marker.scale.y = cell_size
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha
        step = self.rviz_area_sample_step_px
        for py in range(0, self.image_height, step):
            for px in range(0, self.image_width, step):
                if mask[py, px] != 255:
                    continue
                marker.points.append(
                    self._marker_point(self.pixel_to_world((px, py)), z=0.015)
                )
        return marker

    def _polygon_line_marker(
        self,
        marker_id: int,
        namespace: str,
        polygon: Polygon,
        red: float,
        green: float,
        blue: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.045
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = 0.95
        closed = list(polygon)
        if closed and closed[0] != closed[-1]:
            closed.append(closed[0])
        marker.points = [
            self._marker_point(self.pixel_to_world(point), z=0.04) for point in closed
        ]
        return marker

    def _route_line_marker(
        self,
        marker_id: int,
        namespace: str,
        points: Sequence[WorldPoint],
        red: float,
        green: float,
        blue: float,
        width: float,
        z: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = width
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = 0.95
        marker.points = [self._marker_point(point, z=z) for point in points]
        return marker

    def _publish_active_chunk_visualization(
        self, active_chunk: Sequence[WorldPoint]
    ) -> None:
        if not self.rviz_visualization_enabled:
            return
        self._publish_path_message(self.active_chunk_pub, list(active_chunk))

    def _publish_rviz_visualization(
        self, active_chunk: Optional[Sequence[WorldPoint]] = None
    ) -> None:
        if not self.rviz_visualization_enabled:
            return
        _selected_mask, safe_selected_mask, exclusion_mask, exclusion_keepout_mask = (
            self._build_masks()
        )
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)
        marker_array.markers.append(
            self._mask_marker(
                0, "coverage_valid_fill", safe_selected_mask, 0.1, 0.9, 0.2, 0.28
            )
        )
        marker_array.markers.append(
            self._mask_marker(
                1,
                "exclusion_keepout_fill",
                exclusion_keepout_mask,
                1.0,
                0.45,
                0.0,
                0.22,
            )
        )
        marker_array.markers.append(
            self._mask_marker(
                2, "exclusion_fill", exclusion_mask, 0.95, 0.05, 0.05, 0.48
            )
        )
        marker_id = 10
        for polygon in self.coverage_polygons:
            marker_array.markers.append(
                self._polygon_line_marker(
                    marker_id, "coverage_boundary", polygon, 0.1, 0.35, 1.0
                )
            )
            marker_id += 1
        for polygon in self.exclusion_polygons:
            marker_array.markers.append(
                self._polygon_line_marker(
                    marker_id, "exclusion_boundary", polygon, 0.95, 0.05, 0.05
                )
            )
            marker_id += 1
        if self.path_world:
            marker_array.markers.append(
                self._route_line_marker(
                    100,
                    "coverage_visual_route",
                    self.path_world,
                    1.0,
                    0.0,
                    1.0,
                    0.035,
                    0.07,
                )
            )
        if self.nav_path_world:
            marker_array.markers.append(
                self._route_line_marker(
                    101,
                    "nav2_waypoint_route",
                    self.nav_path_world,
                    1.0,
                    0.85,
                    0.0,
                    0.055,
                    0.09,
                )
            )
        mandatory_marker = Marker()
        mandatory_marker.header.frame_id = "map"
        mandatory_marker.header.stamp = self.get_clock().now().to_msg()
        mandatory_marker.ns = "mandatory_waypoints"
        mandatory_marker.id = 102
        mandatory_marker.type = Marker.SPHERE_LIST
        mandatory_marker.action = Marker.ADD
        mandatory_marker.scale.x = 0.14
        mandatory_marker.scale.y = 0.14
        mandatory_marker.scale.z = 0.08
        mandatory_marker.color.r = 1.0
        mandatory_marker.color.g = 0.55
        mandatory_marker.color.b = 0.0
        mandatory_marker.color.a = 1.0
        mandatory_marker.points = [
            self._marker_point(point, z=0.12) for point in self.mandatory_path_world
        ]
        marker_array.markers.append(mandatory_marker)
        if active_chunk:
            marker_array.markers.append(
                self._route_line_marker(
                    103, "active_chunk", active_chunk, 0.0, 1.0, 1.0, 0.075, 0.14
                )
            )
        self.rviz_marker_pub.publish(marker_array)
        self._publish_path_message(self.visual_path_pub, self.path_world)
        self._publish_path_message(self.nav_waypoints_pub, self.nav_path_world)
        self._publish_path_message(
            self.active_chunk_pub, list(active_chunk) if active_chunk else []
        )

    def _clear_rviz_visualization(self) -> None:
        if not self.rviz_visualization_enabled:
            return
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)
        self.rviz_marker_pub.publish(marker_array)
        self._publish_path_message(self.visual_path_pub, [])
        self._publish_path_message(self.nav_waypoints_pub, [])
        self._publish_path_message(self.active_chunk_pub, [])

    def _render(self) -> np.ndarray:
        image = self.display_base.copy()
        selected_mask, safe_selected_mask, exclusion_mask, exclusion_keepout_mask = (
            self._build_masks()
        )
        unsafe_selected = cv2.bitwise_and(
            selected_mask, cv2.bitwise_not(self.safe_free_mask)
        )
        overlay = image.copy()
        overlay[safe_selected_mask == 255] = (0, 255, 0)
        overlay[unsafe_selected == 255] = (0, 0, 255)
        overlay[exclusion_keepout_mask == 255] = (0, 165, 255)
        overlay[exclusion_mask == 255] = (0, 0, 220)
        overlay[self.actual_coverage_mask == 255] = (255, 255, 0)
        cv2.addWeighted(overlay, 0.42, image, 0.58, 0.0, image)
        for polygon in self.coverage_polygons:
            cv2.polylines(
                image, [np.asarray(polygon, dtype=np.int32)], True, (255, 0, 0), 2
            )
        for polygon in self.exclusion_polygons:
            cv2.polylines(
                image, [np.asarray(polygon, dtype=np.int32)], True, (0, 0, 180), 2
            )
        if self.current_polygon:
            current_color = (
                (0, 0, 255) if self.draw_mode == "exclusion" else (0, 255, 255)
            )
            points = np.asarray(self.current_polygon, dtype=np.int32)
            if len(points) > 1:
                cv2.polylines(image, [points], False, current_color, 2)
            for point in self.current_polygon:
                cv2.circle(image, point, 3, current_color, -1)
        if self.path_pixels:
            for index in range(len(self.path_pixels) - 1):
                cv2.line(
                    image,
                    self.path_pixels[index],
                    self.path_pixels[index + 1],
                    (255, 0, 255),
                    2,
                )
            for point in self.path_pixels:
                cv2.circle(image, point, 2, (0, 165, 255), -1)
        robot_world = self._get_robot_world_position(log_error=False)
        if robot_world is not None:
            robot_pixel = self.world_to_pixel(robot_world)
            if self._pixel_inside(robot_pixel):
                cv2.circle(image, robot_pixel, 5, (255, 255, 0), -1)
                cv2.putText(
                    image,
                    "R",
                    (robot_pixel[0] + 7, robot_pixel[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        mode_text = "EXCLUSION (D)" if self.draw_mode == "exclusion" else "COVERAGE (A)"
        mode_color = (0, 0, 255) if self.draw_mode == "exclusion" else (255, 0, 0)
        cv2.rectangle(image, (2, 2), (130, 35), (255, 255, 255), -1)
        cv2.putText(
            image,
            mode_text,
            (5, 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            mode_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Cov={len(self.coverage_polygons)} Exc={len(self.exclusion_polygons)}",
            (5, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        actual_percentage = self._actual_coverage_percentage()
        cv2.putText(
            image,
            f"Covered={actual_percentage:.1f}%",
            (5, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        return image
