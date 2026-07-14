#!/usr/bin/env python3
from __future__ import annotations
from typing import Optional
import cv2
import rclpy
from coverage_modules import (
    CoverageMapMixin,
    CoverageMissionMixin,
    CoveragePlannerMixin,
    CoverageSetupMixin,
    CoverageUiMixin,
    CoverageVisualizationMixin,
)
from rclpy.node import Node


class CoverageNavigator(
    CoverageSetupMixin,
    CoverageMapMixin,
    CoveragePlannerMixin,
    CoverageVisualizationMixin,
    CoverageUiMixin,
    CoverageMissionMixin,
    Node,
):
    WINDOW_NAME = "Robot ROS2 Coverage Navigator"

    def plan_and_send(self) -> None:
        if not self.coverage_polygons:
            self.get_logger().error("coverage polygon missing")
            return
        if (
            self.active_goal_handle is not None
            or self.send_goal_future is not None
            or self.get_result_future is not None
            or self.route_chunks
            or self.cancel_pending
        ):
            self.get_logger().warning("mission active")
            return
        robot_world = self._get_robot_world_position(log_error=True)
        if robot_world is None:
            return
        robot_pixel = self.world_to_pixel(robot_world)
        if not self._pixel_inside(robot_pixel):
            self.get_logger().error(f"robot outside map: {robot_world}")
            return
        _, safe_selected_mask, _, exclusion_keepout_mask = self._build_masks()
        if cv2.countNonZero(safe_selected_mask) == 0:
            self.get_logger().error("no safe coverage")
            return
        navigation_mask = cv2.bitwise_and(
            self.safe_free_mask, cv2.bitwise_not(exclusion_keepout_mask)
        )
        if navigation_mask[robot_pixel[1], robot_pixel[0]] != 255:
            self.get_logger().error("robot inside keepout")
            return
        reachable_mask = self._reachable_mask_from_robot(robot_pixel, navigation_mask)
        if reachable_mask is None:
            return
        validated_mask = self._reject_unreachable_components(
            safe_selected_mask, reachable_mask
        )
        if validated_mask is None:
            return
        path_pixels = self._generate_boustrophedon_path(
            validated_mask, reachable_mask, robot_pixel, self.robot_yaw
        )
        if len(path_pixels) < 2:
            self.get_logger().error("route too short")
            return
        nav_source_pixels = list(path_pixels)
        if (
            self.start_entry_pixel is not None
            and nav_source_pixels
            and (nav_source_pixels[0] == self.start_entry_pixel)
        ):
            nav_source_pixels = nav_source_pixels[1:]
        nav_path_pixels = self._simplify_for_nav2(nav_source_pixels, reachable_mask)
        if len(nav_path_pixels) < 2:
            self.get_logger().error("simplified route too short")
            return
        self.path_pixels = path_pixels
        self.path_world = [self.pixel_to_world(point) for point in path_pixels]
        nav_path_world = [self.pixel_to_world(point) for point in nav_path_pixels]
        self.nav_path_pixels = list(nav_path_pixels)
        self.nav_path_world = list(nav_path_world)
        self.mandatory_path_world = [
            self.pixel_to_world(point) for point in self.mandatory_path_pixels
        ]
        self._publish_rviz_visualization()
        self._publish_keepout_filter()
        self._start_actual_coverage_tracking(validated_mask)
        self._start_staged_navigation(nav_path_world)

    def close(self) -> None:
        cv2.destroyAllWindows()
        self.destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[CoverageNavigator] = None
    try:
        node = CoverageNavigator()
        while rclpy.ok() and node.running:
            node.ui_step()
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is None:
            raise
        node.get_logger().fatal(f"startup: {exc}")
    finally:
        if node is not None:
            if node.active_goal_handle is not None and rclpy.ok():
                try:
                    node.cancel_current_goal()
                    for _ in range(20):
                        if not rclpy.ok():
                            break
                        rclpy.spin_once(node, timeout_sec=0.05)
                except Exception:
                    pass
            try:
                node.close()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
