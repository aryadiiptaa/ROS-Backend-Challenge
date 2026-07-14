from __future__ import annotations
import cv2


class CoverageUiMixin:

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._save_current_polygon(as_exclusion=self.draw_mode == "exclusion")

    def _save_current_polygon(self, as_exclusion: bool) -> None:
        if len(self.current_polygon) < 3:
            self.get_logger().warning("polygon needs 3 points")
            return
        polygon = list(self.current_polygon)
        if as_exclusion:
            self.exclusion_polygons.append(polygon)
        else:
            self.coverage_polygons.append(polygon)
        self.current_polygon.clear()
        self.path_pixels.clear()
        self.path_world.clear()
        self.start_entry_pixel = None
        self.nav_path_pixels.clear()
        self.nav_path_world.clear()
        self.mandatory_path_pixels.clear()
        self.mandatory_path_world.clear()
        self._reset_actual_coverage_tracking(clear_target=True)
        self._publish_keepout_filter()
        self._publish_rviz_visualization()

    def ui_step(self) -> None:
        cv2.imshow(self.WINDOW_NAME, self._render())
        key = cv2.waitKey(10) & 255
        if key == 27:
            self.running = False
        elif key in (ord("a"), ord("A")):
            self.draw_mode = "coverage"
            self.current_polygon.clear()
        elif key in (ord("d"), ord("D")):
            self.draw_mode = "exclusion"
            self.current_polygon.clear()
        elif key == ord("c"):
            if (
                self.active_goal_handle is not None
                or self.send_goal_future is not None
                or self.cancel_pending
            ):
                self.get_logger().warning("mission active")
            else:
                self.coverage_polygons.clear()
                self.exclusion_polygons.clear()
                self.current_polygon.clear()
                self.path_pixels.clear()
                self.path_world.clear()
                self.nav_path_pixels.clear()
                self.nav_path_world.clear()
                self.mandatory_path_pixels.clear()
                self.mandatory_path_world.clear()
                self.start_entry_pixel = None
                self._reset_actual_coverage_tracking(clear_target=True)
                self._publish_keepout_filter()
                self._clear_rviz_visualization()
        elif key in (ord("e"), ord("E")):
            self._save_current_polygon(as_exclusion=True)
        elif key == ord("p"):
            self.plan_and_send()
        elif key == ord("x"):
            self.cancel_current_goal()
