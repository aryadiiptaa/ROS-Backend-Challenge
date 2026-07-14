from __future__ import annotations
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple
import cv2
import numpy as np
import yaml
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from .types import PixelPoint, Polygon, WorldPoint


class CoverageMapMixin:

    def _load_map(self) -> Tuple[np.ndarray, Path, float, Tuple[float, float]]:
        configured_path = str(self.get_parameter("map_yaml").value).strip()
        if configured_path:
            map_yaml_path = Path(configured_path).expanduser().resolve()
        else:
            map_yaml_path = self.package_share / "maps" / "arena_map.yaml"
        if not map_yaml_path.exists():
            raise FileNotFoundError(f"map yaml missing: {map_yaml_path}")
        with map_yaml_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        image_value = Path(str(config["image"])).expanduser()
        if image_value.is_absolute():
            map_image_path = image_value
        else:
            map_image_path = (map_yaml_path.parent / image_value).resolve()
        map_image = cv2.imread(str(map_image_path), cv2.IMREAD_GRAYSCALE)
        if map_image is None:
            raise RuntimeError(f"map image read failed: {map_image_path}")
        resolution = float(config["resolution"])
        origin = (float(config["origin"][0]), float(config["origin"][1]))
        return (map_image, map_image_path, resolution, origin)

    def _make_safe_free_mask(self) -> np.ndarray:
        radius_px = max(1, int(math.ceil(self.safety_margin_m / self.resolution)))
        kernel_size = 2 * radius_px + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        return cv2.erode(self.free_mask, kernel, iterations=1)

    def pixel_to_world(self, point: PixelPoint) -> WorldPoint:
        px, py = point
        origin_x, origin_y = self.origin
        x = origin_x + (px + 0.5) * self.resolution
        y = origin_y + (self.image_height - py - 0.5) * self.resolution
        return (round(x, 4), round(y, 4))

    def world_to_pixel(self, point: WorldPoint) -> PixelPoint:
        x, y = point
        origin_x, origin_y = self.origin
        px = int(math.floor((x - origin_x) / self.resolution))
        map_y = int(math.floor((y - origin_y) / self.resolution))
        py = self.image_height - 1 - map_y
        return (px, py)

    @staticmethod
    def _fill_polygons(
        shape: Tuple[int, int], polygons: Sequence[Polygon]
    ) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        for polygon in polygons:
            points = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [points], 255)
        return mask

    def _inflate_mask(self, mask: np.ndarray, margin_m: float) -> np.ndarray:
        if cv2.countNonZero(mask) == 0 or margin_m <= 0.0:
            return mask.copy()
        radius_px = max(1, int(math.ceil(margin_m / self.resolution)))
        kernel_size = 2 * radius_px + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        return cv2.dilate(mask, kernel, iterations=1)

    def _build_masks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        coverage_mask = self._fill_polygons(
            self.map_image.shape, self.coverage_polygons
        )
        exclusion_mask = self._fill_polygons(
            self.map_image.shape, self.exclusion_polygons
        )
        exclusion_keepout_mask = self._inflate_mask(
            exclusion_mask, self.exclusion_margin_m
        )
        selected_mask = cv2.bitwise_and(
            coverage_mask, cv2.bitwise_not(exclusion_keepout_mask)
        )
        safe_selected_mask = cv2.bitwise_and(selected_mask, self.safe_free_mask)
        return (
            selected_mask,
            safe_selected_mask,
            exclusion_mask,
            exclusion_keepout_mask,
        )

    def _pixel_inside(self, point: PixelPoint) -> bool:
        px, py = point
        return 0 <= px < self.image_width and 0 <= py < self.image_height

    def _get_robot_world_position(self, log_error: bool = True) -> Optional[WorldPoint]:
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", self.base_frame, Time(), timeout=Duration(seconds=0.15)
            )
        except TransformException as exc:
            if log_error:
                self.get_logger().error(f"TF unavailable: {self.base_frame}: {exc}")
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        siny_cosp = 2.0 * (rotation.w * rotation.z + rotation.x * rotation.y)
        cosy_cosp = 1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        return (float(translation.x), float(translation.y))

    def _nearest_safe_pixel(
        self, point: PixelPoint, mask: Optional[np.ndarray] = None, max_radius: int = 12
    ) -> Optional[PixelPoint]:
        px, py = point
        search_mask = self.safe_free_mask if mask is None else mask
        if self._pixel_inside(point) and search_mask[py, px] == 255:
            return point
        for radius in range(1, max_radius + 1):
            x_min = max(0, px - radius)
            x_max = min(self.image_width - 1, px + radius)
            y_min = max(0, py - radius)
            y_max = min(self.image_height - 1, py + radius)
            region = search_mask[y_min : y_max + 1, x_min : x_max + 1]
            coordinates = np.argwhere(region == 255)
            if coordinates.size > 0:
                distances = (coordinates[:, 1] + x_min - px) ** 2 + (
                    coordinates[:, 0] + y_min - py
                ) ** 2
                best = coordinates[int(np.argmin(distances))]
                return (int(best[1] + x_min), int(best[0] + y_min))
        return None

    def _reachable_mask_from_robot(
        self, robot_pixel: PixelPoint, navigation_mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        source_mask = (
            self.safe_free_mask if navigation_mask is None else navigation_mask
        )
        seed = self._nearest_safe_pixel(robot_pixel, source_mask)
        if seed is None:
            self.get_logger().error("robot outside safe map")
            return None
        _, labels = cv2.connectedComponents(source_mask, connectivity=4)
        robot_label = int(labels[seed[1], seed[0]])
        if robot_label == 0:
            return None
        return np.where(labels == robot_label, 255, 0).astype(np.uint8)

    def _reject_unreachable_components(
        self, safe_selected_mask: np.ndarray, reachable_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        reachable_selection = cv2.bitwise_and(safe_selected_mask, reachable_mask)
        unreachable_selection = cv2.bitwise_and(
            safe_selected_mask, cv2.bitwise_not(reachable_mask)
        )
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            unreachable_selection, connectivity=8
        )
        del labels
        significant_components = []
        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= self.minimum_component_pixels:
                significant_components.append(area)
        if significant_components:
            self.get_logger().error("coverage unreachable")
            return None
        if cv2.countNonZero(reachable_selection) == 0:
            self.get_logger().error("no reachable coverage")
            return None
        return reachable_selection
