from __future__ import annotations
import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from .types import PixelPoint


class CoveragePlannerMixin:

    @staticmethod
    def _normalize_vector(dx: float, dy: float) -> Optional[Tuple[float, float]]:
        length = math.hypot(dx, dy)
        if length < 1e-09:
            return None
        return (dx / length, dy / length)

    @staticmethod
    def _bresenham_line(start: PixelPoint, end: PixelPoint) -> List[PixelPoint]:
        x0, y0 = start
        x1, y1 = end
        points: List[PixelPoint] = []
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            error_twice = 2 * error
            if error_twice >= dy:
                error += dy
                x0 += sx
            if error_twice <= dx:
                error += dx
                y0 += sy
        return points

    def _line_is_safe(
        self, start: PixelPoint, end: PixelPoint, mask: np.ndarray
    ) -> bool:
        for point in self._bresenham_line(start, end):
            if not self._pixel_inside(point):
                return False
            if mask[point[1], point[0]] != 255:
                return False
        return True

    def _polyline_is_safe(self, points: Sequence[PixelPoint], mask: np.ndarray) -> bool:
        if not points:
            return False
        for point in points:
            if not self._pixel_inside(point):
                return False
            if mask[point[1], point[0]] != 255:
                return False
        for index in range(len(points) - 1):
            if not self._line_is_safe(points[index], points[index + 1], mask):
                return False
        return True

    @staticmethod
    def _extract_row_segments(
        mask: np.ndarray, y: int, minimum_length_px: int
    ) -> List[Tuple[int, int]]:
        valid_x = np.flatnonzero(mask[y, :] == 255)
        if valid_x.size == 0:
            return []
        split_indices = np.where(np.diff(valid_x) > 1)[0] + 1
        groups = np.split(valid_x, split_indices)
        segments: List[Tuple[int, int]] = []
        for group in groups:
            if group.size < minimum_length_px:
                continue
            segments.append((int(group[0]), int(group[-1])))
        return segments

    def _make_oriented_sweep_segment(
        self, x_min: int, x_max: int, y: int, left_to_right: bool
    ) -> Tuple[PixelPoint, PixelPoint]:
        segment_width = x_max - x_min
        requested_inset_px = max(0, int(round(self.turn_radius_m / self.resolution)))
        maximum_inset_px = max(0, (segment_width - 1) // 3)
        inset_px = min(requested_inset_px, maximum_inset_px)
        usable_left = x_min + inset_px
        usable_right = x_max - inset_px
        if usable_left >= usable_right:
            usable_left = x_min
            usable_right = x_max
        if left_to_right:
            return ((usable_left, y), (usable_right, y))
        return ((usable_right, y), (usable_left, y))

    def _create_smooth_turn(
        self,
        start: PixelPoint,
        end: PixelPoint,
        previous_heading: Tuple[float, float],
        current_heading: Tuple[float, float],
        mask: np.ndarray,
    ) -> List[PixelPoint]:
        previous_unit = self._normalize_vector(previous_heading[0], previous_heading[1])
        current_unit = self._normalize_vector(current_heading[0], current_heading[1])
        if previous_unit is None or current_unit is None:
            return []
        desired_control_px = max(1, int(round(self.turn_radius_m / self.resolution)))
        p0 = np.asarray(start, dtype=np.float64)
        p3 = np.asarray(end, dtype=np.float64)
        previous_vector = np.asarray(previous_unit, dtype=np.float64)
        current_vector = np.asarray(current_unit, dtype=np.float64)
        for control_px in range(desired_control_px, 0, -1):
            p1 = p0 + previous_vector * control_px
            p2 = p3 - current_vector * control_px
            candidate: List[PixelPoint] = []
            for index in range(1, self.turn_steps + 1):
                t = index / self.turn_steps
                one_minus_t = 1.0 - t
                point = (
                    one_minus_t**3 * p0
                    + 3.0 * one_minus_t**2 * t * p1
                    + 3.0 * one_minus_t * t**2 * p2
                    + t**3 * p3
                )
                pixel = (int(round(point[0])), int(round(point[1])))
                if not candidate or pixel != candidate[-1]:
                    candidate.append(pixel)
            if candidate and candidate[-1] != end:
                candidate.append(end)
            full_candidate = [start] + candidate
            if self._polyline_is_safe(full_candidate, mask):
                return candidate
        return []

    @staticmethod
    def _astar_neighbors() -> Sequence[Tuple[int, int, float]]:
        diagonal = math.sqrt(2.0)
        return (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, diagonal),
            (-1, 1, diagonal),
            (1, -1, diagonal),
            (1, 1, diagonal),
        )

    def _find_astar_path(
        self, start: PixelPoint, goal: PixelPoint, mask: np.ndarray
    ) -> List[PixelPoint]:
        if not self._pixel_inside(start):
            return []
        if not self._pixel_inside(goal):
            return []
        if mask[start[1], start[0]] != 255:
            return []
        if mask[goal[1], goal[0]] != 255:
            return []
        if start == goal:
            return [start]
        open_heap: List[Tuple[float, float, int, PixelPoint]] = []
        came_from: Dict[PixelPoint, PixelPoint] = {}
        clearance_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        desired_clearance_px = max(1.0, self.astar_clearance_radius_m / self.resolution)
        g_score: Dict[PixelPoint, float] = {start: 0.0}
        closed = set()
        sequence = 0
        start_h = math.dist(start, goal)
        heapq.heappush(open_heap, (start_h, 0.0, sequence, start))
        while open_heap:
            _, current_g, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            closed.add(current)
            current_x, current_y = current
            for dx, dy, movement_cost in self._astar_neighbors():
                next_x = current_x + dx
                next_y = current_y + dy
                neighbor = (next_x, next_y)
                if not self._pixel_inside(neighbor):
                    continue
                if mask[next_y, next_x] != 255:
                    continue
                if dx != 0 and dy != 0:
                    if mask[current_y, next_x] != 255 or mask[next_y, current_x] != 255:
                        continue
                clearance_px = float(clearance_map[next_y, next_x])
                clearance_deficit = max(
                    0.0, (desired_clearance_px - clearance_px) / desired_clearance_px
                )
                clearance_penalty = self.astar_clearance_weight * clearance_deficit**2
                tentative_g = current_g + movement_cost * (1.0 + clearance_penalty)
                if tentative_g >= g_score.get(neighbor, math.inf):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                sequence += 1
                heuristic = math.dist(neighbor, goal)
                heapq.heappush(
                    open_heap,
                    (tentative_g + heuristic, tentative_g, sequence, neighbor),
                )
        return []

    def _simplify_path_line_of_sight(
        self, path: Sequence[PixelPoint], mask: np.ndarray
    ) -> List[PixelPoint]:
        if len(path) <= 2:
            return list(path)
        clearance_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        preferred_clearance_px = max(
            1.0, 0.5 * self.astar_clearance_radius_m / self.resolution
        )
        simplified = [path[0]]
        anchor_index = 0
        while anchor_index < len(path) - 1:
            candidate_index = len(path) - 1
            while candidate_index > anchor_index + 1:
                start = path[anchor_index]
                end = path[candidate_index]
                if self._line_is_safe(start, end, mask):
                    line_points = self._bresenham_line(start, end)
                    interior = (
                        line_points[2:-2] if len(line_points) > 5 else line_points
                    )
                    minimum_clearance_px = min(
                        (
                            float(clearance_map[point[1], point[0]])
                            for point in interior
                        ),
                        default=preferred_clearance_px,
                    )
                    if minimum_clearance_px >= preferred_clearance_px:
                        break
                candidate_index -= 1
            simplified.append(path[candidate_index])
            anchor_index = candidate_index
        return simplified

    def _densify_polyline(
        self, points: Sequence[PixelPoint], spacing_px: int
    ) -> List[PixelPoint]:
        if not points:
            return []
        if len(points) == 1:
            return [points[0]]
        spacing_px = max(1, spacing_px)
        dense: List[PixelPoint] = [points[0]]
        for index in range(len(points) - 1):
            start = points[index]
            end = points[index + 1]
            line_points = self._bresenham_line(start, end)
            for line_index in range(spacing_px, len(line_points), spacing_px):
                point = line_points[line_index]
                if point != dense[-1]:
                    dense.append(point)
            if end != dense[-1]:
                dense.append(end)
        return dense

    def _build_safe_connector(
        self,
        previous_start: PixelPoint,
        previous_end: PixelPoint,
        current_start: PixelPoint,
        current_end: PixelPoint,
        coverage_mask: np.ndarray,
        connector_mask: np.ndarray,
    ) -> List[PixelPoint]:
        previous_heading = (
            previous_end[0] - previous_start[0],
            previous_end[1] - previous_start[1],
        )
        current_heading = (
            current_end[0] - current_start[0],
            current_end[1] - current_start[1],
        )
        smooth_turn = self._create_smooth_turn(
            previous_end,
            current_start,
            previous_heading,
            current_heading,
            connector_mask,
        )
        if smooth_turn:
            return smooth_turn
        astar_path = self._find_astar_path(previous_end, current_start, connector_mask)
        if not astar_path:
            return []
        simplified = self._simplify_path_line_of_sight(astar_path, connector_mask)
        sample_spacing_px = max(
            1, int(round(self.connector_sample_spacing_m / self.resolution))
        )
        dense = self._densify_polyline(simplified, sample_spacing_px)
        return dense[1:]

    def _build_scanline_tracks(
        self, row_segments: Sequence[Tuple[int, Sequence[Tuple[int, int]]]]
    ) -> List[List[Tuple[int, int, int]]]:
        tracks: Dict[int, List[Tuple[int, int, int]]] = {}
        active: List[Tuple[int, int, int, int]] = []
        next_track_id = 0
        for y, segments in row_segments:
            current = [(int(x_min), int(x_max), int(y)) for x_min, x_max in segments]
            candidates = []
            for previous_index, (
                track_id,
                previous_min,
                previous_max,
                previous_y,
            ) in enumerate(active):
                previous_center = 0.5 * (previous_min + previous_max)
                for current_index, (x_min, x_max, _) in enumerate(current):
                    overlap = min(previous_max, x_max) - max(previous_min, x_min) + 1
                    if overlap <= 0:
                        continue
                    current_center = 0.5 * (x_min + x_max)
                    candidates.append(
                        (
                            -overlap,
                            abs(previous_center - current_center),
                            previous_index,
                            current_index,
                            track_id,
                        )
                    )
            candidates.sort()
            used_previous = set()
            used_current = set()
            current_active: List[Tuple[int, int, int, int]] = []
            for _, _, previous_index, current_index, track_id in candidates:
                if previous_index in used_previous:
                    continue
                if current_index in used_current:
                    continue
                x_min, x_max, current_y = current[current_index]
                tracks[track_id].append((x_min, x_max, current_y))
                current_active.append((track_id, x_min, x_max, current_y))
                used_previous.add(previous_index)
                used_current.add(current_index)
            for current_index, (x_min, x_max, current_y) in enumerate(current):
                if current_index in used_current:
                    continue
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = [(x_min, x_max, current_y)]
                current_active.append((track_id, x_min, x_max, current_y))
            active = current_active
        return [tracks[track_id] for track_id in sorted(tracks) if tracks[track_id]]

    def _find_start_entry_point(
        self, first: PixelPoint, second: PixelPoint, mask: np.ndarray
    ) -> Optional[PixelPoint]:
        direction = self._normalize_vector(second[0] - first[0], second[1] - first[1])
        if direction is None:
            return None
        desired_px = max(1, int(round(self.start_entry_distance_m / self.resolution)))
        minimum_px = max(
            1, int(round(self.start_entry_min_distance_m / self.resolution))
        )
        unit_x, unit_y = direction
        for distance_px in range(desired_px, minimum_px - 1, -1):
            entry = (
                int(round(first[0] - unit_x * distance_px)),
                int(round(first[1] - unit_y * distance_px)),
            )
            if entry == first:
                continue
            if not self._pixel_inside(entry):
                continue
            if mask[entry[1], entry[0]] != 255:
                continue
            physical_clearance_m = float(self.free_clearance_m[entry[1], entry[0]])
            if physical_clearance_m < self.start_entry_clearance_m:
                continue
            if not self._line_is_safe(entry, first, mask):
                continue
            return entry
        return None

    def _prepend_start_alignment(
        self, path: Sequence[PixelPoint], mask: np.ndarray
    ) -> List[PixelPoint]:
        self.start_entry_pixel = None
        if len(path) < 2:
            return list(path)
        first = path[0]
        second = path[1]
        entry = self._find_start_entry_point(first, second, mask)
        if entry is None:
            self.get_logger().warning("no safe entry")
            return list(path)
        self.start_entry_pixel = entry
        return self._remove_duplicate_points([entry] + list(path))

    def _build_track_path(
        self,
        track: Sequence[Tuple[int, int, int]],
        coverage_mask: np.ndarray,
        connector_mask: np.ndarray,
    ) -> List[PixelPoint]:
        if not track:
            return []
        ordered_segments = sorted(track, key=lambda item: item[2])
        path: List[PixelPoint] = []
        previous_segment = None
        left_to_right = True
        for x_min, x_max, y in ordered_segments:
            start, end = self._make_oriented_sweep_segment(
                x_min, x_max, y, left_to_right
            )
            if not path:
                path.extend([start, end])
            else:
                previous_start, previous_end = previous_segment
                connector = self._build_safe_connector(
                    previous_start,
                    previous_end,
                    start,
                    end,
                    coverage_mask,
                    connector_mask,
                )
                if connector:
                    path.extend(connector)
                elif start != path[-1]:
                    path.append(start)
                if end != path[-1]:
                    path.append(end)
            previous_segment = (start, end)
            left_to_right = not left_to_right
        return self._remove_duplicate_points(path)

    def _generate_boustrophedon_path(
        self,
        coverage_mask: np.ndarray,
        connector_mask: np.ndarray,
        robot_pixel: PixelPoint,
        robot_yaw: float,
    ) -> List[PixelPoint]:
        spacing_px = max(1, int(round(self.line_spacing_m / self.resolution)))
        minimum_segment_px = max(
            2, int(round(self.minimum_segment_length_m / self.resolution))
        )
        rows_with_coverage = np.where(np.any(coverage_mask == 255, axis=1))[0]
        if rows_with_coverage.size == 0:
            return []
        raw_first_y = int(rows_with_coverage[0])
        raw_last_y = int(rows_with_coverage[-1])
        edge_offset_px = max(
            0, int(round(self.scanline_edge_offset_m / self.resolution))
        )
        first_y = raw_first_y + edge_offset_px
        last_y = raw_last_y - edge_offset_px
        if first_y > last_y:
            middle_y = (raw_first_y + raw_last_y) // 2
            first_y = middle_y
            last_y = middle_y
        row_segments = []
        for y in range(first_y, last_y + 1, spacing_px):
            segments = self._extract_row_segments(coverage_mask, y, minimum_segment_px)
            if segments:
                row_segments.append((y, segments))
        if not row_segments:
            return []
        tracks = self._build_scanline_tracks(row_segments)
        track_paths = []
        for track in tracks:
            track_path = self._build_track_path(track, coverage_mask, connector_mask)
            if len(track_path) >= 2:
                track_paths.append(track_path)
        if not track_paths:
            return []
        first_choice = None
        first_score = math.inf
        for index, track_path in enumerate(track_paths):
            for reverse in (False, True):
                candidate = list(reversed(track_path)) if reverse else list(track_path)
                entry = self._find_start_entry_point(
                    candidate[0], candidate[1], connector_mask
                )
                approach_target = entry if entry is not None else candidate[0]
                start_distance = math.dist(robot_pixel, approach_target)
                end_distance = math.dist(robot_pixel, candidate[-1])
                dx_px = candidate[1][0] - candidate[0][0]
                dy_px = candidate[1][1] - candidate[0][1]
                candidate_heading = math.atan2(-dy_px, dx_px)
                heading_error = abs(
                    math.atan2(
                        math.sin(candidate_heading - robot_yaw),
                        math.cos(candidate_heading - robot_yaw),
                    )
                )
                heading_penalty_px = (
                    self.start_heading_weight_m
                    / self.resolution
                    * heading_error
                    / math.pi
                )
                score = start_distance + heading_penalty_px - 0.05 * end_distance
                if entry is None:
                    score += self.start_entry_distance_m / self.resolution
                if score < first_score:
                    first_score = score
                    first_choice = (index, candidate)
        if first_choice is None:
            return []
        first_index, path = first_choice
        unvisited = set(range(len(track_paths)))
        unvisited.remove(first_index)
        connector_failures = 0
        while unvisited:
            previous_start = path[-2]
            previous_end = path[-1]
            best = None
            best_score = math.inf
            for index in sorted(unvisited):
                base_path = track_paths[index]
                for reverse in (False, True):
                    candidate = (
                        list(reversed(base_path)) if reverse else list(base_path)
                    )
                    current_start = candidate[0]
                    current_end = candidate[1]
                    connector = self._build_safe_connector(
                        previous_start,
                        previous_end,
                        current_start,
                        current_end,
                        coverage_mask,
                        connector_mask,
                    )
                    if not connector:
                        continue
                    connector_points = [previous_end] + connector
                    connector_length = sum(
                        (
                            math.dist(
                                connector_points[position - 1],
                                connector_points[position],
                            )
                            for position in range(1, len(connector_points))
                        )
                    )
                    if connector_length < best_score:
                        best_score = connector_length
                        best = (index, candidate, connector)
            if best is None:
                fallback = None
                fallback_score = math.inf
                for index in sorted(unvisited):
                    base_path = track_paths[index]
                    for reverse in (False, True):
                        candidate = (
                            list(reversed(base_path)) if reverse else list(base_path)
                        )
                        score = math.dist(previous_end, candidate[0])
                        if score < fallback_score:
                            fallback_score = score
                            fallback = (index, candidate)
                if fallback is None:
                    break
                index, candidate = fallback
                connector = []
                connector_failures += 1
                self.get_logger().warning("connector fallback")
            else:
                index, candidate, connector = best
            if connector:
                path.extend(connector)
            elif candidate[0] != path[-1]:
                path.append(candidate[0])
            if candidate[0] == path[-1]:
                path.extend(candidate[1:])
            else:
                path.extend(candidate)
            unvisited.remove(index)
        if connector_failures:
            self.get_logger().warning(f"{connector_failures} connector fallback(s)")
        path = self._remove_duplicate_points(path)
        path = self._prune_collinear_points(path, connector_mask)
        return self._prepend_start_alignment(path, connector_mask)

    def _prune_collinear_points(
        self, points: Sequence[PixelPoint], mask: np.ndarray
    ) -> List[PixelPoint]:
        if len(points) <= 2:
            return list(points)
        output = [points[0]]
        for index in range(1, len(points) - 1):
            previous = output[-1]
            current = points[index]
            following = points[index + 1]
            ax = current[0] - previous[0]
            ay = current[1] - previous[1]
            bx = following[0] - current[0]
            by = following[1] - current[1]
            a_len = math.hypot(ax, ay)
            b_len = math.hypot(bx, by)
            if a_len < 1e-06 or b_len < 1e-06:
                continue
            cross = abs(ax * by - ay * bx) / (a_len * b_len)
            dot = (ax * bx + ay * by) / (a_len * b_len)
            nearly_straight = cross < 0.035 and dot > 0.0
            if nearly_straight and self._line_is_safe(previous, following, mask):
                continue
            output.append(current)
        output.append(points[-1])
        return output

    @staticmethod
    def _point_line_distance(
        point: PixelPoint, start: PixelPoint, end: PixelPoint
    ) -> float:
        if start == end:
            return math.dist(point, start)
        px, py = point
        x1, y1 = start
        x2, y2 = end
        numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
        denominator = math.hypot(y2 - y1, x2 - x1)
        return numerator / denominator

    def _safe_rdp(
        self, points: Sequence[PixelPoint], epsilon_px: float, mask: np.ndarray
    ) -> List[PixelPoint]:
        if len(points) <= 2:
            return list(points)
        start = points[0]
        end = points[-1]
        maximum_distance = -1.0
        split_index = 0
        for index in range(1, len(points) - 1):
            distance = self._point_line_distance(points[index], start, end)
            if distance > maximum_distance:
                maximum_distance = distance
                split_index = index
        if maximum_distance <= epsilon_px and self._line_is_safe(start, end, mask):
            return [start, end]
        if split_index <= 0 or split_index >= len(points) - 1:
            split_index = len(points) // 2
        left = self._safe_rdp(points[: split_index + 1], epsilon_px, mask)
        right = self._safe_rdp(points[split_index:], epsilon_px, mask)
        return left[:-1] + right

    def _mandatory_waypoint_indices(self, points: Sequence[PixelPoint]) -> List[int]:
        if not points:
            return []
        mandatory = {0, len(points) - 1}
        minimum_length_px = max(
            2.0, self.mandatory_sweep_min_length_m / self.resolution
        )
        for index, (first, second) in enumerate(zip(points[:-1], points[1:])):
            dx = second[0] - first[0]
            dy = second[1] - first[1]
            length_px = math.hypot(dx, dy)
            is_sweep = length_px >= minimum_length_px and abs(dx) >= max(
                2.0, 4.0 * abs(dy)
            )
            if is_sweep:
                mandatory.add(index)
                mandatory.add(index + 1)
        return sorted(mandatory)

    def _simplify_with_mandatory_points(
        self,
        points: Sequence[PixelPoint],
        mandatory_indices: Sequence[int],
        epsilon_px: float,
        mask: np.ndarray,
    ) -> List[PixelPoint]:
        if len(points) <= 2:
            return list(points)
        output: List[PixelPoint] = []
        for start_index, end_index in zip(
            mandatory_indices[:-1], mandatory_indices[1:]
        ):
            span = list(points[start_index : end_index + 1])
            if len(span) <= 2:
                reduced = span
            else:
                reduced = self._safe_rdp(span, epsilon_px, mask)
                reduced = self._prune_collinear_points(reduced, mask)
            for point in reduced:
                if not output or point != output[-1]:
                    output.append(point)
        return output

    def _simplify_for_nav2(
        self, points: Sequence[PixelPoint], mask: np.ndarray
    ) -> List[PixelPoint]:
        if len(points) <= 2:
            self.mandatory_path_pixels = list(points)
            return list(points)
        mandatory_indices = self._mandatory_waypoint_indices(points)
        self.mandatory_path_pixels = [points[index] for index in mandatory_indices]
        epsilon_m = self.nav_simplify_tolerance_m
        maximum_epsilon_m = max(epsilon_m, min(0.18, self.line_spacing_m * 0.45))
        simplified = list(points)
        while True:
            epsilon_px = max(1.0, epsilon_m / self.resolution)
            simplified = self._simplify_with_mandatory_points(
                points, mandatory_indices, epsilon_px, mask
            )
            if (
                len(simplified) <= self.max_nav_waypoints
                or epsilon_m >= maximum_epsilon_m - 1e-09
            ):
                break
            epsilon_m = min(maximum_epsilon_m, epsilon_m + 0.02)
        if len(simplified) > self.max_nav_waypoints:
            self.get_logger().warning(f"waypoint limit: {len(simplified)} poses")
        return simplified

    @staticmethod
    def _remove_duplicate_points(points: Sequence[PixelPoint]) -> List[PixelPoint]:
        if not points:
            return []
        filtered = [points[0]]
        for point in points[1:]:
            if point != filtered[-1]:
                filtered.append(point)
        return filtered
