#!/usr/bin/env python3
"""
Ultra-Advanced Visualization Module
Real-time graphs, heatmaps, trajectory maps, and statistics
"""

import wx
import cv2
import numpy as np
from typing import List, Tuple, Dict, Deque
from collections import deque
from datetime import datetime
import logging

try:
    import matplotlib
    matplotlib.use('WXAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger('UltraVisualization')


# ============================================================================
# REAL-TIME PERFORMANCE GRAPHS
# ============================================================================

class PerformanceGraphPanel(wx.Panel):
    """Real-time performance graphs (FPS, CPU, Memory)"""

    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(wx.Colour(30, 30, 30))

        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available - graphs disabled")
            return

        # Create figure with 3 subplots
        self.figure = Figure(figsize=(8, 6), facecolor='#1e1e1e')
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)

        # Create subplots
        self.ax1 = self.figure.add_subplot(311)
        self.ax2 = self.figure.add_subplot(312)
        self.ax3 = self.figure.add_subplot(313)

        # Style subplots
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor('#2a2a2a')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')

        # Data buffers
        self.fps_data = deque(maxlen=100)
        self.cpu_data = deque(maxlen=100)
        self.memory_data = deque(maxlen=100)
        self.time_data = deque(maxlen=100)

        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)

        self.start_time = datetime.now()

    def update(self, fps: float, cpu: float, memory: float):
        """Update graphs with new data"""
        if not MATPLOTLIB_AVAILABLE:
            return

        # Add data
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.time_data.append(elapsed)
        self.fps_data.append(fps)
        self.cpu_data.append(cpu)
        self.memory_data.append(memory)

        # Clear and redraw
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        time_array = list(self.time_data)

        # FPS graph
        self.ax1.plot(time_array, self.fps_data, 'g-', linewidth=2, label='FPS')
        self.ax1.fill_between(time_array, self.fps_data, alpha=0.3, color='green')
        self.ax1.set_ylabel('FPS', color='white')
        self.ax1.set_title('Performance Metrics', color='white', fontsize=12)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='upper right')
        self.ax1.set_ylim([0, max(list(self.fps_data) + [30])])

        # CPU graph
        self.ax2.plot(time_array, self.cpu_data, 'b-', linewidth=2, label='CPU %')
        self.ax2.fill_between(time_array, self.cpu_data, alpha=0.3, color='blue')
        self.ax2.set_ylabel('CPU %', color='white')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper right')
        self.ax2.set_ylim([0, 100])

        # Memory graph
        self.ax3.plot(time_array, self.memory_data, 'r-', linewidth=2, label='Memory MB')
        self.ax3.fill_between(time_array, self.memory_data, alpha=0.3, color='red')
        self.ax3.set_ylabel('Memory (MB)', color='white')
        self.ax3.set_xlabel('Time (seconds)', color='white')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend(loc='upper right')

        # Redraw
        self.figure.tight_layout()
        self.canvas.draw()


# ============================================================================
# DETECTION CONFIDENCE GRAPH
# ============================================================================

class ConfidenceGraphPanel(wx.Panel):
    """Graph showing detection confidence over time"""

    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(wx.Colour(30, 30, 30))

        if not MATPLOTLIB_AVAILABLE:
            return

        self.figure = Figure(figsize=(6, 4), facecolor='#1e1e1e')
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.ax = self.figure.add_subplot(111)

        # Style
        self.ax.set_facecolor('#2a2a2a')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_color('white')

        # Data
        self.confidence_history = deque(maxlen=50)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def update(self, detections: List):
        """Update with new detections"""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'ax'):
            return

        # Calculate average confidence
        if detections:
            avg_conf = np.mean([d.confidence for d in detections if hasattr(d, 'confidence')])
            self.confidence_history.append(avg_conf)
        else:
            self.confidence_history.append(0)

        # Redraw
        self.ax.clear()
        self.ax.plot(list(self.confidence_history), 'y-', linewidth=2)
        self.ax.fill_between(range(len(self.confidence_history)),
                            self.confidence_history, alpha=0.3, color='yellow')
        self.ax.set_ylabel('Avg Confidence', color='white')
        self.ax.set_xlabel('Frames', color='white')
        self.ax.set_title('Detection Confidence', color='white')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim([0, 1])

        self.figure.tight_layout()
        self.canvas.draw()


# ============================================================================
# HEATMAP TIMELINE
# ============================================================================

class HeatmapTimeline:
    """Create heatmap showing danger zones over time"""

    def __init__(self, width: int = 1280, height: int = 720, history_length: int = 300):
        self.width = width
        self.height = height
        self.history_length = history_length
        self.danger_map = np.zeros((height, width), dtype=np.float32)
        self.timeline = deque(maxlen=history_length)
        self.decay_rate = 0.95

    def update(self, collision_zones: List[Tuple], frame_shape: Tuple):
        """Update heatmap with collision zones"""
        h, w = frame_shape[:2]

        if (h, w) != (self.height, self.width):
            self.danger_map = cv2.resize(self.danger_map, (w, h))
            self.height, self.width = h, w

        # Decay existing values
        self.danger_map *= self.decay_rate

        # Add new danger zones
        for zone in collision_zones:
            if len(zone) >= 2:
                cx, cy = zone[:2]
                radius = zone[2] if len(zone) > 2 else 50

                # Add Gaussian blob
                y, x = np.ogrid[:self.height, :self.width]
                mask = ((x - cx)**2 + (y - cy)**2) <= radius**2
                self.danger_map[mask] += 0.3

        # Normalize
        self.danger_map = np.clip(self.danger_map, 0, 1)

        # Store in timeline
        self.timeline.append(self.danger_map.copy())

    def visualize(self) -> np.ndarray:
        """Generate heatmap visualization"""
        # Convert to color
        heatmap_uint8 = (self.danger_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        return colored

    def get_timeline_visualization(self) -> np.ndarray:
        """Get timeline view (time on X axis, position on Y axis)"""
        if len(self.timeline) == 0:
            return np.zeros((100, 300, 3), dtype=np.uint8)

        # Create timeline image
        timeline_height = 100
        timeline_width = len(self.timeline)

        # Average each column
        timeline_img = np.zeros((timeline_height, timeline_width), dtype=np.float32)

        for i, heatmap in enumerate(self.timeline):
            # Average columns
            column_avg = np.mean(heatmap, axis=0)
            # Resize to timeline height
            resized = cv2.resize(column_avg.reshape(1, -1), (1, timeline_height))
            timeline_img[:, i] = resized.flatten()[:timeline_height]

        # Colorize
        timeline_uint8 = (timeline_img * 255).astype(np.uint8)
        colored = cv2.applyColorMap(timeline_uint8, cv2.COLORMAP_HOT)

        return colored


# ============================================================================
# OBJECT TRAJECTORY MAP
# ============================================================================

class TrajectoryMapPanel(wx.Panel):
    """Top-down view showing object trajectories"""

    def __init__(self, parent, map_size: Tuple = (400, 600)):
        super().__init__(parent)
        self.map_width, self.map_height = map_size
        self.SetMinSize(wx.Size(*map_size))
        self.SetBackgroundColour(wx.Colour(30, 30, 30))

        self.trajectory_map = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        self.bitmap = None

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)

    def update(self, tracked_objects: List):
        """Update trajectory map with tracked objects"""
        # Decay existing map
        self.trajectory_map = (self.trajectory_map * 0.95).astype(np.uint8)

        # Draw grid
        self.trajectory_map[::50, :] = (30, 30, 30)
        self.trajectory_map[:, ::50] = (30, 30, 30)

        # Draw ego vehicle at bottom center
        ego_x = self.map_width // 2
        ego_y = self.map_height - 30
        cv2.rectangle(self.trajectory_map,
                     (ego_x - 15, ego_y - 25),
                     (ego_x + 15, ego_y),
                     (0, 255, 0), -1)
        cv2.putText(self.trajectory_map, "EGO",
                   (ego_x - 15, ego_y - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw tracked objects and trajectories
        for obj in tracked_objects:
            if not hasattr(obj, 'distance') or not hasattr(obj, 'history'):
                continue

            # Map distance to Y coordinate (closer = bottom)
            obj_y = int(self.map_height - 30 - (obj.distance * 5))
            obj_y = max(10, min(self.map_height - 10, obj_y))

            # Use centroid X for lateral position (normalized)
            if hasattr(obj, 'centroid'):
                lateral_offset = obj.centroid[0] - 640  # Assuming 1280 width
                obj_x = int(self.map_width // 2 + lateral_offset // 5)
                obj_x = max(10, min(self.map_width - 10, obj_x))

                # Draw trajectory
                if len(obj.history) > 1:
                    for i in range(len(obj.history) - 1):
                        cv2.line(self.trajectory_map,
                                (obj_x, obj_y),
                                (obj_x + (obj.history[i][0] - obj.centroid[0]) // 10,
                                 obj_y - 5 * i),
                                (0, 200, 200), 1)

                # Draw object
                color = (0, 0, 255) if obj.ttc < 2.0 else (0, 255, 255)
                cv2.circle(self.trajectory_map, (obj_x, obj_y), 5, color, -1)

                # Draw predicted trajectory
                if hasattr(obj, 'predicted_trajectory') and obj.predicted_trajectory:
                    for pred_pos in obj.predicted_trajectory:
                        pred_x = int(self.map_width // 2 + (pred_pos[0] - 640) // 5)
                        pred_y = int(obj_y - 10)
                        if 0 <= pred_x < self.map_width and 0 <= pred_y < self.map_height:
                            cv2.circle(self.trajectory_map, (pred_x, pred_y), 2, (255, 165, 0), -1)

                # Label
                label = f"ID:{obj.track_id}"
                cv2.putText(self.trajectory_map, label,
                           (obj_x - 15, obj_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Convert to wx.Bitmap
        rgb = cv2.cvtColor(self.trajectory_map, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.bitmap = wx.Bitmap.FromBuffer(w, h, rgb)

        self.Refresh()

    def on_paint(self, event):
        """Paint event handler"""
        dc = wx.BufferedPaintDC(self)
        dc.SetBackground(wx.Brush(wx.Colour(30, 30, 30)))
        dc.Clear()

        if self.bitmap:
            dc.DrawBitmap(self.bitmap, 0, 0)

    def on_size(self, event):
        """Handle resize"""
        self.Refresh()
        event.Skip()


# ============================================================================
# STATISTICS DASHBOARD
# ============================================================================

class StatisticsDashboard(wx.Panel):
    """Comprehensive statistics dashboard"""

    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(wx.Colour(25, 25, 35))

        self._create_ui()

        # Statistics
        self.stats = {
            'session_start': datetime.now(),
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'max_fps': 0,
            'avg_fps': 0,
            'lane_departures': 0,
            'collision_warnings': 0,
            'scene_type': "Unknown",
            'weather': "Unknown",
            'driving_score': 100
        }

    def _create_ui(self):
        """Create dashboard UI"""
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Title
        title = wx.StaticText(self, label="📊 SESSION STATISTICS")
        title.SetForegroundColour(wx.Colour(0, 200, 255))
        title.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        main_sizer.Add(title, 0, wx.ALL | wx.ALIGN_CENTER, 10)

        # Stats grid
        grid = wx.FlexGridSizer(rows=15, cols=2, vgap=8, hgap=15)

        self.labels = {}
        stats_config = [
            ("Session Duration:", "duration", "00:00:00"),
            ("Total Frames:", "frames", "0"),
            ("Total Detections:", "detections", "0"),
            ("Unique Tracks:", "tracks", "0"),
            ("Current FPS:", "fps", "0.0"),
            ("Max FPS:", "max_fps", "0.0"),
            ("Avg FPS:", "avg_fps", "0.0"),
            ("Lane Departures:", "departures", "0"),
            ("Collision Warnings:", "warnings", "0"),
            ("Scene Type:", "scene", "Unknown"),
            ("Weather:", "weather", "Unknown"),
            ("Traffic:", "traffic", "Unknown"),
            ("Road Condition:", "road", "Unknown"),
            ("Driving Score:", "score", "100"),
            ("Behavior:", "behavior", "Normal")
        ]

        for label_text, key, default in stats_config:
            label = wx.StaticText(self, label=label_text)
            label.SetForegroundColour(wx.Colour(150, 150, 150))
            label.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

            value = wx.StaticText(self, label=default)
            value.SetForegroundColour(wx.Colour(255, 255, 255))
            value.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))

            self.labels[key] = value

            grid.Add(label, 0, wx.ALIGN_LEFT)
            grid.Add(value, 0, wx.ALIGN_RIGHT)

        main_sizer.Add(grid, 0, wx.ALL | wx.EXPAND, 10)

        self.SetSizer(main_sizer)

    def update(self, metrics: Dict):
        """Update statistics"""
        self.stats['total_frames'] += 1

        if 'num_detections' in metrics:
            self.stats['total_detections'] += metrics['num_detections']

        if 'num_tracked_objects' in metrics:
            self.stats['total_tracks'] = max(self.stats['total_tracks'], metrics['num_tracked_objects'])

        if 'fps' in metrics:
            self.stats['max_fps'] = max(self.stats['max_fps'], metrics['fps'])
            # Calculate running average
            n = self.stats['total_frames']
            self.stats['avg_fps'] = (self.stats['avg_fps'] * (n-1) + metrics['fps']) / n

        # Update UI
        duration = datetime.now() - self.stats['session_start']
        self.labels['duration'].SetLabel(str(duration).split('.')[0])
        self.labels['frames'].SetLabel(str(self.stats['total_frames']))
        self.labels['detections'].SetLabel(str(self.stats['total_detections']))
        self.labels['tracks'].SetLabel(str(self.stats['total_tracks']))

        if 'fps' in metrics:
            self.labels['fps'].SetLabel(f"{metrics['fps']:.1f}")
        self.labels['max_fps'].SetLabel(f"{self.stats['max_fps']:.1f}")
        self.labels['avg_fps'].SetLabel(f"{self.stats['avg_fps']:.1f}")

        self.labels['departures'].SetLabel(str(self.stats['lane_departures']))
        self.labels['warnings'].SetLabel(str(self.stats['collision_warnings']))

        # Update scene info if available
        if 'scene_context' in metrics:
            scene = metrics['scene_context']
            if hasattr(scene, 'road_type'):
                self.labels['scene'].SetLabel(scene.road_type.name)
            if hasattr(scene, 'weather'):
                self.labels['weather'].SetLabel(scene.weather.title())
            if hasattr(scene, 'traffic_density'):
                self.labels['traffic'].SetLabel(scene.traffic_density.name)
            if hasattr(scene, 'road_condition'):
                self.labels['road'].SetLabel(scene.road_condition.name)

        # Update driving score
        if 'driving_score' in metrics:
            score = metrics['driving_score']
            self.stats['driving_score'] = score
            self.labels['score'].SetLabel(str(score))

            # Color code
            if score < 60:
                self.labels['score'].SetForegroundColour(wx.Colour(255, 0, 0))
                self.labels['behavior'].SetLabel("Aggressive")
            elif score > 85:
                self.labels['score'].SetForegroundColour(wx.Colour(0, 255, 0))
                self.labels['behavior'].SetLabel("Cautious")
            else:
                self.labels['score'].SetForegroundColour(wx.Colour(255, 255, 0))
                self.labels['behavior'].SetLabel("Normal")


# Only for testing
if __name__ == "__main__":
    print("Ultra Visualization Module Loaded")
    print("Available Components:")
    print("  - PerformanceGraphPanel")
    print("  - ConfidenceGraphPanel")
    print("  - HeatmapTimeline")
    print("  - TrajectoryMapPanel")
    print("  - StatisticsDashboard")
