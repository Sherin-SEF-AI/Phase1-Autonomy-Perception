#!/usr/bin/env python3
"""
Safety Alert System for ADAS
Implements FCW, LDW, and DMS with audio warnings
"""

import cv2
import numpy as np
import time
from collections import deque
import subprocess
import threading
import platform

class SafetyAlertSystem:
    """Comprehensive safety alert system with audio warnings"""

    def __init__(self):
        # Forward Collision Warning (FCW) settings
        self.fcw_enabled = True
        self.fcw_ttc_threshold = 2.0  # Time-to-collision threshold in seconds
        self.fcw_last_alert_time = 0
        self.fcw_alert_cooldown = 2.0  # Minimum seconds between alerts
        self.object_history = {}  # Track object sizes over time

        # Lane Departure Warning (LDW) settings
        self.ldw_enabled = True
        self.ldw_center_threshold = 50  # Pixels from image center
        self.ldw_last_alert_time = 0
        self.ldw_alert_cooldown = 3.0
        self.turn_signal_active = False  # Set to True when turn signal is on
        self.lane_center_history = deque(maxlen=10)

        # Driver Monitoring System (DMS) settings
        self.dms_enabled = True
        self.dms_distraction_threshold = 3.0  # Seconds looking away
        self.dms_phone_looking_start = None
        self.dms_last_alert_time = 0
        self.dms_alert_cooldown = 5.0

        # Audio system
        self.audio_enabled = True
        self.platform = platform.system()

        print("Safety Alert System initialized")
        print(f"  FCW: Enabled (TTC < {self.fcw_ttc_threshold}s)")
        print(f"  LDW: Enabled (±{self.ldw_center_threshold}px)")
        print(f"  DMS: Enabled (>{self.dms_distraction_threshold}s)")

    # ========================================================================
    # FORWARD COLLISION WARNING (FCW)
    # ========================================================================

    def check_forward_collision(self, detections, frame_width, frame_height):
        """
        Check for imminent collisions using bounding box growth rate

        Args:
            detections: List of detection objects with bbox, class_name, track_id
            frame_width: Width of the frame
            frame_height: Height of the frame

        Returns:
            dict: Alert information or None
        """
        if not self.fcw_enabled:
            return None

        current_time = time.time()

        # Cooldown check
        if current_time - self.fcw_last_alert_time < self.fcw_alert_cooldown:
            return None

        for detection in detections:
            # Only check for vehicles (cars, trucks, buses)
            if detection.get('class_name') not in ['car', 'truck', 'bus', 'vehicle']:
                continue

            bbox = detection.get('bbox')
            track_id = detection.get('track_id', -1)

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox

            # Calculate bounding box area
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = frame_width * frame_height
            bbox_percentage = (bbox_area / frame_area) * 100

            # Calculate center position
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Only check vehicles in center 60% of frame (directly ahead)
            if not (frame_width * 0.2 < center_x < frame_width * 0.8):
                continue

            # Track object history
            if track_id not in self.object_history:
                self.object_history[track_id] = deque(maxlen=30)  # 1 second at 30fps

            self.object_history[track_id].append({
                'time': current_time,
                'area': bbox_area,
                'bbox_percentage': bbox_percentage
            })

            # Need at least 10 frames to calculate TTC
            if len(self.object_history[track_id]) < 10:
                continue

            # Calculate Time-to-Collision (TTC)
            ttc = self._calculate_ttc(self.object_history[track_id])

            if ttc is not None and ttc < self.fcw_ttc_threshold:
                # COLLISION IMMINENT!
                self.fcw_last_alert_time = current_time

                alert_info = {
                    'type': 'FCW',
                    'severity': 'CRITICAL' if ttc < 1.0 else 'WARNING',
                    'ttc': ttc,
                    'vehicle_type': detection.get('class_name', 'vehicle'),
                    'bbox_percentage': bbox_percentage,
                    'message': 'BRAKE! BRAKE!' if ttc < 1.0 else 'Collision Warning!',
                    'audio': 'brake_brake' if ttc < 1.0 else 'collision_warning'
                }

                # Play audio alert
                self._play_alert(alert_info['audio'], alert_info['message'])

                return alert_info

        return None

    def _calculate_ttc(self, history):
        """
        Calculate Time-to-Collision based on bbox area growth

        TTC = current_size / rate_of_size_change
        """
        if len(history) < 10:
            return None

        # Get recent data points
        recent = list(history)[-10:]
        oldest = recent[0]
        newest = recent[-1]

        # Calculate growth rate
        time_delta = newest['time'] - oldest['time']

        if time_delta < 0.1:  # Avoid division by zero
            return None

        area_growth = newest['area'] - oldest['area']
        growth_rate = area_growth / time_delta  # pixels²/second

        # If object is shrinking or not growing, no collision
        if growth_rate <= 100:  # Minimal growth threshold
            return None

        # TTC = current_size / growth_rate
        current_area = newest['area']

        # Estimate when object will fill 60% of frame (collision imminent)
        critical_area = (640 * 480) * 0.6  # Assuming VGA resolution
        area_to_critical = critical_area - current_area

        if area_to_critical <= 0:
            return 0.1  # Already critical

        ttc = area_to_critical / growth_rate

        # Sanity check
        if ttc < 0 or ttc > 10:
            return None

        return ttc

    # ========================================================================
    # LANE DEPARTURE WARNING (LDW)
    # ========================================================================

    def check_lane_departure(self, lane_center_x, frame_center_x, frame_width):
        """
        Check if vehicle is drifting out of lane

        Args:
            lane_center_x: X-coordinate of lane center (from green line)
            frame_center_x: X-coordinate of frame center
            frame_width: Width of frame

        Returns:
            dict: Alert information or None
        """
        if not self.ldw_enabled or lane_center_x is None:
            return None

        current_time = time.time()

        # Cooldown check
        if current_time - self.ldw_last_alert_time < self.ldw_alert_cooldown:
            return None

        # Don't alert if turn signal is active
        if self.turn_signal_active:
            return None

        # Calculate offset from center
        offset = lane_center_x - frame_center_x

        # Add to history for smoothing
        self.lane_center_history.append(offset)

        # Need enough history
        if len(self.lane_center_history) < 5:
            return None

        # Calculate average offset (smoothing)
        avg_offset = sum(self.lane_center_history) / len(self.lane_center_history)

        # Check if significantly off center
        if abs(avg_offset) > self.ldw_center_threshold:
            self.ldw_last_alert_time = current_time

            direction = 'Left' if avg_offset < 0 else 'Right'

            alert_info = {
                'type': 'LDW',
                'severity': 'WARNING',
                'direction': direction,
                'offset': avg_offset,
                'message': f'Drifting {direction}',
                'audio': f'drifting_{direction.lower()}'
            }

            # Play audio alert
            self._play_alert(alert_info['audio'], alert_info['message'])

            return alert_info

        return None

    # ========================================================================
    # DRIVER MONITORING SYSTEM (DMS)
    # ========================================================================

    def check_driver_attention(self, driver_state):
        """
        Monitor driver attention and detect phone usage/distraction

        Args:
            driver_state: Dict with keys:
                - 'looking_down': bool
                - 'eyes_closed': bool
                - 'head_pose': str ('forward', 'down', 'left', 'right')
                - 'drowsiness_score': float (0-1)

        Returns:
            dict: Alert information or None
        """
        if not self.dms_enabled or driver_state is None:
            return None

        current_time = time.time()

        # Cooldown check
        if current_time - self.dms_last_alert_time < self.dms_alert_cooldown:
            # Still track duration even during cooldown
            if driver_state.get('looking_down') or driver_state.get('head_pose') == 'down':
                if self.dms_phone_looking_start is None:
                    self.dms_phone_looking_start = current_time
            else:
                self.dms_phone_looking_start = None
            return None

        # Check if driver is looking down (phone usage)
        if driver_state.get('looking_down') or driver_state.get('head_pose') == 'down':
            if self.dms_phone_looking_start is None:
                self.dms_phone_looking_start = current_time
            else:
                duration = current_time - self.dms_phone_looking_start

                if duration > self.dms_distraction_threshold:
                    self.dms_last_alert_time = current_time
                    self.dms_phone_looking_start = None  # Reset

                    alert_info = {
                        'type': 'DMS',
                        'severity': 'CRITICAL',
                        'distraction_type': 'phone',
                        'duration': duration,
                        'message': 'Eyes on the Road!',
                        'audio': 'eyes_on_road'
                    }

                    # Play audio alert
                    self._play_alert(alert_info['audio'], alert_info['message'])

                    return alert_info
        else:
            # Driver looking forward, reset timer
            self.dms_phone_looking_start = None

        # Check for drowsiness
        if driver_state.get('eyes_closed') or driver_state.get('drowsiness_score', 0) > 0.7:
            self.dms_last_alert_time = current_time

            alert_info = {
                'type': 'DMS',
                'severity': 'WARNING',
                'distraction_type': 'drowsiness',
                'message': 'Driver Drowsiness Detected',
                'audio': 'drowsiness_alert'
            }

            # Play audio alert
            self._play_alert(alert_info['audio'], alert_info['message'])

            return alert_info

        return None

    # ========================================================================
    # AUDIO ALERT SYSTEM
    # ========================================================================

    def _play_alert(self, alert_type, message):
        """Play audio alert using system TTS or beep"""
        if not self.audio_enabled:
            print(f"[ALERT] {message}")
            return

        # Use threading to avoid blocking
        thread = threading.Thread(target=self._play_alert_async, args=(alert_type, message))
        thread.daemon = True
        thread.start()

    def _play_alert_async(self, alert_type, message):
        """Asynchronously play audio alert"""
        try:
            if self.platform == 'Linux':
                # Use espeak or festival for TTS on Linux
                try:
                    # Try espeak first (faster)
                    subprocess.run(['espeak', '-a', '200', '-s', '150', message],
                                 check=True,
                                 stderr=subprocess.DEVNULL,
                                 timeout=3)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    try:
                        # Fallback to festival
                        subprocess.run(['echo', message, '|', 'festival', '--tts'],
                                     shell=True,
                                     stderr=subprocess.DEVNULL,
                                     timeout=3)
                    except:
                        # Fallback to beep
                        print(f"\a{message}")  # System beep

            elif self.platform == 'Darwin':  # macOS
                subprocess.run(['say', '-r', '200', message],
                             stderr=subprocess.DEVNULL,
                             timeout=3)

            elif self.platform == 'Windows':
                # Use Windows SAPI
                import winsound
                winsound.Beep(1000, 500)  # 1000Hz for 500ms
                print(f"[ALERT] {message}")

            else:
                print(f"[ALERT] {message}")

        except Exception as e:
            print(f"[ALERT] {message} (Audio error: {e})")

    # ========================================================================
    # CONTROL METHODS
    # ========================================================================

    def set_turn_signal(self, active, direction=None):
        """Set turn signal state (disable LDW when active)"""
        self.turn_signal_active = active
        if active:
            print(f"Turn signal: {direction}")

    def enable_fcw(self, enabled=True):
        """Enable/disable Forward Collision Warning"""
        self.fcw_enabled = enabled
        print(f"FCW: {'Enabled' if enabled else 'Disabled'}")

    def enable_ldw(self, enabled=True):
        """Enable/disable Lane Departure Warning"""
        self.ldw_enabled = enabled
        print(f"LDW: {'Enabled' if enabled else 'Disabled'}")

    def enable_dms(self, enabled=True):
        """Enable/disable Driver Monitoring System"""
        self.dms_enabled = enabled
        print(f"DMS: {'Enabled' if enabled else 'Disabled'}")

    def enable_audio(self, enabled=True):
        """Enable/disable audio alerts"""
        self.audio_enabled = enabled
        print(f"Audio Alerts: {'Enabled' if enabled else 'Disabled'}")

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def draw_alert(self, frame, alert_info):
        """
        Draw alert overlay on frame

        Args:
            frame: Video frame
            alert_info: Alert information dict
        """
        if alert_info is None:
            return frame

        h, w = frame.shape[:2]
        alert_type = alert_info['type']
        severity = alert_info['severity']
        message = alert_info['message']

        # Color based on severity
        if severity == 'CRITICAL':
            color = (0, 0, 255)  # Red
            thickness = 3
        else:
            color = (0, 165, 255)  # Orange
            thickness = 2

        # Draw alert banner
        cv2.rectangle(frame, (0, 0), (w, 80), color, -1)
        cv2.rectangle(frame, (0, 0), (w, 80), (255, 255, 255), 2)

        # Alert type
        cv2.putText(frame, alert_type, (10, 30),
                   cv2.FONT_HERSHEY_BOLD, 1.0, (255, 255, 255), 2)

        # Message
        cv2.putText(frame, message, (10, 65),
                   cv2.FONT_HERSHEY_BOLD, 1.2, (255, 255, 255), 2)

        # Additional info
        info_y = 100

        if alert_type == 'FCW' and 'ttc' in alert_info:
            ttc_text = f"TTC: {alert_info['ttc']:.1f}s"
            cv2.putText(frame, ttc_text, (w - 200, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        elif alert_type == 'LDW' and 'direction' in alert_info:
            offset_text = f"Offset: {alert_info['offset']:.0f}px {alert_info['direction']}"
            cv2.putText(frame, offset_text, (w - 250, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        elif alert_type == 'DMS' and 'duration' in alert_info:
            dur_text = f"Distracted: {alert_info['duration']:.1f}s"
            cv2.putText(frame, dur_text, (w - 250, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame

    def get_status_text(self):
        """Get status text for display"""
        status = []
        status.append(f"FCW: {'ON' if self.fcw_enabled else 'OFF'}")
        status.append(f"LDW: {'ON' if self.ldw_enabled else 'OFF'}")
        status.append(f"DMS: {'ON' if self.dms_enabled else 'OFF'}")
        return " | ".join(status)


# ============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# ============================================================================

def extract_lane_center(lane_lines):
    """
    Extract lane center X coordinate from lane detection results

    Args:
        lane_lines: List of lane line points or coordinates

    Returns:
        float: X coordinate of lane center, or None
    """
    if not lane_lines or len(lane_lines) < 2:
        return None

    # Assuming lane_lines is a list of [x1, y1, x2, y2] or similar
    # Calculate center between leftmost and rightmost lanes
    try:
        if isinstance(lane_lines[0], (list, tuple)) and len(lane_lines[0]) >= 4:
            # Format: [[x1, y1, x2, y2], ...]
            x_coords = []
            for line in lane_lines:
                x_coords.extend([line[0], line[2]])

            if x_coords:
                left_edge = min(x_coords)
                right_edge = max(x_coords)
                center = (left_edge + right_edge) / 2
                return center

        return None

    except Exception as e:
        print(f"Error extracting lane center: {e}")
        return None


def create_driver_state_from_dms(dms_result):
    """
    Convert DMS detection result to driver state dict

    Args:
        dms_result: Result from DriverAttentionMonitor

    Returns:
        dict: Driver state information
    """
    if dms_result is None:
        return None

    return {
        'looking_down': dms_result.get('head_down', False),
        'eyes_closed': dms_result.get('eyes_closed', False),
        'head_pose': dms_result.get('head_pose', 'unknown'),
        'drowsiness_score': dms_result.get('drowsiness', 0.0)
    }


if __name__ == "__main__":
    # Test the safety alert system
    print("Safety Alert System Test")
    print("=" * 70)

    safety = SafetyAlertSystem()

    # Test FCW
    print("\nTesting Forward Collision Warning...")
    test_detections = [
        {
            'class_name': 'truck',
            'track_id': 1,
            'bbox': [200, 150, 450, 350]  # Large truck ahead
        }
    ]

    # Simulate growing bbox (truck approaching)
    for i in range(15):
        scale = 1 + (i * 0.1)
        test_detections[0]['bbox'] = [
            int(200 / scale),
            int(150 / scale),
            int(450 * scale),
            int(350 * scale)
        ]

        alert = safety.check_forward_collision(test_detections, 640, 480)
        if alert:
            print(f"  FCW Alert: {alert}")
            break
        time.sleep(0.1)

    # Test LDW
    print("\nTesting Lane Departure Warning...")
    for offset in range(0, 100, 10):
        lane_center = 320 + offset  # Drifting right
        alert = safety.check_lane_departure(lane_center, 320, 640)
        if alert:
            print(f"  LDW Alert: {alert}")
            break
        time.sleep(0.1)

    # Test DMS
    print("\nTesting Driver Monitoring System...")
    for i in range(35):  # 3.5 seconds at 10fps
        driver_state = {
            'looking_down': True,
            'eyes_closed': False,
            'head_pose': 'down',
            'drowsiness_score': 0.0
        }
        alert = safety.check_driver_attention(driver_state)
        if alert:
            print(f"  DMS Alert: {alert}")
            break
        time.sleep(0.1)

    print("\n" + "=" * 70)
    print("Test complete!")
