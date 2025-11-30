#!/usr/bin/env python3
"""
Basic Usage Example - ADAS Perception System
Demonstrates basic object detection and tracking
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from adas_perception import ObjectDetector, AdvancedObjectTracker, DistanceEstimator


def main():
    print("=" * 60)
    print("ADAS Perception - Basic Usage Example")
    print("=" * 60)

    # Initialize components
    detector = ObjectDetector(confidence_threshold=0.5)
    tracker = AdvancedObjectTracker()
    distance_estimator = DistanceEstimator()

    # Open video or webcam
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide video file path

    if not cap.isOpened():
        print("ERROR: Could not open video source")
        return

    print("\nProcessing video... Press 'q' to quit")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect objects
        detections = detector.detect(frame)

        # Estimate distances
        detections = distance_estimator.estimate(detections, frame.shape[0])

        # Track objects
        tracked_objects = tracker.update(detections)

        # Draw results
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            color = (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"ID:{obj.track_id} {obj.class_name} {obj.distance:.1f}m"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw trajectory
            if len(obj.history) > 1:
                points = np.array(list(obj.history), dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)

        # Display info
        info_text = f"Frame: {frame_count} | Objects: {len(tracked_objects)}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show frame
        cv2.imshow('ADAS Perception - Basic Example', frame)

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames")
    print("Done!")


if __name__ == "__main__":
    main()
