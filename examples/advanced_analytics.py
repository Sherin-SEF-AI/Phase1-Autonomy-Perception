#!/usr/bin/env python3
"""
Advanced Analytics Example - ADAS Perception System
Demonstrates performance profiling, behavior analysis, and data logging
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import time
from advanced_modules import (
    PerformanceProfiler,
    DrivingAnalyticsEngine,
    DataLogger,
    HeatMapGenerator
)


def main():
    print("=" * 60)
    print("ADAS Perception - Advanced Analytics Example")
    print("=" * 60)

    # Initialize components
    profiler = PerformanceProfiler()
    analytics = DrivingAnalyticsEngine()
    logger = DataLogger(output_dir="example_logs")
    heatmap_gen = HeatMapGenerator()

    # Start logging
    logger.start_logging()
    print("\n✓ Data logging started")

    # Open video
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open video source")
        return

    print("✓ Video source opened")
    print("\nProcessing... Press 'q' to quit\n")

    frame_count = 0
    start_time = time.time()

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Simulate processing (replace with actual perception pipeline)
        detections = []  # Your detections here
        tracked_objects = []  # Your tracked objects here
        lane_info = None  # Your lane info here

        # Update heatmap
        heatmap_gen.update(detections, frame.shape)

        # Calculate processing time
        processing_time = (time.time() - frame_start) * 1000

        # Update profiler
        profiler.update(processing_time)

        # Analyze behavior
        metrics = {
            'collision_risk': 'NONE',
            'closest_object_distance': float('inf')
        }
        behavior = analytics.analyze_behavior(metrics, lane_info, tracked_objects)

        # Log frame data
        if frame_count % 10 == 0:  # Log every 10th frame
            logger.log_frame(frame, detections, metrics)

        # Display stats every second
        if frame_count % 30 == 0:
            stats = profiler.get_stats()
            bottlenecks = profiler.get_bottlenecks()

            print(f"\n--- Frame {frame_count} Stats ---")
            print(f"FPS: {stats['fps']['current']:.1f} (avg: {stats['fps']['mean']:.1f})")
            print(f"CPU: {stats['cpu']['current']:.1f}% (avg: {stats['cpu']['mean']:.1f}%)")
            print(f"Memory: {stats['memory']['current']:.0f} MB")
            print(f"Latency: {stats['latency']['current']:.1f} ms")
            print(f"Behavior: {behavior['type']} (score: {behavior['score']})")

            if bottlenecks:
                print(f"⚠ Bottlenecks: {', '.join(bottlenecks)}")

        # Create visualization
        vis_frame = frame.copy()

        # Overlay heatmap
        heatmap = heatmap_gen.get_attention_heatmap()
        if heatmap is not None and heatmap.shape[:2] == vis_frame.shape[:2]:
            vis_frame = cv2.addWeighted(vis_frame, 0.7, heatmap, 0.3, 0)

        # Add performance info
        stats = profiler.get_stats()
        cv2.putText(vis_frame, f"FPS: {stats['fps']['current']:.1f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"CPU: {stats['cpu']['current']:.1f}%",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Behavior: {behavior['type']}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show frame
        cv2.imshow('Advanced Analytics', vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Stop logging
    logger.stop_logging()
    logger.close()

    # Print session summary
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)

    summary = analytics.get_session_summary()
    print(f"Duration: {summary['duration_formatted']}")
    print(f"Total Frames: {frame_count}")
    print(f"Average FPS: {frame_count / (time.time() - start_time):.1f}")
    print(f"Lane Departures: {summary['lane_departures']}")
    print(f"Collision Warnings: {summary['collision_warnings']}")
    print(f"Aggressive Events: {summary['aggressive_events']}")
    print(f"Behavior Score: {summary['average_behavior_score']:.1f}/100")

    print("\nRecommendations:")
    for rec in analytics.get_recommendations():
        print(f"  {rec}")

    print("\n✓ Session data logged to:", logger.db_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
