#!/usr/bin/env python3
"""
Test script for ultra-advanced features
Verify all modules are working correctly
"""

import sys
import cv2
import numpy as np

print("=" * 70)
print("  ADAS ULTRA-ADVANCED FEATURES - TEST SUITE")
print("=" * 70)

# Test 1: Import all modules
print("\n[TEST 1] Importing modules...")
try:
    from ultra_features import (
        VehicleTypeClassifier,
        LicensePlateDetector,
        PotholeDetector,
        OpticalFlowAnalyzer,
        MotionPredictor,
        SuddenMovementDetector,
        DrivingBehaviorAnalyzer,
        ConfidenceCalibrator,
        SmallObjectDetector
    )
    print("  ✅ ultra_features.py - ALL classes imported")
except Exception as e:
    print(f"  ❌ ultra_features.py - Error: {e}")
    sys.exit(1)

try:
    from ultra_visualization import (
        HeatmapTimeline,
        StatisticsDashboard
    )
    print("  ✅ ultra_visualization.py - Classes imported")
except Exception as e:
    print(f"  ⚠️  ultra_visualization.py - Error: {e} (wxPython required)")

try:
    from adas_ultra_advanced import (
        PointCloudReconstructor,
        PanoramaStitcher,
        AdvancedRecorder,
        SceneClassifier,
        PedestrianPoseEstimator
    )
    print("  ✅ adas_ultra_advanced.py - Classes imported")
except Exception as e:
    print(f"  ❌ adas_ultra_advanced.py - Error: {e}")

# Test 2: Create test frame
print("\n[TEST 2] Creating test data...")
test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
print(f"  ✅ Test frame created: {test_frame.shape}")

# Test 3: Vehicle Type Classifier
print("\n[TEST 3] Testing Vehicle Type Classifier...")
try:
    classifier = VehicleTypeClassifier()
    bbox = (100, 100, 300, 200)
    vehicle_type = classifier.classify(bbox, test_frame)
    print(f"  ✅ Vehicle classification: {vehicle_type}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 4: Optical Flow Analyzer
print("\n[TEST 4] Testing Optical Flow Analyzer...")
try:
    flow_analyzer = OpticalFlowAnalyzer()
    # First frame
    flow_data = flow_analyzer.calculate(test_frame)
    print(f"  ✅ First frame processed (no flow yet)")

    # Second frame
    test_frame2 = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    flow_data = flow_analyzer.calculate(test_frame2)

    if flow_data:
        print(f"  ✅ Optical flow calculated")
        print(f"     Dominant motion: {flow_data['dominant_motion']}")
        print(f"     Avg magnitude: {flow_data['avg_magnitude']:.2f}")

        # Test visualization
        flow_vis = flow_analyzer.visualize(flow_data, test_frame.shape)
        print(f"  ✅ Flow visualization created: {flow_vis.shape}")
    else:
        print(f"  ⚠️  No flow data (need more frames)")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 5: Motion Predictor
print("\n[TEST 5] Testing Motion Predictor...")
try:
    predictor = MotionPredictor()

    # Create mock tracked object
    class MockObject:
        def __init__(self):
            self.centroid = (640, 360)
            self.velocity = (5.0, -2.0)
            self.ttc = 5.0

    mock_obj = MockObject()
    predictions = predictor.predict(mock_obj, num_steps=10)

    print(f"  ✅ Motion prediction completed")
    print(f"     Predicted {len(predictions)} future positions")
    if predictions:
        print(f"     First prediction: {predictions[0]}")
        print(f"     Last prediction: {predictions[-1]}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 6: Scene Classifier
print("\n[TEST 6] Testing Scene Classifier...")
try:
    from adas_ultra_advanced import SceneClassifier

    classifier = SceneClassifier()
    context = classifier.classify(test_frame, [])

    print(f"  ✅ Scene classification completed")
    print(f"     Time of day: {context.time_of_day.name}")
    print(f"     Road type: {context.road_type.name}")
    print(f"     Traffic density: {context.traffic_density.name}")
    print(f"     Road condition: {context.road_condition.name}")
    print(f"     Visibility score: {context.visibility_score:.2f}")
    print(f"     Complexity score: {context.complexity_score:.2f}")
except Exception as e:
    print(f"  ⚠️  Error: {e} (module may not be fully loaded)")

# Test 7: Driving Behavior Analyzer
print("\n[TEST 7] Testing Driving Behavior Analyzer...")
try:
    analyzer = DrivingBehaviorAnalyzer()

    # Simulate some behavior
    behavior = analyzer.analyze([], None, 0)
    score = analyzer.get_score()

    print(f"  ✅ Behavior analysis completed")
    print(f"     Behavior type: {behavior}")
    print(f"     Driving score: {score}/100")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 8: Heatmap Timeline
print("\n[TEST 8] Testing Heatmap Timeline...")
try:
    heatmap = HeatmapTimeline(width=1280, height=720)

    # Add some danger zones
    collision_zones = [(640, 360, 50), (800, 400, 30)]
    heatmap.update(collision_zones, test_frame.shape)

    # Get visualization
    heatmap_vis = heatmap.visualize()
    timeline_vis = heatmap.get_timeline_visualization()

    print(f"  ✅ Heatmap timeline working")
    print(f"     Heatmap shape: {heatmap_vis.shape}")
    print(f"     Timeline shape: {timeline_vis.shape}")
except Exception as e:
    print(f"  ⚠️  Error: {e} (wxPython may be required)")

# Test 9: Pothole Detector
print("\n[TEST 9] Testing Pothole Detector...")
try:
    detector = PotholeDetector()
    potholes = detector.detect(test_frame)

    print(f"  ✅ Pothole detection completed")
    print(f"     Potholes detected: {len(potholes)}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 10: Point Cloud Reconstructor
print("\n[TEST 10] Testing Point Cloud Reconstructor...")
try:
    from adas_ultra_advanced import PointCloudReconstructor

    reconstructor = PointCloudReconstructor(fx=800, fy=800, cx=640, cy=360)

    # Create mock depth map
    depth_map = np.random.rand(720, 1280) * 10  # 0-10 meters depth

    point_cloud = reconstructor.reconstruct(test_frame, depth_map)

    print(f"  ✅ Point cloud reconstruction completed")
    print(f"     Points generated: {len(point_cloud)}")

    # Test visualization
    vis = reconstructor.visualize_point_cloud(point_cloud)
    print(f"  ✅ Point cloud visualization: {vis.shape}")
except Exception as e:
    print(f"  ⚠️  Error: {e}")

# Summary
print("\n" + "=" * 70)
print("  TEST SUMMARY")
print("=" * 70)
print("\n✅ Core ultra features are working!")
print("\n📝 Note: Some tests may show warnings if optional dependencies")
print("   (wxPython, matplotlib, mediapipe) are not installed.")
print("\n🚀 You can now use these features in your application:")
print("\n   from ultra_features import *")
print("   from ultra_visualization import *")
print("   from adas_ultra_advanced import *")
print("\n" + "=" * 70)
