# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Live Demo for PyTorch Online TAPIR."""

import time

import cv2
import numpy as np

from tapnet.torch import tapir_model
import torch
import torch.nn.functional as F
import tree
import pyrealsense2 as rs
from std_msgs.msg import Int32MultiArray
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

NUM_POINTS = 8

class PointTrackerNode(Node):
    def __init__(self, debug=False):
        super().__init__('point_tracking_node')
        
        self.debug = debug
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Initialize tracking variables
        self.rgb_frame = None
        self.first_frame_received = False
        self.have_point = [False] * NUM_POINTS
        self.point_idx = []
        self.query_frame = True
        self.next_query_idx = 0
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        # Load and initialize model
        self.load_model()
        
        # Initialize query features and state - will be set properly in image_callback
        self.query_features = None
        self.causal_state = None
        
        # Create publishers
        self.tracking_points_pub = self.create_publisher(
            Int32MultiArray, '/current_tracking_points', 10)
        
        # Create subscribers
        self.rgb_sub = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.image_callback, 10)
        self.points_sub = self.create_subscription(
            Int32MultiArray, '/tracking_points', self.point_callback, 10)
        
        # Send debug test message if debug mode is enabled
        if self.debug:
            self.debug_timer = self.create_timer(2.0, self.send_debug_message)  # Send test message after 2 seconds
            self.debug_pub = self.create_publisher(Int32MultiArray, '/tracking_points', 10)
            self.get_logger().info("Debug mode enabled: will send test tracking points.")

        self.get_logger().info(f"PointTrackerNode initialized (debug={self.debug})")
        
    def send_debug_message(self):
        """Send a test message to the tracking points topic for debugging"""
        test_msg = Int32MultiArray()
        # Send test tracking points: point 1 at (400, 300), point 2 at (600, 400)
        test_msg.data = [1, 400, 300, 2, 600, 400]
        self.debug_pub.publish(test_msg)
        self.get_logger().info("Sent debug test message: [1, 400, 300, 2, 600, 400]")
        
        # Cancel the timer after sending the message once
        self.destroy_timer(self.debug_timer)
        
    def load_model(self):
        """Load and initialize the TAPIR model following the live_demo.py pattern"""
        self.get_logger().info("Creating model...")
        
        # Load checkpoint similar to JAX version
        dir_path = os.path.dirname(os.path.realpath(__file__))
        checkpoint_path = os.path.join(dir_path, "checkpoints/causal_bootstapir_checkpoint.pt")
        
        # Create model and load state dict
        self.model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Disable gradients for inference
        torch.set_grad_enabled(False)
        
        self.get_logger().info("Model loaded successfully")
        
    def preprocess_frames(self, frames):
        """Preprocess frames to model inputs - following live_demo.py pattern"""
        # Convert to float and normalize to [-1, 1] range
        frames = frames.float() / 255.0 * 2.0 - 1.0
        return frames
        
    def online_model_init(self, frames, points):
        """Initialize query features for the query points - following live_demo.py pattern"""
        frames = self.preprocess_frames(frames)
        feature_grids = self.model.get_feature_grids(frames, is_training=False)
        features = self.model.get_query_features(
            frames,
            is_training=False,
            query_points=points,
            feature_grids=feature_grids,
        )
        return features
        
    def postprocess_occlusions(self, occlusions, expected_dist):
        """Process occlusion predictions - following live_demo.py pattern"""
        visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
        return visibles
        
    def online_model_predict(self, frames, features, causal_context):
        """Compute point tracks and occlusions - following live_demo.py pattern"""
        frames = self.preprocess_frames(frames)
        feature_grids = self.model.get_feature_grids(frames, is_training=False)
        trajectories = self.model.estimate_trajectories(
            frames.shape[-3:-1],
            is_training=False,
            feature_grids=feature_grids,
            query_features=features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=causal_context,
            get_causal_context=True,
        )
        causal_context = trajectories["causal_context"]
        del trajectories["causal_context"]
        
        # Extract results similar to JAX version
        tracks = trajectories["tracks"][-1]
        occlusions = trajectories["occlusion"][-1]
        expected_dist = trajectories["expected_dist"][-1]
        visibles = self.postprocess_occlusions(occlusions, expected_dist)
        
        return tracks, visibles, causal_context
        
    def image_callback(self, msg):
        """Process received image messages and convert to numpy array, crop to square."""
        self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Crop image to square - following live_demo.py pattern
        trunc = abs(self.rgb_frame.shape[1] - self.rgb_frame.shape[0]) // 2
        if self.rgb_frame.shape[1] > self.rgb_frame.shape[0]:
            self.rgb_frame = self.rgb_frame[:, trunc:-trunc]
        elif self.rgb_frame.shape[1] < self.rgb_frame.shape[0]:
            self.rgb_frame = self.rgb_frame[trunc:-trunc]
            
        # Set flag that first frame is received
        if not self.first_frame_received:
            self.first_frame_received = True
            self.get_logger().info("First frame received, ready for tracking points")
            
            # Initialize model similar to JAX demo compilation step
            self.compile_model()
            
    def compile_model(self):
        """Compile model similar to JAX demo - run once for optimization"""
        if self.rgb_frame is not None:
            self.get_logger().info("Compiling PyTorch model (this may take a while...)")
            
            frame_tensor = torch.tensor(self.rgb_frame, dtype=torch.float32).to(self.device)
            dummy_points = torch.zeros([NUM_POINTS, 3], dtype=torch.float32).to(self.device)
            
            # Run once to warm up
            with torch.no_grad():
                _ = self.online_model_init(
                    frames=frame_tensor[None, None], 
                    points=dummy_points[None, 0:1]
                )
                
                # Initialize full query features and causal state
                self.query_features = self.online_model_init(
                    frames=frame_tensor[None, None], 
                    points=dummy_points[None, :]
                )
                
                self.causal_state = self.model.construct_initial_causal_state(
                    NUM_POINTS, len(self.query_features.resolutions) - 1
                )
                
                # Run a dummy prediction to warm up
                _, _, _ = self.online_model_predict(
                    frames=frame_tensor[None, None],
                    features=self.query_features,
                    causal_context=self.causal_state,
                )
                
            self.get_logger().info("Model compilation completed")
            
    def point_callback(self, msg):
        """Handle tracking point information from external source."""
        self.get_logger().info(f"Received tracking point message: {msg}.")
        data = np.array(msg.data).reshape(-1, 3)
        self.get_logger().info(f"Received tracking points: {data}")
        
        if not self.first_frame_received:
            self.get_logger().warning("No frame received yet, cannot initialize tracking")
            return
            
        if len(data) > NUM_POINTS:
            self.get_logger().warning(f"Received more than {NUM_POINTS} points, only using the first {NUM_POINTS}.")
            data = data[:NUM_POINTS]
        
        # Process each point similar to JAX demo mouse clicks
        for i, point in enumerate(data):
            idx, x, y = point
            # Adjust x coordinate for cropping
            x_adjusted = x - (1280 - 720) / 2
            self.get_logger().info(f"Received point {int(idx)}: ({x_adjusted}, {y})")
            
            # Add point following JAX demo pattern
            self.add_point(int(idx), x_adjusted, y)
            
        # Start tracking loop
        if not hasattr(self, '_tracking_active'):
            self._tracking_active = True
            self.start_tracking()
            
    def add_point(self, point_idx, x, y):
        """Add a new point to track - following JAX demo pattern"""
        if self.rgb_frame is None or self.query_features is None:
            self.get_logger().warning("Model not ready for tracking")
            return
            
        # Find next available slot (following JAX demo's next_query_idx pattern)
        available_slot = self.next_query_idx
        
        # Convert position to query point format (t, y, x) like JAX demo
        pos = (y, x)
        query_point = torch.tensor([0] + list(pos), dtype=torch.float32).to(self.device)
        
        # Initialize query features for this point
        frame_tensor = torch.tensor(self.rgb_frame, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            init_query_features = self.online_model_init(
                frames=frame_tensor[None, None],
                points=query_point[None, None],
            )
            
            # Update query features following JAX demo pattern
            self.query_features, self.causal_state = self.model.update_query_features(
                query_features=self.query_features,
                new_query_features=init_query_features,
                idx_to_update=np.array([available_slot]),  # numpy array as in JAX demo
                causal_state=self.causal_state,
            )
            
        self.have_point[available_slot] = True
        self.point_idx.append(point_idx)
        self.next_query_idx = (self.next_query_idx + 1) % NUM_POINTS
        
        self.get_logger().info(f"Added tracking point {point_idx} at slot {available_slot}")
            
    def start_tracking(self):
        """Start the main tracking loop - following JAX demo pattern"""
        self.get_logger().info("Starting tracking loop")
        cv2.namedWindow("Point Tracking")
        
        with torch.no_grad():
            while rclpy.ok() and any(self.have_point):
                if self.rgb_frame is None:
                    time.sleep(0.01)
                    continue
                    
                frame = self.rgb_frame.copy()
                
                # Run prediction if we have active points (following JAX demo pattern)
                if any(self.have_point) and self.query_features is not None and self.causal_state is not None:
                    frame_tensor = torch.tensor(self.rgb_frame, dtype=torch.float32).to(self.device)
                    
                    tracks, visibles, self.causal_state = self.online_model_predict(
                        frames=frame_tensor[None, None],
                        features=self.query_features,
                        causal_context=self.causal_state,
                    )
                    
                    # Convert to numpy for processing
                    tracks = tracks.cpu().numpy()
                    visibles = visibles.cpu().numpy()
                    
                    tracked_points = []
                    
                    # Draw tracking results following JAX demo pattern
                    for i in range(NUM_POINTS):
                        if self.have_point[i] and visibles[0, i, 0]:
                            # Get track coordinates
                            track_x = int(tracks[0, i, 0, 0])
                            track_y = int(tracks[0, i, 0, 1])
                            
                            # Draw circle
                            cv2.circle(frame, (track_x, track_y), 5, (255, 0, 0), -1)
                            
                            # Adjust coordinates back for publishing
                            x_original = int(track_x + (1280 - 720) / 2)
                            point_id = self.point_idx[i] if i < len(self.point_idx) else i
                            tracked_points.append((int(point_id), x_original, int(track_y)))

                    # Publish tracked points
                    if tracked_points:
                        msg_to_send = Int32MultiArray()
                        msg_to_send.data = [int(item) for sublist in tracked_points for item in sublist]
                        self.tracking_points_pub.publish(msg_to_send)

                # Display frame (flipped horizontally like JAX demo)
                cv2.imshow("Point Tracking", frame[:, ::-1])
                key = cv2.waitKey(1)
                if key == 27:  # exit on ESC
                    break
                    
                # Process ROS callbacks
                rclpy.spin_once(self, timeout_sec=0.01)

def main(args=None):
    """Main function to run the point tracker node"""
    print("Welcome to the TAPIR PyTorch live demo.")
    print("Please note that if the framerate is low (<~12 fps), TAPIR performance")
    print("may degrade and you may need a more powerful GPU.")
    
    rclpy.init(args=args)
    
    try:
        # Check for debug flag in command line arguments
        import sys
        debug_mode = '--debug' in sys.argv
        
        node = PointTrackerNode(debug=debug_mode)
        # multi-threaded
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)

        # Wait for first frame and tracking points
        node.get_logger().info("Waiting for tracking point messages...")
        executor.spin()
        
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()