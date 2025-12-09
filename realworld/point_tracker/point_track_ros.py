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

"""Live Demo for JAX Online TAPIR."""

import time
import cv2
import jax
import jax.numpy as jnp
import numpy as np
from tapnet.models import tapir_model
from tapnet.utils import model_utils
import os
import threading
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy

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

        # Reentrant group so timer + subs can run concurrently
        self.cb_group = ReentrantCallbackGroup()
        
        # Load and initialize model
        self.load_model()
        
        # Initialize query features and state - will be set properly in image_callback
        self.query_features = None
        self.causal_state = None
        
        # Create publishers
        self.tracking_points_pub = self.create_publisher(
            Int32MultiArray, 
            '/current_tracking_points', 
            10
            )
        
        self.rgb_frame = None
        self.frame_lock = threading.Lock()
        self.tracking_timer = None                   # NEW: timer handle
        self._tracking_active = False                # preserve existing flag
        self.step_lock = threading.Lock()
        self.add_points_queue = []
        self.points_queue_lock = threading.Lock()
        
        # Create subscribers
        self.rgb_sub = self.create_subscription(
            Image, "/rgb/image_raw", self.image_callback, 
            qos_profile_sensor_data,
            callback_group=self.cb_group)
        self.points_sub = self.create_subscription(
            Int32MultiArray, '/tracking_points', self.point_callback, 10, callback_group=self.cb_group)
        
        #self.get_logger().info(f"PointTrackerNode initialized (debug={self.debug})")
        
    def load_model(self):
        """Load and initialize the TAPIR model"""
        #self.get_logger().info("Creating model...")
        params, state = self.load_checkpoint("./checkpoints/causal_tapir_checkpoint.npy")
        self.tapir = tapir_model.ParameterizedTAPIR(
            params=params,
            state=state,
            tapir_kwargs=dict(
                use_causal_conv=True, bilinear_interp_with_depthwise_conv=False
            ),
        )
        self.online_init_apply = jax.jit(self.online_model_init)
        self.online_predict_apply = jax.jit(self.online_model_predict)
        #self.get_logger().info("Model loaded and initialized.")
        
    def load_checkpoint(self, checkpoint_path):
        ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
        return ckpt_state["params"], ckpt_state["state"]
        
    def preprocess_frames(self, frames):
        """Preprocess frames to model inputs."""
        frames = frames.astype(np.float32)
        frames = frames / 255 * 2 - 1
        return frames
        
    def online_model_init(self, frames, points):
        """Initialize query features for the query points."""
        frames = self.preprocess_frames(frames)
        feature_grids = self.tapir.get_feature_grids(frames, is_training=False)
        features = self.tapir.get_query_features(
            frames,
            is_training=False,
            query_points=points,
            feature_grids=feature_grids,
        )
        return features
        
    def postprocess_occlusions(self, occlusions, expected_dist):
        """Process occlusion predictions"""
        visibles = (1 - jax.nn.sigmoid(occlusions)) * (1 - jax.nn.sigmoid(expected_dist)) > 0.5
        return visibles
        
    def online_model_predict(self, frames, features, causal_context):
        """Compute point tracks and occlusions given frames and query points."""
        frames = self.preprocess_frames(frames)
        feature_grids = self.tapir.get_feature_grids(frames, is_training=False)
        trajectories = self.tapir.estimate_trajectories(
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
        tracks = trajectories["tracks"][-1]
        occlusions = trajectories["occlusion"][-1]
        uncertainty = trajectories["expected_dist"][-1]
        visibles = self.postprocess_occlusions(occlusions, uncertainty)
        return tracks, visibles, causal_context
        
    def image_callback(self, msg):
        """Process received image messages and convert to numpy array, crop to square."""
        #self.get_logger().debug("Received image data.")
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Crop image to square
        trunc = abs(frame.shape[1] - frame.shape[0]) // 2
        if frame.shape[1] > frame.shape[0]:
            frame = frame[:, trunc:-trunc]
        elif frame.shape[1] < frame.shape[0]:
            frame = frame[trunc:-trunc]

        # Store atomically
        acquired = self.frame_lock.acquire(timeout=1.0)
        if not acquired:
            #self.get_logger().warning("Failed to acquire frame lock in image callback")
            return
        else:
            self.rgb_frame = frame
            self.frame_lock.release()

        if self.query_features is None and not self.first_frame_received:
            #self.get_logger().info("First frame received, ready for tracking points")
            self.first_frame_received = True

    def point_callback(self, msg):
        """Handle tracking point information from external source."""
        #self.get_logger().info(f"Received tracking point message: {msg}.")
        data = np.array(msg.data).reshape(-1, 3)
        #self.get_logger().info(f"Received tracking points: {data}")
        self.point_idx = []
        
        if len(data) > NUM_POINTS:
            #self.get_logger().warning(f"Received more than {NUM_POINTS} points, only using the first {NUM_POINTS}.")
            data = data[:NUM_POINTS]
        
        # Process each point and add to tracking
        for i, point in enumerate(data):
            idx, x, y = point
            self.point_idx.append(int(idx))
            # Adjust x coordinate for cropping
            x_adjusted = x - (1280 - 720) / 2
            #self.get_logger().info(f"Received point {int(idx)}: ({x_adjusted}, {y})")
            
            # Initialize tracking for this point
            # self.add_tracking_point(int(idx), x_adjusted, y)
            acquired = self.points_queue_lock.acquire(timeout=1.0)
            if not acquired:
                #self.get_logger().warning("Failed to acquire points queue lock in point callback")
                return
            else:
                self.add_points_queue.append((int(idx), x_adjusted, y))
                self.points_queue_lock.release()
        #self.get_logger().info(f"Queued {len(data)} tracking points for addition.")
        if not self._tracking_active:
            self._tracking_active = True
            # ~30 Hz; adjust if you want lighter CPU load (e.g., 0.05 for 20 Hz)
            self.tracking_timer = self.create_timer(10.0, self.tracking_step, callback_group=self.cb_group)
            #self.get_logger().info("Started tracking timer")

    def add_tracking_point_from_queue(self):
        """Process points in the queue to add them to tracking."""
        while self.add_points_queue:
            if self.points_queue_lock.acquire():
                point = self.add_points_queue.pop(0)
                self.points_queue_lock.release()
                idx, x, y = point
                self.add_tracking_point(idx, x, y)
            else:
                time.sleep(0.1)  # Avoid busy waiting
        #self.get_logger().info("Finished processing points from queue.")
            
    def add_tracking_point(self, point_idx, x, y):
        """Add a new point to track"""
        if self.rgb_frame is None:
            #self.get_logger().warning("No frame available for tracking point initialization")
            return
    
            # Find available slot
        available_slot = None
        for i in range(NUM_POINTS):
            if not self.have_point[i]:
                available_slot = i
                break
                
        if available_slot is None:
            #self.get_logger().warning("No available slots for new tracking point")
            return
            
        # Initialize query features if this is the first point
        if self.query_features is None:
            frame_tensor = jnp.array(self.rgb_frame)
            # Initialize with dummy points for all slots
            dummy_points = jnp.zeros([NUM_POINTS, 3], dtype=jnp.float32)
            self.query_features = self.online_init_apply(
                frames=model_utils.preprocess_frames(frame_tensor[None, None]), 
                points=dummy_points[None, :]
            )
            self.causal_state = self.tapir.construct_initial_causal_state(
                NUM_POINTS, len(self.query_features.resolutions) - 1
            )
            #self.get_logger().info("Intialized query features and causal state")
        
        # Now update the specific point
        frame_tensor = jnp.array(self.rgb_frame)
        query_point = jnp.array([0, y, x], dtype=jnp.float32)
        
        init_query_features = self.online_init_apply(
            frames=model_utils.preprocess_frames(frame_tensor[None, None]), 
            points=query_point[None, None]
        )
        
        # Update query features for this point
        self.query_features, self.causal_state = self.tapir.update_query_features(
            query_features=self.query_features,
            new_query_features=init_query_features,
            idx_to_update=np.array([available_slot]),  # numpy array as expected
            causal_state=self.causal_state,
        )
        
        self.have_point[available_slot] = True
        #self.get_logger().info(f"Added tracking point {point_idx} at slot {available_slot}")
        
    def tracking_step(self):
        """One non-blocking tracking step driven by a ROS timer (asynchronous)."""
        if not self.step_lock.acquire(blocking=False):
            #self.get_logger().debug("Skipping tracking step to avoid overlap")
            return
        
        #self.get_logger().info("adding tracking points")
        self.add_tracking_point_from_queue()
        
        if not any(self.have_point):
            return

        acquired = self.frame_lock.acquire(timeout=1.0)
        if not acquired:
            #self.get_logger().warning("Failed to acquire frame lock in tracking step")
            return
        else:
            if self.rgb_frame is None:
                return
            frame_np = self.rgb_frame.copy()
            self.frame_lock.release()

        #self.get_logger().info("Performing tracking step")
        
        frame_disp = frame_np.copy()  # for imshow (optional)
        if self.query_features is not None and self.causal_state is not None:
            frame_tensor = jnp.array(frame_np)
            track, visible, self.causal_state = self.online_predict_apply(
                frames=model_utils.preprocess_frames(frame_tensor[None, None]),
                features=self.query_features,
                causal_context=self.causal_state,
            )
            track = np.array(track)
            visible = np.array(visible)

            tracked_points = []
            for i in range(NUM_POINTS):
                if self.have_point[i] and visible[0, i, 0]:
                    x, y = int(track[0, i, 0, 0]), int(track[0, i, 0, 1])
                    x_original = int(x + (1280 - 720) / 2)  # keep your original offset
                    pt_id = int(self.point_idx[i] if i < len(self.point_idx) else i)
                    tracked_points.append((pt_id, x_original, int(y)))
                    cv2.circle(frame_disp, (x, y), 5, (255, 0, 0), -1)

            if tracked_points:
                msg_to_send = Int32MultiArray()
                msg_to_send.data = [int(v) for triplet in tracked_points for v in triplet]
                self.tracking_points_pub.publish(msg_to_send)
            #self.get_logger().info(f"Published tracked points: {tracked_points}")
        self.step_lock.release()

def main(args=None):
    """Main function to run the point tracker node"""
    print("Welcome to the TAPIR JAX live demo.")
    print("Please note that if the framerate is low (<~12 fps), TAPIR performance")
    print("may degrade and you may need a more powerful GPU.")
    rclpy.init(args=args)

    try:
        # Check for debug flag in command line arguments
        import sys
        debug_mode = '--debug' in sys.argv
        
        node = PointTrackerNode(debug=True)
        # multi-threaded
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)

        # Wait for first frame and tracking points
        #node.get_logger().info("Waiting for tracking point messages...")
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