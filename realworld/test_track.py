import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray

class TestTrackNode(Node):
    def __init__(self):
        super().__init__('test_track_node')
        self.get_logger().info('Test Track Node has been started.')
        self.pub = self.create_publisher(Int32MultiArray, '/tracking_points', 10)
        self.timer = self.create_timer(1.0, self.publish_tracking_points)
    
    def publish_tracking_points(self):
        msg = Int32MultiArray()
        msg.data = [0, 549, 481, 1, 451, 440]  # Example tracking points
        self.pub.publish(msg)
        self.get_logger().info(f'Published tracking points: {msg.data}')
        self.timer.cancel()  # Publish only once for testing

def main(args=None):
    rclpy.init(args=args)
    node = TestTrackNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()