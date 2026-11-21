"""Nodo mínimo para verificar el workspace (Unidad 1, sección Workspace ROS 2).

Ubicación: ros2_ws/src/cmdvel_tools/cmdvel_tools/dummy_node.py
"""
import rclpy
from rclpy.node import Node


class DummyNode(Node):
    def __init__(self):
        super().__init__('dummy_node')
        self.get_logger().info('Dummy node running')


def main():
    rclpy.init()
    node = DummyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
