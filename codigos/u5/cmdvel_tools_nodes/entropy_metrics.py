#!/usr/bin/env python3
"""
entropy_metrics: entropías en línea sobre /cmd_vel (Unidad 5).

Publica en /entropy_metrics (std_msgs/Float64MultiArray), a publish_hz:
    data = [t, n_msgs, H_v, Hcond_v, H_dt]
calculadas sobre los mensajes recibidos en los últimos window_sec segundos
(NaN si hay menos de min_samples mensajes). Usa OnlineIndicators de u5_core.

Uso:
  ros2 run cmdvel_tools entropy_metrics --ros-args -p use_sim_time:=true \
      -p window_sec:=5.0 -p min_samples:=30 -p publish_hz:=2.0
"""
import rclpy
from rclpy.node import Node
from cmdvel_tools.cmdvel_msg import declare_stamped, msg_type, read_cmd
from std_msgs.msg import Float64MultiArray

from cmdvel_tools.u5_core import OnlineIndicators

FIELDS = ['t', 'n_msgs', 'H_v', 'Hcond_v', 'H_dt']


class EntropyMetrics(Node):
    def __init__(self):
        super().__init__('entropy_metrics')
        p = self.declare_parameter
        self.ind = OnlineIndicators(
            window_sec=float(p('window_sec', 5.0).value),
            min_samples=int(p('min_samples', 30).value),
            v_range=(float(p('v_min', -0.22).value), float(p('v_max', 0.22).value)),
            n_bins=int(p('n_bins', 15).value),
            dt_max=float(p('dt_max', 0.3).value),
            n_bins_dt=int(p('n_bins_dt', 20).value))
        self.pub = self.create_publisher(Float64MultiArray, 'entropy_metrics', 10)
        self.create_subscription(msg_type(declare_stamped(self)), 'cmd_vel', self.on_msg, 100)
        self.create_timer(1.0 / float(p('publish_hz', 2.0).value), self.tick)
        self.get_logger().info('entropy_metrics: data = [' + ', '.join(FIELDS) + ']')

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def on_msg(self, msg):
        t = self.now()
        if t > 0.0:
            v, w, _ = read_cmd(msg)
            self.ind.add(t, v, w)

    def tick(self):
        t = self.now()
        if t == 0.0:
            return
        m = self.ind.compute(t)
        out = Float64MultiArray()
        out.data = [t, float(m['n_msgs']), m['H_v'], m['Hcond_v'], m['H_dt']]
        self.pub.publish(out)


def main():
    rclpy.init()
    node = EntropyMetrics()
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
