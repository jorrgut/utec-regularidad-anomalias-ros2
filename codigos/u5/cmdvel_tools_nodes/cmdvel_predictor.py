#!/usr/bin/env python3
"""
cmdvel_predictor: predictibilidad en línea de /cmd_vel (Unidad 5).

Predictor de persistencia mensaje a mensaje (equivale al ZOH de la Unidad 2).
Publica en /cmd_vel_pred_metrics (std_msgs/Float64MultiArray), a publish_hz:
    data = [t, n_msgs, mse_v, mse_w]
con el Rolling MSE del error de persistencia sobre los últimos window_sec segundos.

No publica NLL: con sigma fija, el NLL es una función lineal del MSE, y el detector
normaliza cada indicador con mediana y MAD, así que el z-score es el mismo (apunte,
"La normalización vuelve irrelevante la elección entre MSE y NLL").

Uso:
  ros2 run cmdvel_tools cmdvel_predictor --ros-args -p use_sim_time:=true -p window_sec:=5.0
"""
import rclpy
from rclpy.node import Node
from cmdvel_tools.cmdvel_msg import declare_stamped, msg_type, read_cmd
from std_msgs.msg import Float64MultiArray

from cmdvel_tools.u5_core import OnlineIndicators

FIELDS = ['t', 'n_msgs', 'mse_v', 'mse_w']


class CmdVelPredictor(Node):
    def __init__(self):
        super().__init__('cmdvel_predictor')
        p = self.declare_parameter
        self.ind = OnlineIndicators(window_sec=float(p('window_sec', 5.0).value),
                                    min_samples=int(p('min_samples', 30).value))
        self.pub = self.create_publisher(Float64MultiArray, 'cmd_vel_pred_metrics', 10)
        self.create_subscription(msg_type(declare_stamped(self)), 'cmd_vel', self.on_msg, 100)
        self.create_timer(1.0 / float(p('publish_hz', 2.0).value), self.tick)
        self.get_logger().info('cmdvel_predictor: data = [' + ', '.join(FIELDS) + ']')

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
        out.data = [t, float(m['n_msgs']), m['mse_v'], m['mse_w']]
        self.pub.publish(out)


def main():
    rclpy.init()
    node = CmdVelPredictor()
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
