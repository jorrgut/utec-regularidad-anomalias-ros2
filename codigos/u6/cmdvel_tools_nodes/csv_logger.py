#!/usr/bin/env python3
"""
csv_logger: registra los tópicos de las Unidades 5 y 6 en CSV (uno por tópico).

Versión de la Unidad 6: igual a la de la Unidad 5, más /model_metrics del nodo model_error.

    cmd_vel.csv               t_sec, v, w, dt   (formato del logger de la Unidad 1)
                              + t_stamp con TwistStamped (Jazzy): hora del EMISOR
    entropy_metrics.csv       t, n_msgs, H_v, Hcond_v, H_dt
    cmd_vel_pred_metrics.csv  t, n_msgs, mse_v, mse_w
    anomaly_debug.csv         t, S, tau, alarm, c_mse_v, c_mse_w, c_Hcond_v, c_H_dt, c_H_v
    model_metrics.csv         t, n_ventanas, err_medio, err_ultimo   (Unidad 6)

Los tiempos son los del reloj del nodo (sim time si use_sim_time:=true). Los archivos se
vacían a disco cada segundo, así que un Ctrl+C no pierde datos.

Uso:
  ros2 run cmdvel_tools csv_logger --ros-args -p use_sim_time:=true -p log_dir:=outputs_u5/run1
"""
import csv
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from cmdvel_tools.cmdvel_msg import declare_stamped, msg_type, read_cmd
from std_msgs.msg import Float64MultiArray

FILES = {
    'cmd_vel': ['t_sec', 'v', 'w', 'dt'],
    'entropy_metrics': ['t', 'n_msgs', 'H_v', 'Hcond_v', 'H_dt'],
    'cmd_vel_pred_metrics': ['t', 'n_msgs', 'mse_v', 'mse_w'],
    'anomaly_debug': ['t', 'S', 'tau', 'alarm', 'c_mse_v', 'c_mse_w', 'c_Hcond_v', 'c_H_dt',
                      'c_H_v'],
    'model_metrics': ['t', 'n_ventanas', 'err_medio', 'err_ultimo'],
}


def fmt(x):
    return '' if (isinstance(x, float) and math.isnan(x)) else f'{x:.6f}'


class CsvLogger(Node):
    def __init__(self):
        super().__init__('csv_logger')
        self.log_dir = self.declare_parameter('log_dir', 'outputs_u5/run').value
        self.stamped = declare_stamped(self)
        os.makedirs(self.log_dir, exist_ok=True)
        self.files, self.writers, self.counts = {}, {}, {}
        for name, header in FILES.items():
            f = open(os.path.join(self.log_dir, f'{name}.csv'), 'w', newline='')
            w = csv.writer(f)
            if name == 'cmd_vel' and self.stamped:
                header = header + ['t_stamp']
            w.writerow(header)
            self.files[name], self.writers[name], self.counts[name] = f, w, 0
        self.last_t = None
        self.create_subscription(msg_type(self.stamped), 'cmd_vel', self.on_cmd, 100)
        for name in ('entropy_metrics', 'cmd_vel_pred_metrics', 'anomaly_debug', 'model_metrics'):
            self.create_subscription(Float64MultiArray, name,
                                     lambda m, n=name: self.on_array(m, n), 20)
        self.create_timer(1.0, self.flush)
        self.get_logger().info(f'Registrando en {self.log_dir}/')

    def on_cmd(self, msg):
        t = self.get_clock().now().nanoseconds / 1e9
        if t == 0.0:
            return
        dt = '' if self.last_t is None else f'{t - self.last_t:.9f}'
        v, w, t_stamp = read_cmd(msg)
        row = [f'{t:.9f}', v, w, dt]
        if self.stamped:
            row.append(f'{t_stamp:.9f}')
        self.writers['cmd_vel'].writerow(row)
        self.counts['cmd_vel'] += 1
        self.last_t = t

    def on_array(self, msg, name):
        d = list(msg.data)
        if len(d) != len(FILES[name]):
            return
        self.writers[name].writerow([fmt(x) for x in d])
        self.counts[name] += 1

    def flush(self):
        for f in self.files.values():
            f.flush()

    def close(self):
        for f in self.files.values():
            f.close()
        self.get_logger().info('Filas registradas: ' +
                               ', '.join(f'{k}: {v}' for k, v in self.counts.items()))


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = CsvLogger()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
