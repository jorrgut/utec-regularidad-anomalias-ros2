#!/usr/bin/env python3
"""
anomaly_detector: score, umbral y persistencia en línea (Unidad 5).

Suscribe /entropy_metrics y /cmd_vel_pred_metrics, y a publish_hz publica:
    /anomaly_score  std_msgs/Float64           score S_t
    /anomaly_alert  std_msgs/Bool              alarma A_t
    /anomaly_debug  std_msgs/Float64MultiArray [t, S, tau, alarma, contribuciones...]
                    (contribuciones en el orden de INDICATORS: mse_v, mse_w, Hcond_v, H_dt, H_v)

Parámetros:
    calib_file     YAML de calibrate_detector.py (OBLIGATORIO)
    tau            umbral; <= 0 usa el de la calibración
    K              persistencia (superaciones consecutivas)
    rule           'consecutive' o 'cusum'
    cusum_k, cusum_h   parámetros del CUSUM (sobre el score centrado y escalado)
    min_ready_sec  calentamiento: no decide antes de este tiempo
    max_age_sec    descarta indicadores más viejos que esto (nodo caído)

Uso:
  ros2 run cmdvel_tools anomaly_detector --ros-args -p use_sim_time:=true \
      -p calib_file:=outputs_u5/calib.yaml -p K:=3
"""
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, Float64MultiArray

from cmdvel_tools.u5_core import INDICATORS, load_calibration, score, OnlinePersistence

ENTROPY_FIELDS = ['t', 'n_msgs', 'H_v', 'Hcond_v', 'H_dt']
PRED_FIELDS = ['t', 'n_msgs', 'mse_v', 'mse_w']


class AnomalyDetector(Node):
    def __init__(self):
        super().__init__('anomaly_detector')
        p = self.declare_parameter
        path = p('calib_file', '').value
        if not path:
            raise ValueError('falta el parámetro calib_file (YAML de calibrate_detector.py)')
        self.cfg, self.centers, self.weights, self.direction = load_calibration(path)
        tau = float(p('tau', 0.0).value)
        self.tau = tau if tau > 0 else float(self.cfg['tau'])
        self.z_max = self.cfg.get('z_max')
        rule = p('rule', 'consecutive').value
        self.persist = OnlinePersistence(
            rule=rule, K=int(p('K', 3).value),
            cusum_k=float(p('cusum_k', 2.0).value), cusum_h=float(p('cusum_h', 5.0).value))
        self.s_med = float(self.cfg.get('score_median', 0.0))
        self.s_scale = float(self.cfg.get('score_scale', 1.0)) or 1.0
        self.min_ready = float(p('min_ready_sec', 10.0).value)
        self.max_age = float(p('max_age_sec', 2.0).value)

        self.latest = {}          # indicador -> (t, valor)
        self.create_subscription(Float64MultiArray, 'entropy_metrics',
                                 lambda m: self.on_metrics(m, ENTROPY_FIELDS), 10)
        self.create_subscription(Float64MultiArray, 'cmd_vel_pred_metrics',
                                 lambda m: self.on_metrics(m, PRED_FIELDS), 10)
        self.pub_s = self.create_publisher(Float64, 'anomaly_score', 10)
        self.pub_a = self.create_publisher(Bool, 'anomaly_alert', 10)
        self.pub_d = self.create_publisher(Float64MultiArray, 'anomaly_debug', 10)
        self.t0 = None
        self.alarm = False
        self.create_timer(1.0 / float(p('publish_hz', 2.0).value), self.tick)
        self.get_logger().info(f'Detector: tau = {self.tau:.3f}, regla = {rule}, '
                               f'calibración = {path}')

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def on_metrics(self, msg, fields):
        d = list(msg.data)
        if len(d) != len(fields):
            self.get_logger().warn(f'mensaje con {len(d)} campos, se esperaban {len(fields)}',
                                   throttle_duration_sec=10.0)
            return
        t = d[0]
        for name, val in zip(fields[2:], d[2:]):
            self.latest[name] = (t, val)

    def tick(self):
        t = self.now()
        if t == 0.0:
            return
        if self.t0 is None:
            self.t0 = t
        values = {}
        for name in INDICATORS:
            tv = self.latest.get(name)
            fresh = tv is not None and (t - tv[0]) <= self.max_age
            values[name] = tv[1] if fresh else math.nan
        S, contrib = score(values, self.centers, self.weights, self.direction, self.z_max)
        S = float(S)
        ready = (t - self.t0) >= self.min_ready
        if ready:
            s_in = (S - self.s_med) / self.s_scale if self.persist.rule == 'cusum' else S
            self.alarm = bool(self.persist.update(s_in, self.tau))
        new_alarm = self.alarm
        self.pub_s.publish(Float64(data=S))
        self.pub_a.publish(Bool(data=new_alarm))
        dbg = Float64MultiArray()
        dbg.data = [t, S, self.tau, float(new_alarm)] + \
                   [float(contrib.get(k, 0.0)) for k in INDICATORS]
        self.pub_d.publish(dbg)
        if new_alarm:
            top = max(contrib, key=contrib.get) if contrib else '—'
            self.get_logger().warn(f'ALARMA t = {t:.1f} s | score {S:.2f} | dominante: {top}',
                                   throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = AnomalyDetector()
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
