#!/usr/bin/env python3
"""
model_error: error de un modelo aprendido sobre /cmd_vel, en línea (Unidad 6).

Hace en línea lo mismo que u6_activity4.py fuera de línea:
  1. arma la grilla con un temporizador a rate_hz (la del modelo): en cada tic toma el
     ÚLTIMO mensaje recibido (retención de orden cero) y cuenta los mensajes llegados desde
     el tic anterior (canal n_k); si el último mensaje es más viejo que max_gap_sec, hay un
     hueco y la ventana se vacía (calentamiento de nuevo);
  2. con las últimas T muestras (T + 1 para la LSTM), estandariza con la media y el desvío
     guardados en el modelo y calcula el error;
  3. a publish_hz publica en /model_metrics (std_msgs/Float64MultiArray):
        data = [t, n_ventanas, err_medio, err_ultimo]
     err_medio  promedio del error en los últimos window_sec segundos (el indicador)
     err_ultimo error de la última ventana (reacciona antes, más ruidoso)
     Antes de tener una ventana completa publica NaN.

Parámetros:
    model          prefijo del modelo de train_model.py (OBLIGATORIO), p. ej. models/ae_T40
    window_sec     ventana de agregación del error (5.0, como los indicadores de la Unidad 5)
    publish_hz     2.0
    max_gap_sec    0.5

Requiere PyTorch en el entorno de ROS 2 (ver README, "Instalación").

Uso:
  ros2 run cmdvel_tools model_error --ros-args -p use_sim_time:=true \
      -p model:=/root/ros2_ws/models/ae_T40
"""
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from cmdvel_tools.cmdvel_msg import declare_stamped, msg_type, read_cmd
from std_msgs.msg import Float64MultiArray

from cmdvel_tools.u6_data import Standardizer
from cmdvel_tools.u6_models import load_model, predict, recon_errors

FIELDS = ['t', 'n_ventanas', 'err_medio', 'err_ultimo']


class ModelError(Node):
    def __init__(self):
        super().__init__('model_error')
        p = self.declare_parameter
        prefix = p('model', '').value
        if not prefix:
            raise ValueError('falta el parámetro model (prefijo de train_model.py)')
        self.model, self.meta = load_model(prefix)
        self.std = Standardizer.from_dict(self.meta['standardizer'])
        self.counts = 'n_k' in self.meta['channels']
        self.kind, self.T = self.meta['kind'], int(self.meta['T'])
        self.need = self.T if self.kind == 'ae' else self.T + 1
        self.window_sec = float(p('window_sec', 5.0).value)
        self.max_gap = float(p('max_gap_sec', 0.5).value)

        self.last = None                         # (t, v, w) del último mensaje
        self.n_since = 0                         # mensajes desde el tic anterior
        self.grid = deque(maxlen=self.need)      # filas de la grilla, sin estandarizar
        self.errs = deque()                      # (t, error)

        self.pub = self.create_publisher(Float64MultiArray, 'model_metrics', 10)
        self.create_subscription(msg_type(declare_stamped(self)), 'cmd_vel', self.on_msg, 100)
        self.create_timer(1.0 / float(self.meta['rate_hz']), self.grid_tick)
        self.create_timer(1.0 / float(p('publish_hz', 2.0).value), self.publish)
        self.get_logger().info(
            f'model_error: {self.kind.upper()} T={self.T}, canales={self.meta["channels"]}, '
            f'grilla {self.meta["rate_hz"]:g} Hz; data = [' + ', '.join(FIELDS) + ']')

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def on_msg(self, msg):
        t = self.now()
        if t <= 0.0:
            return
        v, w, _ = read_cmd(msg)
        self.last = (t, v, w)
        self.n_since += 1

    def grid_tick(self):
        t = self.now()
        if t <= 0.0 or self.last is None:
            return
        n_k, self.n_since = self.n_since, 0
        if t - self.last[0] > self.max_gap or t < self.last[0]:   # hueco o reloj reiniciado
            self.grid.clear()
            return
        row = [self.last[1], self.last[2]] + ([float(n_k)] if self.counts else [])
        self.grid.append(row)
        if len(self.grid) < self.need:
            return
        W = self.std.transform(np.array(self.grid, dtype=float))[None]   # (1, need, d)
        if self.kind == 'ae':
            e_mean, e_last, _ = recon_errors(self.model, W)
            e = float(e_last[0] if self.meta.get('err', 'mean') == 'last' else e_mean[0])
        else:
            e = float(((predict(self.model, W[:, :-1]) - W[:, -1]) ** 2).mean())
        self.errs.append((t, e))

    def publish(self):
        t = self.now()
        if t <= 0.0:
            return
        while self.errs and self.errs[0][0] <= t - self.window_sec:
            self.errs.popleft()
        nan = float('nan')
        vals = [e for (_, e) in self.errs]
        out = Float64MultiArray()
        out.data = [t, float(len(vals)), float(np.mean(vals)) if vals else nan,
                    vals[-1] if vals else nan]
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ModelError()
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
