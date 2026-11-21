#!/usr/bin/env python3
"""
Logger de /cmd_vel a CSV (Unidad 1).

Registra, para cada mensaje geometry_msgs/Twist (o TwistStamped) recibido:
    t_sec : tiempo de RECEPCIÓN según el reloj del nodo (ROS_TIME), en segundos
    v     : velocidad lineal  (linear.x,  m/s)
    w     : velocidad angular (angular.z, rad/s)
    dt    : intervalo respecto del mensaje anterior (s); vacío en la primera fila
    t_stamp : (solo con TwistStamped) hora de EMISIÓN, header.stamp, en segundos.
              t_sec - t_stamp es el retraso de cada mensaje (Unidad 1, epsilon + L).

Uso típico:
    python3 cmd_vel_logger.py --use-sim-time --out data/cmdvel_u1.csv --duration 300

La duración se mide con el reloj del nodo desde el PRIMER mensaje recibido.
El CSV se guarda al alcanzar la duración o al interrumpir con Ctrl+C.
"""
import argparse
import csv
import os
import statistics

import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist, TwistStamped


# Humble usa geometry_msgs/Twist en /cmd_vel; desde Jazzy, TurtleBot3 usa TwistStamped
# (mismo contenido más header.stamp, la hora del EMISOR). Por defecto se elige según
# ROS_DISTRO; se puede forzar con --stamped o --no-stamped.
def default_stamped():
    return os.environ.get('ROS_DISTRO', 'humble').lower() not in ('foxy', 'galactic', 'humble', 'iron')


class CmdVelLogger(Node):
    def __init__(self, output_file, duration, use_sim_time, topic, stamped=False):
        super().__init__(
            'cmd_vel_logger',
            parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, use_sim_time)
            ])
        self.buffer = []
        self.output_file = output_file
        self.duration = duration
        self.use_sim_time = use_sim_time
        self.start_time = None
        self.done = False
        self.stamped = stamped

        self.subscription = self.create_subscription(
            TwistStamped if stamped else Twist, topic, self.callback, 10)

        # Verifica la duración aunque dejen de llegar mensajes
        self.timer = self.create_timer(0.5, self.check_duration)

        self.get_logger().info(
            f'Logger iniciado. tópico={topic} ({"TwistStamped" if stamped else "Twist"}), '
            f'use_sim_time={use_sim_time}, '
            f'duración={duration} s, salida={output_file}')

    def now_sec(self):
        # ROS_TIME: respeta use_sim_time
        return self.get_clock().now().nanoseconds / 1e9

    def callback(self, msg):
        t_sec = self.now_sec()          # tiempo de RECEPCIÓN
        if self.use_sim_time and t_sec == 0.0:
            # Aún no llegó /clock: el tiempo no es válido
            self.get_logger().warn(
                'Mensaje descartado: /clock todavía no publicó tiempo',
                throttle_duration_sec=5.0)
            return
        if self.start_time is None:
            self.start_time = t_sec
            self.get_logger().info('Primer mensaje recibido: comienza el registro')
        if self.stamped:
            st = msg.header.stamp
            self.buffer.append((t_sec, msg.twist.linear.x, msg.twist.angular.z,
                                st.sec + st.nanosec * 1e-9))
        else:
            self.buffer.append((t_sec, msg.linear.x, msg.angular.z))

    def check_duration(self):
        if self.start_time is None:
            return
        if self.now_sec() - self.start_time >= self.duration:
            self.done = True

    def save(self):
        folder = os.path.dirname(self.output_file)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t_sec', 'v', 'w', 'dt'] + (['t_stamp'] if self.stamped else []))
            prev = None
            for row in self.buffer:
                t, v, w = row[:3]
                dt = '' if prev is None else f'{t - prev:.9f}'
                out = [f'{t:.9f}', v, w, dt]
                if self.stamped:
                    out.append(f'{row[3]:.9f}')
                writer.writerow(out)
                prev = t

        n = len(self.buffer)
        msg = f'Guardadas {n} muestras en {self.output_file}'
        if n >= 2:
            dts = [b[0] - a[0] for a, b in zip(self.buffer, self.buffer[1:])]
            dur = self.buffer[-1][0] - self.buffer[0][0]
            med = statistics.median(dts)
            freq = 1.0 / med if med > 0 else float('nan')
            msg += f' | duración {dur:.2f} s | freq. efectiva {freq:.2f} Hz'
        self.get_logger().info(msg)


def main():
    parser = argparse.ArgumentParser(description='Registra /cmd_vel en CSV')
    parser.add_argument('--out', default='data/cmdvel_u1.csv', help='archivo CSV de salida')
    parser.add_argument('--duration', type=float, default=60.0, help='duración del registro (s)')
    parser.add_argument('--use-sim-time', action='store_true', help='usar tiempo de simulación')
    parser.add_argument('--topic', default='/cmd_vel', help='tópico a registrar')
    parser.add_argument('--stamped', action=argparse.BooleanOptionalAction, default=None,
                        help='esperar TwistStamped (por defecto: según ROS_DISTRO; Jazzy en adelante)')
    args, _ = parser.parse_known_args()
    if args.stamped is None:
        args.stamped = default_stamped()

    # Manejamos Ctrl+C nosotros, así el contexto sigue vivo para
    # guardar antes de cerrar.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = CmdVelLogger(args.out, args.duration, args.use_sim_time, args.topic, args.stamped)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info('Interrumpido por el usuario')
    finally:
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
