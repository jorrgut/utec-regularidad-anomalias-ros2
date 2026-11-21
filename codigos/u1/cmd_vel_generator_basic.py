#!/usr/bin/env python3
"""
Generador automático de comandos /cmd_vel (Unidad 1).

Publica geometry_msgs/Twist a frecuencia constante siguiendo un patrón cíclico:
    1. avanzar   t_fwd  segundos a v_fwd  (m/s)
    2. detenerse t_stop segundos
    3. girar     t_turn segundos a w_turn (rad/s)
    4. detenerse t_stop segundos
    (repetir)

La fase del ciclo se calcula a partir del reloj del nodo (ROS_TIME), de modo que
respeta use_sim_time: si se pausa Gazebo, el patrón también se pausa.

Uso típico:
    python3 cmd_vel_generator_basic.py --use-sim-time --rate 10 \
        --v-fwd 0.20 --w-turn 0.80 --t-fwd 6.0 --t-turn 3.0 --t-stop 2.0

Opcional: --duration 300 detiene el generador tras 300 s (del reloj del nodo).
Al terminar (Ctrl+C o --duration) publica un Twist nulo para detener el robot.
"""
import argparse
import os

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


class CmdVelGenerator(Node):
    def __init__(self, args):
        super().__init__(
            'cmd_vel_generator_basic',
            parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, args.use_sim_time)
            ])
        self.args = args
        self.stamped = args.stamped
        self.pub = self.create_publisher(TwistStamped if self.stamped else Twist, args.topic, 10)

        # Fases del ciclo: (nombre, duración, v, w)
        self.phases = [
            ('avanzar', args.t_fwd, args.v_fwd, 0.0),
            ('detener', args.t_stop, 0.0, 0.0),
            ('girar', args.t_turn, 0.0, args.w_turn),
            ('detener', args.t_stop, 0.0, 0.0),
        ]
        self.cycle = sum(p[1] for p in self.phases)
        if self.cycle <= 0:
            raise ValueError('La duración total del ciclo debe ser > 0')

        self.t0 = None
        self.n_pub = 0
        self.last_phase = None
        self.done = False

        # El temporizador usa el reloj del nodo: con use_sim_time=true
        # se dispara según /clock.
        self.timer = self.create_timer(1.0 / args.rate, self.on_timer)

        self.get_logger().info(
            f'Generador iniciado: rate={args.rate} Hz, ciclo={self.cycle:.1f} s, '
            f'use_sim_time={args.use_sim_time}, tópico={args.topic}, '
            f'tipo={"TwistStamped" if self.stamped else "Twist"}')
        if args.use_sim_time:
            self.get_logger().info(
                'Esperando /clock... (si no avanza, verificá que Gazebo esté corriendo)')

    def make_msg(self, v, w):
        if self.stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()   # hora del emisor
            msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
            return msg
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        return msg

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def current_command(self, elapsed):
        tau = elapsed % self.cycle
        acc = 0.0
        for name, dur, v, w in self.phases:
            acc += dur
            if tau < acc:
                return name, v, w
        name, _, v, w = self.phases[-1]
        return name, v, w

    def on_timer(self):
        t = self.now_sec()
        if self.args.use_sim_time and t == 0.0:
            return  # todavía no llegó ningún /clock
        if self.t0 is None:
            self.t0 = t
        elapsed = t - self.t0

        if self.args.duration is not None and elapsed >= self.args.duration:
            self.done = True
            return

        name, v, w = self.current_command(elapsed)
        self.pub.publish(self.make_msg(v, w))
        self.n_pub += 1

        if name != self.last_phase:
            n_cycle = int(elapsed // self.cycle) + 1
            self.get_logger().info(
                f't={elapsed:7.2f} s | ciclo {n_cycle} | fase: {name} (v={v:.2f}, w={w:.2f})')
            self.last_phase = name

    def stop_robot(self):
        # Varios mensajes nulos por si alguno se pierde
        for _ in range(3):
            self.pub.publish(self.make_msg(0.0, 0.0))
        self.get_logger().info(f'Publicados {self.n_pub} comandos. Robot detenido.')


def main():
    p = argparse.ArgumentParser(description='Generador cíclico de /cmd_vel')
    p.add_argument('--use-sim-time', action='store_true',
                   help='usar tiempo de simulación (/clock)')
    p.add_argument('--rate', type=float, default=10.0, help='frecuencia nominal (Hz)')
    p.add_argument('--v-fwd', type=float, default=0.20, help='velocidad lineal al avanzar (m/s)')
    p.add_argument('--w-turn', type=float, default=0.80, help='velocidad angular al girar (rad/s)')
    p.add_argument('--t-fwd', type=float, default=6.0, help='duración de avance (s)')
    p.add_argument('--t-turn', type=float, default=3.0, help='duración de giro (s)')
    p.add_argument('--t-stop', type=float, default=2.0, help='duración de cada detención (s)')
    p.add_argument('--duration', type=float, default=None,
                   help='duración total (s); por defecto corre hasta Ctrl+C')
    p.add_argument('--topic', default='/cmd_vel', help='tópico de salida')
    p.add_argument('--stamped', action=argparse.BooleanOptionalAction, default=None,
                   help='publicar TwistStamped (por defecto: según ROS_DISTRO; Jazzy en adelante)')
    args, _ = p.parse_known_args()
    if args.stamped is None:
        args.stamped = default_stamped()

    if args.rate <= 0:
        p.error('--rate debe ser > 0')
    if abs(args.v_fwd) > 0.22:
        print('[aviso] TurtleBot3 burger tiene v máx ~0.22 m/s; el controlador puede saturar.')
    if abs(args.w_turn) > 2.84:
        print('[aviso] TurtleBot3 burger tiene w máx ~2.84 rad/s; el controlador puede saturar.')

    # Manejamos Ctrl+C nosotros, así el contexto sigue vivo para
    # guardar/publicar antes de cerrar.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = CmdVelGenerator(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
