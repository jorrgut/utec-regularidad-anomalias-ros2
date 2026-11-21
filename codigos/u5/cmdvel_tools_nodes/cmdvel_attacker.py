#!/usr/bin/env python3
"""
cmdvel_attacker: inyecta anomalías en /cmd_vel durante un intervalo (Unidad 5).

  flood   publica rate_hz mensajes extra por segundo, con el ÚLTIMO comando legítimo
          (valores coherentes: solo cambia el ritmo de llegada)
  replay  reenvía, intercalados con el tráfico legítimo, los comandos que se publicaron
          replay_shift segundos antes (los graba antes de empezar el ataque)

El ataque empieza start_after segundos después de que arranca el nodo y dura 'duration'.
Imprime el intervalo y, si events_file no está vacío, lo AGREGA a ese archivo en el mismo
reloj que usan los CSV (sim time), para pasárselo a evaluate_detector.py.

AVISO: este nodo existe solo para el laboratorio, sobre el robot simulado.

Uso:
  ros2 run cmdvel_tools cmdvel_attacker --ros-args -p use_sim_time:=true \
      -p mode:=flood -p start_after:=60.0 -p duration:=20.0 \
      -p events_file:=outputs_u5/run1/events.txt
"""
from collections import deque

import rclpy
from rclpy.node import Node
from cmdvel_tools.cmdvel_msg import declare_stamped, make_cmd, msg_type, read_cmd


class CmdVelAttacker(Node):
    def __init__(self):
        super().__init__('cmdvel_attacker')
        p = self.declare_parameter
        self.mode = p('mode', 'flood').value
        self.start_after = float(p('start_after', 60.0).value)
        self.duration = float(p('duration', 20.0).value)
        self.rate = float(p('rate_hz', 30.0).value)
        self.shift = float(p('replay_shift', 6.5).value)
        self.events_file = p('events_file', '').value
        if self.mode not in ('flood', 'replay'):
            raise ValueError(f'modo desconocido: {self.mode}')
        if self.mode == 'replay':
            self.rate = float(p('replay_rate_hz', 10.0).value)

        self.stamped = declare_stamped(self)
        self.pub = self.create_publisher(msg_type(self.stamped), 'cmd_vel', 10)
        self.sub = self.create_subscription(msg_type(self.stamped), 'cmd_vel', self.on_msg, 50)
        self.history = deque()          # (t, v, w) legítimos, grabados ANTES del ataque
        self.last_legit = (0.0, 0.0)    # (v, w) del último comando legítimo
        self.t_node0 = None
        self.t_start = self.t_end = None
        self.state = 'esperando'
        self.timer = self.create_timer(1.0 / self.rate, self.tick)
        self.get_logger().info(f'Atacante: modo={self.mode}, empieza {self.start_after:g} s '
                               f'después de arrancar, dura {self.duration:g} s')

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def on_msg(self, msg):
        # Solo se graba antes del ataque: durante el ataque, este nodo también recibe
        # sus propios mensajes y no se pueden distinguir de los legítimos.
        if self.state != 'esperando':
            return
        t = self.now()
        v, w, _ = read_cmd(msg)
        self.last_legit = (v, w)
        self.history.append((t, v, w))
        while self.history and self.history[0][0] < t - (self.shift + self.duration + 5):
            self.history.popleft()

    def tick(self):
        t = self.now()
        if t == 0.0:
            return
        if self.t_node0 is None:
            self.t_node0 = t
        if self.state == 'esperando' and t - self.t_node0 >= self.start_after:
            self.state, self.t_start = 'atacando', t
            self.t_end = t + self.duration
            self.get_logger().warn(f'ATAQUE {self.mode} desde t = {t:.2f} s')
        if self.state != 'atacando':
            return
        if t >= self.t_end:
            self.state = 'terminado'
            self.get_logger().warn(f'Fin del ataque en t = {t:.2f} s | evento: '
                                   f'{self.t_start:.2f} {self.t_end:.2f}')
            if self.events_file:
                with open(self.events_file, 'a', encoding='utf-8') as f:
                    f.write(f'{self.t_start:.2f} {self.t_end:.2f}   # {self.mode}\n')
            return
        if self.mode == 'flood':
            # el atacante vuelve a sellar cada mensaje con su propio reloj (TwistStamped)
            self.pub.publish(make_cmd(self, self.stamped, *self.last_legit))
            return
        target = t - self.shift
        old = min(self.history, key=lambda r: abs(r[0] - target), default=None)
        if old is not None:
            self.pub.publish(make_cmd(self, self.stamped, old[1], old[2]))


def main():
    rclpy.init()
    node = CmdVelAttacker()
    try:
        while rclpy.ok() and node.state != 'terminado':
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
