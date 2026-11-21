#!/usr/bin/env python3
"""
cmd_vel_generator: generador de /cmd_vel con modos (Unidad 5).

Modos (parámetro 'mode'):
  structured      el ciclo de la Unidad 1: avanzar t_fwd, detener t_stop, girar t_turn,
                  detener t_stop (nominal)
  shuffle_blocks  bloques de block_len mensajes, cada uno con una fase elegida al azar
                  (avanzar / detener / girar): mismos comandos, sin la estructura temporal
  iid             cada mensaje con v y w uniformes al azar en [0, v_fwd] y [0, w_turn]

Uso:
  ros2 run cmdvel_tools cmd_vel_generator --ros-args -p use_sim_time:=true -p mode:=structured
  ros2 run cmdvel_tools cmd_vel_generator --ros-args -p use_sim_time:=true \
      -p mode:=shuffle_blocks -p block_len:=30
"""
import random

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from cmdvel_tools.cmdvel_msg import declare_stamped, make_cmd, msg_type


class CmdVelGenerator(Node):
    def __init__(self):
        super().__init__('cmd_vel_generator')
        p = self.declare_parameter
        self.mode = p('mode', 'structured').value
        self.rate = float(p('rate_hz', 10.0).value)
        self.v_fwd = float(p('v_fwd', 0.20).value)
        self.w_turn = float(p('w_turn', 0.80).value)
        self.t_fwd = float(p('t_fwd', 6.0).value)
        self.t_turn = float(p('t_turn', 3.0).value)
        self.t_stop = float(p('t_stop', 2.0).value)
        self.block_len = int(p('block_len', 30).value)
        self.duration = float(p('duration', 0.0).value)   # 0 = sin límite
        self.rng = random.Random(int(p('seed', 0).value))
        if self.mode not in ('structured', 'shuffle_blocks', 'iid'):
            raise ValueError(f'modo desconocido: {self.mode}')

        self.stamped = declare_stamped(self)
        self.pub = self.create_publisher(msg_type(self.stamped), 'cmd_vel', 10)
        self.phases = [('avanzar', self.t_fwd, self.v_fwd, 0.0),
                       ('detener', self.t_stop, 0.0, 0.0),
                       ('girar', self.t_turn, 0.0, self.w_turn),
                       ('detener', self.t_stop, 0.0, 0.0)]
        self.cycle = sum(ph[1] for ph in self.phases)
        self.t0 = None
        self.count = 0
        self.block_cmd = (0.0, 0.0)
        self.done = False
        self.timer = self.create_timer(1.0 / self.rate, self.tick)
        self.get_logger().info(f'Generador: modo={self.mode}, {self.rate:g} Hz')

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def command(self, elapsed):
        if self.mode == 'structured':
            tau = elapsed % self.cycle
            acc = 0.0
            for _, dur, v, w in self.phases:
                acc += dur
                if tau < acc:
                    return v, w
            return 0.0, 0.0
        if self.mode == 'shuffle_blocks':
            if self.count % self.block_len == 0:
                _, _, v, w = self.rng.choice(self.phases)
                self.block_cmd = (v, w)
            return self.block_cmd
        return self.rng.uniform(0, self.v_fwd), self.rng.uniform(0, self.w_turn)

    def tick(self):
        t = self.now()
        if t == 0.0:          # con use_sim_time, todavía no llegó /clock
            return
        if self.t0 is None:
            self.t0 = t
        elapsed = t - self.t0
        if self.duration > 0 and elapsed >= self.duration:
            self.done = True
            return
        v, w = self.command(elapsed)
        self.pub.publish(make_cmd(self, self.stamped, v, w))
        self.count += 1

    def stop(self):
        for _ in range(3):
            self.pub.publish(make_cmd(self, self.stamped))


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = CmdVelGenerator()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
