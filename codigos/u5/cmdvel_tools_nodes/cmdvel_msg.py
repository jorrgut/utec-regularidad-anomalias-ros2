"""
cmdvel_msg: compatibilidad entre geometry_msgs/Twist y geometry_msgs/TwistStamped.

En ROS 2 Humble, /cmd_vel del TurtleBot3 es Twist. Desde Jazzy, los paquetes de
TurtleBot3 usan TwistStamped: el mismo contenido (twist.linear, twist.angular) más un
encabezado con la marca de tiempo del EMISOR (header.stamp).

Todos los nodos del paquete declaran el parámetro booleano "stamped". Su valor por
defecto sale de ROS_DISTRO: falso en Humble (y anteriores), verdadero en Jazzy y
posteriores. Se puede forzar con:  --ros-args -p stamped:=true
"""
import os

from geometry_msgs.msg import Twist, TwistStamped

DISTROS_TWIST = ('foxy', 'galactic', 'humble', 'iron')


def default_stamped():
    """True si la distribución de ROS 2 usa TwistStamped en /cmd_vel."""
    return os.environ.get('ROS_DISTRO', 'humble').lower() not in DISTROS_TWIST


def declare_stamped(node):
    """Declara el parámetro "stamped" y devuelve su valor."""
    return bool(node.declare_parameter('stamped', default_stamped()).value)


def msg_type(stamped):
    return TwistStamped if stamped else Twist


def make_cmd(node, stamped, v=0.0, w=0.0, frame_id=''):
    """Arma el mensaje de comando; si es TwistStamped, lo sella con el reloj del nodo."""
    if stamped:
        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        return msg
    msg = Twist()
    msg.linear.x, msg.angular.z = float(v), float(w)
    return msg


def read_cmd(msg):
    """Devuelve (v, w, t_stamp). t_stamp es la hora del emisor en segundos, o None (Twist)."""
    if hasattr(msg, 'twist'):
        st = msg.header.stamp
        return msg.twist.linear.x, msg.twist.angular.z, st.sec + st.nanosec * 1e-9
    return msg.linear.x, msg.angular.z, None
