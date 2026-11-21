FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ---------------------------
# Utilidades + Python + gedit
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget nano vim \
    gedit \
    python3-pip python3-venv \
    python3-colcon-common-extensions \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Gazebo + integración ROS 2
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gazebo \
    ros-humble-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# TurtleBot3 (simulación)
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-turtlebot3* \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Python deps para análisis (versiones fijas)
# numpy 1.x: ROS 2 Humble está compilado contra numpy 1.x
# --ignore-installed: evita el error al reemplazar
#   paquetes instalados con apt
# ---------------------------
RUN pip3 install --no-cache-dir --ignore-installed \
    "numpy==1.26.4" "pandas==2.2.3" \
    "matplotlib==3.8.4" "python-dateutil==2.9.0.post0"

# ---------------------------
# PyTorch CPU (solo Unidad 6). Para no instalarlo:
#   docker-compose build --build-arg INSTALL_TORCH=0
# ---------------------------
ARG INSTALL_TORCH=1
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
      pip3 install --no-cache-dir --ignore-installed "torch==2.5.1" \
        --index-url https://download.pytorch.org/whl/cpu ; \
    fi

# ---------------------------
# Entrypoint (como root)
# ---------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh

# ---------------------------
# Usuario no-root (didáctico)
# ---------------------------
ARG USERNAME=student
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd -m --uid ${USER_UID} --gid ${USER_GID} -s /bin/bash ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USERNAME}
WORKDIR /home/${USERNAME}

# ---------------------------
# ROS 2 (y el workspace, si existe) en TODA terminal,
# también en las que se abren con "docker exec"
# ---------------------------
RUN echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc \
 && echo '[ -f ~/project/ros2_ws/install/setup.bash ] && source ~/project/ros2_ws/install/setup.bash' >> ~/.bashrc

ENV TURTLEBOT3_MODEL=burger

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
