from __future__ import annotations

from typing import Iterable, Mapping, Optional
import numpy as np

from prometheus_client import Gauge, CollectorRegistry, start_http_server


class UR5ePrometheusMonitor:
    """
    Real-time Prometheus exporter for UR5e experiments.

    Keeps monitoring concerns separate from:
    - robot communication
    - controller
    - CBF
    - simulation
    - experiment logging
    """

    def __init__(
        self,
        port: int = 8000,
        start_server: bool = True,
    ):
        self.port = int(port)

        # Custom registry prevents accidental pollution of the
        # default Prometheus registry.
        self.registry = CollectorRegistry()

        # ---------------------------------------------------------
        # Joint state
        # ---------------------------------------------------------
        self.q_actual = Gauge(
            "ur5e_q_actual",
            "Actual joint position (rad)",
            ["joint_id"],
            registry=self.registry,
        )

        self.dq_actual = Gauge(
            "ur5e_dq_actual",
            "Actual joint velocity (rad/s)",
            ["joint_id"],
            registry=self.registry,
        )

        self.tau_total = Gauge(
            "ur5e_tau_total",
            "Total commanded torque (Nm)",
            ["joint_id"],
            registry=self.registry,
        )

        # ---------------------------------------------------------
        # Cartesian tracking
        # ---------------------------------------------------------
        self.pos_actual = Gauge(
            "ur5e_pos_actual",
            "Actual EE position (m)",
            ["axis"],
            registry=self.registry,
        )

        self.pos_desired = Gauge(
            "ur5e_pos_desired",
            "Desired EE position (m)",
            ["axis"],
            registry=self.registry,
        )

        self.error_x = Gauge(
            "ur5e_error_x",
            "EE tracking error on X axis (m)",
            registry=self.registry,
        )

        self.v_target_x = Gauge(
            "ur5e_v_target_x",
            "Target/reference X value",
            registry=self.registry,
        )

        self.v_cmd_x = Gauge(
            "ur5e_v_cmd_x",
            "Commanded X value",
            registry=self.registry,
        )

        # ---------------------------------------------------------
        # CBF
        # ---------------------------------------------------------
        self.dq_safe = Gauge(
            "ur5e_dq_safe",
            "First joint value of output command after safety filter",
            registry=self.registry,
        )

        self.h_values = Gauge(
            "ur5e_h_values",
            "CBF constraint value",
            ["constraint"],
            registry=self.registry,
        )

        # ---------------------------------------------------------
        # Digital Twin
        # ---------------------------------------------------------
        self.q_sim = Gauge(
            "ur5e_q_sim",
            "PyBullet joint position (rad)",
            ["joint_id"],
            registry=self.registry,
        )

        self.dq_sim = Gauge(
            "ur5e_dq_sim",
            "PyBullet joint velocity (rad/s)",
            ["joint_id"],
            registry=self.registry,
        )

        # ---------------------------------------------------------
        # Safety status
        # ---------------------------------------------------------
        self.safety_status = Gauge(
            "ur5e_safety_status",
            "Overall safety status: 0=SAFE, 1=DANGER",
            registry=self.registry,
        )

        self.obs_status = Gauge(
            "ur5e_obs_status",
            "Obstacle avoidance status: 1=ACTIVE",
            registry=self.registry,
        )

        self.wb_status = Gauge(
            "ur5e_wb_status",
            "Whole-body collision status: 1=ACTIVE",
            registry=self.registry,
        )

        # ---------------------------------------------------------
        # Obstacles
        # ---------------------------------------------------------
        self.obs_position = Gauge(
            "ur5e_obs_position",
            "Obstacle center position (m)",
            ["obs_name", "axis"],
            registry=self.registry,
        )

        self.obs_radius = Gauge(
            "ur5e_obs_radius",
            "Obstacle radius (m)",
            ["obs_name"],
            registry=self.registry,
        )

        if start_server:
            self.start()

    def start(self):
        start_http_server(
            self.port,
            registry=self.registry,
        )
        print(
            f"Prometheus metrics server running on port {self.port}"
        )

    # -----------------------------------------------------------------
    # Joint metrics
    # -----------------------------------------------------------------

    def update_joint_state(
        self,
        q_actual: Iterable[float],
        dq_actual: Iterable[float],
        tau_total: Optional[Iterable[float]] = None,
    ):
        q_actual = np.asarray(q_actual, dtype=float)
        dq_actual = np.asarray(dq_actual, dtype=float)

        for i in range(6):
            joint = f"joint_{i + 1}"

            self.q_actual.labels(joint_id=joint).set(q_actual[i])
            self.dq_actual.labels(joint_id=joint).set(dq_actual[i])

            if tau_total is not None:
                tau = np.asarray(tau_total, dtype=float)
                self.tau_total.labels(joint_id=joint).set(tau[i])

    # -----------------------------------------------------------------
    # Cartesian metrics
    # -----------------------------------------------------------------

    def update_cartesian(
        self,
        x_actual: Iterable[float],
        x_desired: Iterable[float],
        error_x: Optional[float] = None,
        v_target_x: Optional[float] = None,
        v_cmd_x: Optional[float] = None,
    ):
        x_actual = np.asarray(x_actual, dtype=float)
        x_desired = np.asarray(x_desired, dtype=float)

        axes = ("x", "y", "z")

        for i, axis in enumerate(axes):
            self.pos_actual.labels(axis=axis).set(x_actual[i])
            self.pos_desired.labels(axis=axis).set(x_desired[i])

        if error_x is not None:
            self.error_x.set(float(error_x))

        if v_target_x is not None:
            self.v_target_x.set(float(v_target_x))

        if v_cmd_x is not None:
            self.v_cmd_x.set(float(v_cmd_x))

    # -----------------------------------------------------------------
    # Safety metrics
    # -----------------------------------------------------------------

    def update_safety(
        self,
        safety_status: bool,
        obstacle_status: bool,
        whole_body_status: bool,
        dq_safe: Optional[float] = None,
        h_values: Optional[Iterable[float]] = None,
    ):
        self.safety_status.set(1 if safety_status else 0)
        self.obs_status.set(1 if obstacle_status else 0)
        self.wb_status.set(1 if whole_body_status else 0)

        if dq_safe is not None:
            self.dq_safe.set(float(dq_safe))

        if h_values is not None:
            h_values = np.asarray(h_values, dtype=float)

            names = [
                "max_x",
                "max_y",
                "max_z",
                "min_x",
                "min_y",
                "min_z",
            ]

            for i, name in enumerate(names[: len(h_values)]):
                self.h_values.labels(
                    constraint=name
                ).set(h_values[i])

    # -----------------------------------------------------------------
    # Digital Twin
    # -----------------------------------------------------------------

    def update_digital_twin(
        self,
        q_sim: Iterable[float],
        dq_sim: Iterable[float],
    ):
        q_sim = np.asarray(q_sim, dtype=float)
        dq_sim = np.asarray(dq_sim, dtype=float)

        for i in range(6):
            joint = f"joint_{i + 1}"

            self.q_sim.labels(joint_id=joint).set(q_sim[i])
            self.dq_sim.labels(joint_id=joint).set(dq_sim[i])

    # -----------------------------------------------------------------
    # Obstacles
    # -----------------------------------------------------------------

    def update_obstacles(
        self,
        obstacles: Mapping[str, Mapping[str, object]],
    ):
        """
        Example:
        {
            "tube_purple": {
                "position": [x, y, z],
                "radius": 0.05,
            }
        }
        """

        for obs_name, obs in obstacles.items():
            position = np.asarray(obs["position"], dtype=float)
            radius = float(obs["radius"])

            for i, axis in enumerate(("x", "y", "z")):
                self.obs_position.labels(
                    obs_name=obs_name,
                    axis=axis,
                ).set(position[i])

            self.obs_radius.labels(
                obs_name=obs_name
            ).set(radius)