from dataclasses import dataclass
import numpy as np

@dataclass
class ControlConfig:
    # -------------------------------------------------
    # Control loop
    # -------------------------------------------------
    control_freq: float = 125.0
    acc: float = 4.0
    max_vel_joint: float = np.pi

    # -------------------------------------------------
    # Task-space gains
    # -------------------------------------------------
    Kp_task: np.ndarray | None = None
    Kd_task: np.ndarray | None = None

    # -------------------------------------------------
    # JSA / TSA
    # -------------------------------------------------
    Kj: float = 50.0

    # -------------------------------------------------
    # Nullspace
    # -------------------------------------------------
    Kp_null: np.ndarray | None = None
    Kd_null: np.ndarray | None = None

    # -------------------------------------------------
    # Admittance
    # -------------------------------------------------
    Ma: np.ndarray | None = None
    Ba: np.ndarray | None = None
    Ka: np.ndarray | None = None

    # -------------------------------------------------
    # Command filtering
    # -------------------------------------------------
    max_velocity_command: float = 0.4
    alpha_filter: float = 0.05
    reset_distance: float = 0.03
    reset_velocity_scale: float = 0.8

    # -------------------------------------------------
    # Trajectory
    # -------------------------------------------------
    amplitude: float = 0.32
    tilt_angle: float = np.pi / 6
    trajectory_offset: np.ndarray = np.array([0, 0, 0], dtype=float)


def create_control_config(
    fitur,
    speed_mode,
    nullspace,
    Kp_arg=None,
    Kd_arg=None,
    Kpn_arg=None,
    kdn_arg=None,
    Kj_arg=None,
):
    config = ControlConfig()

    # =================================================
    # CONTROL LOOP
    # =================================================

    config.control_freq = 125.0
    if fitur == "torque" and speed_mode == "slow":
        config.acc = 4.0
    else:
        config.acc = 4.0
    config.max_vel_joint = np.pi

    # =================================================
    # ADMITTANCE PARAMETERS
    # =================================================

    if fitur == "admittance":
        if speed_mode == "slow":
            config.Ma = np.diag(
                [2.0, 2.0, 2.0]
            )
            config.Ba = np.diag(
                [15.0, 15.0, 15.0]
            )
            config.Ka = np.diag(
                [5.0, 5.0, 5.0]
            )
        elif speed_mode == "normal":
            config.Ma = np.diag(
                [1.2, 1.2, 1.2]
            )
            config.Ba = np.diag(
                [35.0, 35.0, 35.0]
            )
            config.Ka = np.diag(
                [20.0, 20.0, 20.0]
            )
        else:
            config.Ma = np.diag(
                [0.5, 0.5, 0.5]
            )
            config.Ba = np.diag(
                [8.0, 8.0, 8.0]
            )
            config.Ka = np.diag(
                [1.0, 1.0, 1.0]
            )
    else:
        # Preserve hil14 behavior:
        # torque/JSA/TSA masuk ke default branch.
        config.Ma = np.diag(
            [1.0, 1.0, 1.0]
        )
        config.Ba = np.diag(
            [10.0, 10.0, 10.0]
        )
        config.Ka = np.diag(
            [0.0, 0.0, 0.0]
        )

    # =================================================
    # TASK PD GAINS
    # =================================================

    if fitur == "torque":
        if speed_mode == "slow":
            Kp = 100.0
            Kd = 20.0
        elif speed_mode == "normal":
            Kp = 250.0
            Kd = 50.0
        else:
            Kp = 200.0
            Kd = 30.0
    else:
        if speed_mode == "slow":
            Kp = 80.0
            Kd = 15.0
        elif speed_mode == "normal":
            Kp = 300.0
            Kd = 150.0
        else:
            Kp = 160.0
            Kd = 25.0

    config.Kp_task = np.eye(3) * Kp
    config.Kd_task = np.eye(3) * Kd

    if Kp_arg is not None:
        config.Kp_task = np.eye(3) * float(Kp_arg)

    if Kd_arg is not None:
        config.Kd_task = np.eye(3) * float(Kd_arg)

    # =================================================
    # JSA / TSA GAIN
    # =================================================

    config.Kj = (
        Kj_arg
        if Kj_arg is not None
        else 50.0
    )

    # =================================================
    # NULLSPACE GAINS
    # =================================================

    if nullspace == "on":
        config.Kp_null = np.eye(6) * 1.0
        config.Kd_null = np.eye(6) * 0.2
    else:
        config.Kp_null = np.zeros((6, 6))
        config.Kd_null = np.zeros((6, 6))

    # User override
    if Kpn_arg is not None:
        config.Kp_null = np.eye(6) * float(Kpn_arg)

    if kdn_arg is not None:
        config.Kd_null = np.eye(6) * float(kdn_arg)
    # =================================================
    # FILTERING
    # =================================================

    config.max_velocity_command = 0.4
    config.alpha_filter = 0.05
    config.reset_distance = 0.03
    config.reset_velocity_scale = 0.8

    # =================================================
    # TRAJECTORY
    # =================================================

    config.amplitude = 0.32
    config.tilt_angle = np.pi / 6
    config.trajectory_offset = np.array([
        0.53,
        0.08,
        -0.40,
    ], dtype=float)
    return config