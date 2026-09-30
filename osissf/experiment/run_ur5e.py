#!/usr/bin/env python3

from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path

from jax import config
import numpy as np
import pybullet as p

# ==============================================================
# CONFIG
# ==============================================================

from osissf.config import control_config
from osissf.config.experiment_config import (
    parse_experiment_config,
)
from osissf.config.control_config import (
    create_control_config,
)
# ==============================================================
# CORE
# ==============================================================
from osissf.core.dynamics import (
    PinocchioDynamicsCalculator,
)
from osissf.core.controllers import (
    OperationalSpaceController,
    JointSpaceApproach,
    TaskSpaceApproach,
)
from osissf.core.admittance import (
    AdmittanceController,
)
from osissf.core.nullspace import (
    NullspaceController,
)
from osissf.core.safety import (
    SafetyFilter,
    ContainmentVelocityConfig,
    ObstacleVelocityConfig,
    BothVelocityConfig,
)
from oscbf.core.manipulator import Manipulator
from osissf.core.collision import (
    ur5e_collision_data,
)
# ==============================================================
# ROBOT
# ==============================================================
from osissf.robot.ur5e_rtde import (
    UR5eRTDE,
)
# ==============================================================
# SIMULATION
# ==============================================================
from osissf.simulation.pybullet_env import (
    PyBulletEnvironment,
)
from osissf.simulation.digital_twin import (
    DigitalTwin,
)
# ==============================================================
# TRAJECTORY
# ==============================================================
from osissf.trajectory.trajectories import (
    TrajectoryGenerator,
)
# ==============================================================
# LOGGING
# ==============================================================
from osissf.datalog.experiment_log import (
    ExperimentLogger,
)
# ==============================================================
# VISUALIZATION
# ==============================================================
from osissf.visualization.pybullet_visuals import (
    PyBulletVisualizer,
)
from osissf.visualization.plots import (
    ExperimentPlotter,
)
# ==============================================================
# MONITORING
# ==============================================================
from osissf.monitoring.prometheus_metrics import (
    UR5ePrometheusMonitor,
)

class _NullMonitor:
    """Pengganti monitor jika port Prometheus sedang dipakai."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def create_result_directory(
    exp_config,
    control_config,
):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    if exp_config.fitur in ("JSA", "TSA"):
        folder_name = (
            f"results/"
            f"{exp_config.traj}_"
            f"{exp_config.fitur}_"
            f"Kj{control_config.Kj:.1f}_"
            f"Kp{control_config.Kp_task[0, 0]:.0f}_"
            f"Kd{control_config.Kd_task[0, 0]:.0f}_"
            f"Kpn{control_config.Kp_null[0, 0]:.0f}_"
            f"kdn{control_config.Kd_null[0, 0]:.0f}_"
            f"{exp_config.cbf}_"
            f"{timestamp}"
        )

    else:
        folder_name = (
            f"results/"
            f"{exp_config.traj}_"
            f"{exp_config.fitur}_"
            f"{exp_config.speed_mode}_"
            f"{exp_config.cbf}_"
            f"{timestamp}"
        )

    path = Path(folder_name)
    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path, timestamp

def setup_robot(config):
    if config.mode == "sim":
        print("🤖 Mode SIM: RTDE tidak digunakan.")
        return None

    robot = UR5eRTDE(
        config.ip,
    )

    robot.connect()

    return robot

def setup_dynamics(config):
    return PinocchioDynamicsCalculator(
        config.urdf_path,
        config.ee_link_pinocchio,
    )

def setup_simulation(config):
    env = PyBulletEnvironment(
        urdf_path=config.urdf_path,
        obstacle_type=config.obstacle,
        show_obstacle=config.show,
        show_env=(config.show_env == "on"),
        show_actual=(config.show_actual == "on"),
        table_height=config.table_height,
        traj=config.traj,
        cbf=config.cbf,
    )

    env.setup()

    twin = DigitalTwin(
        environment=env,
    )

    return env, twin

def setup_trajectory(exp_config, dt):

    if exp_config.speed_mode == "slow":
        omega = -2.0 * np.pi * 0.05
        move_duration = 10.0

    elif exp_config.speed_mode == "normal":
        omega = -2.0 * np.pi * 0.07
        move_duration = 5.0

    else:  # fast
        omega = -2.0 * np.pi * 0.5
        move_duration = 2.0

    return TrajectoryGenerator(
        trajectory=exp_config.traj,
        amplitude=exp_config.A,
        omega=omega,
        tilt_angle=exp_config.tilt_angle,
        move_duration=move_duration,
        ramp_duration=2.0,
        table_height=0.59,
        trajectory_offset=(
            exp_config.circle_2d_offset
            if exp_config.traj == "circle"
            else exp_config.trajectory_offset
        ),
    )

def setup_controller(
    exp_config,
    control_config,
    dynamics,
    dt,
):
    if exp_config.fitur == "torque":
        if exp_config.nullspace == "on":
            print(
                "⚠️ torque + nullspace on belum diport dari hil14 "
                "(compute_tau_total_with_nullspace). Nullspace diabaikan."
            )
        return OperationalSpaceController(
            dynamics,
            control_config.Kp_task,
            control_config.Kd_task,
            dt=dt,
        )

    if exp_config.fitur == "JSA":
        return JointSpaceApproach(
            dynamics,
            control_config.Kj,
            dt,
        )

    if exp_config.fitur == "TSA":
        return TaskSpaceApproach(
            dynamics,
            control_config.Kj,
            dt,
        )

    return AdmittanceController(
        Ma=control_config.Ma,
        Ba=control_config.Ba,
        Ka=control_config.Ka,
        dt=dt,
        max_velocity=control_config.max_velocity_command,
        alpha_filter=control_config.alpha_filter,
        reset_distance=control_config.reset_distance,
        reset_velocity_scale=control_config.reset_velocity_scale,
    )

def setup_nullspace(exp_config, control_config):
    if exp_config.nullspace == "off":
        return None

    return NullspaceController(
        Kp_null=control_config.Kp_null,
        Kd_null=control_config.Kd_null,
        k_singularity=1.0,
        q_home=exp_config.q_home,
    )

def setup_safety(
    exp_config,
    dynamics,
    env,
):
    if exp_config.cbf == "off":
        return None

    # ==========================================================
    # 1. BUILD CBF ROBOT MODEL
    # ==========================================================

    robot_cbf = Manipulator.from_urdf(
        exp_config.urdf_path,
        ee_offset=np.eye(4),
        collision_data=ur5e_collision_data,
    )

    # ==========================================================
    # 2. CONTAINMENT BOUNDS
    #    Satu sumber data dengan visualisasi PyBullet
    #    (env.containment_params), persis seperti hil14.
    # ==========================================================

    cp = env.containment_params

    pyb_min = np.asarray(cp["xyz_min"], dtype=float)
    pyb_max = np.asarray(cp["xyz_max"], dtype=float)
    wb_pyb_min = np.asarray(cp["wb_xyz_min"], dtype=float)
    wb_pyb_max = np.asarray(cp["wb_xyz_max"], dtype=float)

    robot_base_position = np.asarray(
        env.robot_base_position,
        dtype=float,
    )

    # hil14: z_offset = robot_base_position[2], SAFETY_MARGIN = 0.00
    z_offset = float(robot_base_position[2])
    safety_margin = float(exp_config.safety_margin)

    # ==========================================================
    # 3. PYBULLET -> RTDE
    # ==========================================================

    rtde_min = np.array(
        [
            -pyb_max[0] + safety_margin,
            -pyb_max[1] + safety_margin,
            (pyb_min[2] - z_offset)
            + safety_margin,
        ]
    )

    rtde_max = np.array(
        [
            -pyb_min[0] - safety_margin,
            -pyb_min[1] - safety_margin,
            (pyb_max[2] - z_offset)
            - safety_margin,
        ]
    )

    wb_rtde_min = np.array(
        [
            -wb_pyb_max[0],
            -wb_pyb_max[1],
            wb_pyb_min[2] - z_offset,
        ]
    )

    wb_rtde_max = np.array(
        [
            -wb_pyb_min[0],
            -wb_pyb_min[1],
            wb_pyb_max[2] - z_offset,
        ]
    )

    print(f"📦 PyBullet Box: Min={pyb_min}, Max={pyb_max}")
    print(f"📦 RTDE Box    : Min={rtde_min}, Max={rtde_max}")
    print(f"📦 RTDE WB Box : Min={wb_rtde_min}, Max={wb_rtde_max}")

    # ==========================================================
    # 4. OBSTACLE DATA
    # ==========================================================

    collision_positions = []
    collision_radii = []

    if exp_config.obstacle == "sphere":

        pos_pyb = np.asarray(
            env.obstacle_params["position"],
            dtype=float,
        )

        pos_rtde = (
            pos_pyb - robot_base_position
        ) * np.array(
            [-1.0, -1.0, 1.0]
        )

        radius = (
            float(
                env.obstacle_params["size"][0]
            )
            / 2.0
        )

        radius += float(
            exp_config.obstacle_margin
        )

        collision_positions.append(
            pos_rtde
        )

        collision_radii.append(
            radius
        )

    elif exp_config.obstacle == "tube":

        for tube in env.tube_params:

            for sphere_position in (
                tube["sphere_positions"]
            ):

                pos_pyb = np.asarray(
                    sphere_position,
                    dtype=float,
                )

                pos_rtde = (
                    pos_pyb
                    - robot_base_position
                ) * np.array(
                    [-1.0, -1.0, 1.0]
                )

                radius = (
                    float(
                        env.tube_radius
                    )
                    + float(
                        exp_config.obstacle_margin
                    )
                )

                collision_positions.append(
                    pos_rtde
                )

                collision_radii.append(
                    radius
                )

    # Safety fallback
    if len(collision_positions) == 0:
        collision_positions = [
            np.array(
                [100.0, 100.0, 100.0]
            )
        ]

        collision_radii = [
            0.1
        ]

    collision_positions = np.asarray(
        collision_positions,
        dtype=float,
    )

    collision_radii = np.asarray(
        collision_radii,
        dtype=float,
    )

    # ==========================================================
    # 5. SELECT CBF CONFIGURATION
    # ==========================================================

    if exp_config.cbf == "containment":

        cbf_config = ContainmentVelocityConfig(
            robot=robot_cbf,
            pos_min_rtde=rtde_min,
            pos_max_rtde=rtde_max,
            epsilon_0=exp_config.epsilon_0,
            issf_threshold=exp_config.issf_threshold,
            alpha_c=exp_config.alpha_c,
        )

    elif exp_config.cbf == "obstacle":
        # hil14: whole-body robot-vs-obstacle (BothVelocityConfig)
        # dengan box raksasa ±10 m agar containment tidak pernah aktif.
        big = np.array([10.0, 10.0, 10.0])
        cbf_config = BothVelocityConfig(
            robot=robot_cbf,
            pos_min_rtde=-big,
            pos_max_rtde=big,
            collision_positions=collision_positions,
            collision_radii=collision_radii,
            wb_pos_min_rtde=-big,
            wb_pos_max_rtde=big,
            epsilon_0=exp_config.epsilon_0,
            issf_threshold=exp_config.issf_threshold,
            alpha_c=exp_config.alpha_c,
        )

    elif exp_config.cbf == "both":

        cbf_config = BothVelocityConfig(
            robot=robot_cbf,
            pos_min_rtde=rtde_min,
            pos_max_rtde=rtde_max,
            collision_positions=collision_positions,
            collision_radii=collision_radii,
            wb_pos_min_rtde=wb_rtde_min,
            wb_pos_max_rtde=wb_rtde_max,
            epsilon_0=exp_config.epsilon_0,
            issf_threshold=exp_config.issf_threshold,
            alpha_c=exp_config.alpha_c,
        )

    else:
        raise ValueError(
            f"Unknown CBF mode: {exp_config.cbf}"
        )

    # ==========================================================
    # 6. WRAP STANDARD / ISSf
    # ==========================================================

    safety = SafetyFilter(
        cbf_config,
        use_issf=exp_config.issf,
    )

    # Expose useful information to runner
    # safety.rtde_min = rtde_min
    # safety.rtde_max = rtde_max
    # safety.wb_rtde_min = wb_rtde_min
    # safety.wb_rtde_max = wb_rtde_max
    # safety.robot = robot_cbf
    # safety.collision_positions = collision_positions
    # safety.collision_radii = collision_radii

    return safety

def initialize_state(
    exp_config,
    robot,
    dynamics,
    env,
):
    if exp_config.mode == "real":

        state = robot.get_state()

        q = state.q.copy()
        dq = state.dq.copy()

        x_actual = state.position.copy()
        v_actual = state.velocity.copy()

    else:
        q, dq = env.get_joint_state()
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
    # ------------------------------------------------------
    # Forward kinematics + Jacobian
    # ------------------------------------------------------

    J = dynamics.get_jacobian(q)

    x_actual = (
        dynamics.data
        .oMf[dynamics.ee_frame_id]
        .translation.copy()
    )

    # Pinocchio -> RTDE coordinate convention
    x_actual *= np.array(
        [-1.0, -1.0, 1.0]
    )

    v_actual = J @ dq

    return (
        q,
        dq,
        x_actual,
        v_actual,
    )

def home_robot(exp_config, robot, env, dt, twin=None):
    q_home = np.asarray(
        exp_config.q_home,
        dtype=float,
    )

    print("\n" + "=" * 60)
    print("                 HOMING ROBOT")
    print("=" * 60)
    print(f"q_home = {q_home}")

    if exp_config.mode == "sim":

        kp_home = 1.5
        max_home_velocity = 0.25
        tolerance = 0.005
        velocity_tolerance = 0.01
        max_home_time = 20.0

        start_time = time.time()

        while True:
            q, dq = env.get_joint_state()

            q = np.asarray(q, dtype=float)
            dq = np.asarray(dq, dtype=float)

            q_error = q_home - q

            if (
                np.max(np.abs(q_error)) <= tolerance
                and
                np.max(np.abs(dq)) <= velocity_tolerance
            ):
                break

            if time.time() - start_time > max_home_time:
                raise RuntimeError(
                    "Homing simulation timeout."
                )

            dq_cmd = kp_home * q_error

            dq_cmd = np.clip(
                dq_cmd,
                -max_home_velocity,
                max_home_velocity,
            )

            env.apply_joint_velocity(
                dq_cmd,
                max_force=150.0,
            )

            env.step()

            time.sleep(dt)

        # Stop motor setelah sampai home
        env.apply_joint_velocity(
            np.zeros(6),
            max_force=150.0,
        )

        q_check, dq_check = env.get_joint_state()

        q_check = np.asarray(q_check, dtype=float)
        dq_check = np.asarray(dq_check, dtype=float)

        print("✅ Homing selesai.")
        print(f"   q_actual = {np.round(q_check, 5)}")
        print(f"   error    = {np.round(q_home - q_check, 5)}")

        return q_check, dq_check

    # ======================================================
    # REAL ROBOT
    # ======================================================

    if not robot.is_connected():
        raise RuntimeError(
            "Robot belum terhubung."
        )

    # ------------------------------------------------------
    # Homing menggunakan speedJ sebagai pre-control loop.
    # Ini dijalankan SEBELUM experimental control loop.
    # ------------------------------------------------------
    tolerance = 0.005
    robot.move_j(
        q_home,
        speed=0.5,
        acceleration=0.5,
        asynchronous=True,
    )

    start_time = time.time()

    while True:

        state = robot.get_state()

        q = np.asarray(
            state.q,
            dtype=float,
        )

        dq = np.asarray(
            state.dq,
            dtype=float,
        )

        # hil14 move_and_sync: PyBullet mengikuti UR5e (100 Hz)
        if twin is not None:
            twin.follow(q, dq)

        q_error = q_home - q

        if (
            np.max(np.abs(q_error))
            <= 0.005
            and
            np.max(np.abs(dq))
            <= 0.01
        ):
            break

        if (
            time.time() - start_time
            > 20.0
        ):
            robot.stop_j()

            raise RuntimeError(
                "Homing timeout. "
                "Robot gagal mencapai q_home."
            )

        time.sleep(0.01)

    robot.stop_j()

    time.sleep(0.2)

    # ------------------------------------------------------
    # VERIFICATION
    # ------------------------------------------------------

    state = robot.get_state()

    q_final = np.asarray(
        state.q,
        dtype=float,
    )

    dq_final = np.asarray(
        state.dq,
        dtype=float,
    )

    home_error = np.max(
        np.abs(
            q_final - q_home
        )
    )

    print(
        "q_actual =",
        np.round(q_final, 4),
    )

    print(
        f"max |q-q_home| = "
        f"{home_error:.6f} rad"
    )

    if home_error > tolerance:

        raise RuntimeError(
            "Homing selesai tetapi "
            "posisi robot belum memenuhi "
            "toleransi q_home."
        )

    # hil14: sinkronisasi final setelah homing (velocity = 0)
    if twin is not None:
        twin.sync_from_robot(q_final, np.zeros(6))

    print("✅ Robot berhasil homing ke q_home.")

def move_to_start_point(
    exp_config,
    robot,
    dynamics,
    env,
    trajectory,
    x_base,
    dt,
    twin=None,
):
    """
    Memindahkan EE dari posisi setelah homing
    menuju titik awal trajectory x_des(0).

    Real:
        - Hitung numerical IK untuk x_start_traj
        - Gerakkan robot menggunakan speedJ
        - Verifikasi posisi aktual

    Sim:
        - Gerakkan robot secara dinamis menggunakan
          Cartesian velocity -> joint velocity
        - Verifikasi posisi aktual
    """

    print(
        f"\n📍 Bergerak ke Titik Awal Lintasan "
        f"({exp_config.traj})..."
    )

    # ==========================================================
    # 1. HITUNG TITIK AWAL TRAJECTORY
    # ==========================================================

    if exp_config.traj == "wp":
        x_start_traj = np.array(
            [-0.300, 0.000, 0.500],
            dtype=float,
        )
    else:
        x_start_traj, _, _ = trajectory.compute(
            0.0,
            x_base,
        )

    x_start_traj = np.asarray(
        x_start_traj,
        dtype=float,
    )

    print(
        f"   -> Target posisi awal: "
        f"{np.round(x_start_traj, 6)}"
    )

    # ==========================================================
    # 2. REAL ROBOT
    # ==========================================================

    if exp_config.mode == "real":

        state = robot.get_state()

        q = np.asarray(
            state.q,
            dtype=float,
        )

        dq = np.asarray(
            state.dq,
            dtype=float,
        )

        # ------------------------------------------------------
        # Numerical IK
        # ------------------------------------------------------

        print(
            "   -> Mencari IK untuk posisi awal..."
        )

        q_start = q.copy()

        ik_tolerance = 1e-4
        ik_max_iterations = 500
        ik_gain = 2.0
        ik_damping = 1e-4
        ik_step = 0.02

        ik_success = False

        for _ in range(ik_max_iterations):

            J = dynamics.get_jacobian(
                q_start
            )

            x_ik = (
                dynamics.data
                .oMf[
                    dynamics.ee_frame_id
                ]
                .translation.copy()
            )

            # Pinocchio -> RTDE
            x_ik *= np.array(
                [-1.0, -1.0, 1.0]
            )

            error = (
                x_start_traj
                - x_ik
            )

            if np.linalg.norm(error) <= ik_tolerance:
                ik_success = True
                break

            # Damped Least Squares
            JJt = (
                J @ J.T
                + ik_damping
                * np.eye(3)
            )

            dq_ik = (
                J.T
                @ np.linalg.solve(
                    JJt,
                    ik_gain * error,
                )
            )

            dq_ik = np.clip(
                dq_ik,
                -0.5,
                0.5,
            )

            q_start = (
                q_start
                + dq_ik * ik_step
            )

            # Respect joint limits
            q_start = np.clip(
                q_start,
                dynamics.model.lowerPositionLimit,
                dynamics.model.upperPositionLimit,
            )

        if not ik_success:
            raise RuntimeError(
                "Numerical IK gagal mencapai "
                "titik awal trajectory."
            )

        print(
            f"   -> IK berhasil: "
            f"q_start = "
            f"{np.round(q_start, 6)}"
        )

        # ------------------------------------------------------
        # Move to q_start
        # ------------------------------------------------------

        print(
            "   -> Bergerak ke posisi awal menggunakan moveJ..."
        )

        robot.move_j(
            q_start,
            speed=0.2,
            acceleration=0.4,
            asynchronous=True,
        )

        start_time = time.time()

        while True:

            state = robot.get_state()

            q = np.asarray(
                state.q,
                dtype=float,
            )

            dq = np.asarray(
                state.dq,
                dtype=float,
            )

            q_error = q_start - q

            # hil14 move_and_sync: teleport PyBullet ke UR5e
            if twin is not None:
                twin.follow(q, dq)

            if (
                np.max(np.abs(q_error))
                <= 0.01
            ):
                break

            if (
                time.time() - start_time
                > 30.0
            ):
                robot.stop_j()

                raise RuntimeError(
                    "Timeout saat bergerak ke titik awal trajectory."
                )

            time.sleep(0.01)

        robot.stop_j()

        time.sleep(0.2)

        # ------------------------------------------------------
        # Final state
        # ------------------------------------------------------

        state = robot.get_state()

        q = np.asarray(
            state.q,
            dtype=float,
        )

        dq = np.asarray(
            state.dq,
            dtype=float,
        )

        J = dynamics.get_jacobian(q)

        x_actual = (
            dynamics.data
            .oMf[
                dynamics.ee_frame_id
            ]
            .translation.copy()
        )

        x_actual *= np.array(
            [-1.0, -1.0, 1.0]
        )

        v_actual = J @ dq

        # hil14: posisi EE memakai TCP pose RTDE, bukan FK Pinocchio
        x_actual = np.asarray(state.position, dtype=float).copy()

        if twin is not None:
            twin.sync_from_robot(q, dq)

    # ==========================================================
    # 3. SIMULATION
    # ==========================================================

    else:

        print(
            "   -> Bergerak ke posisi awal "
            "secara dinamis di PyBullet..."
        )

        position_tolerance = 0.002
        velocity_tolerance = 0.01
        max_cartesian_velocity = 0.15
        kp_move = 1.5
        max_move_time = 20.0

        start_time = time.time()

        while True:

            q, dq = (
                env.get_joint_state()
            )

            q = np.asarray(
                q,
                dtype=float,
            )

            dq = np.asarray(
                dq,
                dtype=float,
            )

            J = dynamics.get_jacobian(
                q
            )

            x_actual = (
                dynamics.data
                .oMf[
                    dynamics.ee_frame_id
                ]
                .translation.copy()
            )

            x_actual *= np.array(
                [-1.0, -1.0, 1.0]
            )

            v_actual = J @ dq

            error = (
                x_start_traj
                - x_actual
            )

            if (
                np.linalg.norm(error)
                <= position_tolerance
                and
                np.max(
                    np.abs(dq)
                )
                <= velocity_tolerance
            ):
                break

            if (
                time.time()
                - start_time
                > max_move_time
            ):
                env.apply_joint_velocity(
                    np.zeros(6),
                    max_force=150.0,
                )

                raise RuntimeError(
                    "Timeout saat simulasi "
                    "bergerak ke titik awal trajectory."
                )

            # Cartesian velocity command
            v_cmd = (
                kp_move * error
            )

            v_cmd = np.clip(
                v_cmd,
                -max_cartesian_velocity,
                max_cartesian_velocity,
            )

            # Cartesian -> joint velocity
            dq_cmd = (
                np.linalg.pinv(J)
                @ v_cmd
            )

            dq_cmd = np.clip(
                dq_cmd,
                -0.25,
                0.25,
            )

            env.apply_joint_velocity(
                dq_cmd,
                max_force=150.0,
            )

            env.step()

            time.sleep(dt)

        # Stop motor setelah sampai
        env.apply_joint_velocity(
            np.zeros(6),
            max_force=150.0,
        )

        time.sleep(0.2)

        # Final state
        q, dq = (
            env.get_joint_state()
        )

        q = np.asarray(
            q,
            dtype=float,
        )

        dq = np.asarray(
            dq,
            dtype=float,
        )

        J = dynamics.get_jacobian(q)

        x_actual = (
            dynamics.data
            .oMf[
                dynamics.ee_frame_id
            ].translation.copy()
        )

        x_actual *= np.array(
            [-1.0, -1.0, 1.0]
        )

        v_actual = J @ dq

    # ==========================================================
    # 4. DEBUG
    # ==========================================================

    print(
        "\n🐞 DEBUG INFO - Setelah "
        "Bergerak ke Titik Awal Lintasan:"
    )

    print(
        f"   Posisi Target Awal Lintasan "
        f"(x_start_traj): "
        f"{np.round(x_start_traj, 3)}"
    )

    print(
        f"   Posisi Aktual EE Robot "
        f"(x_ee_pose): "
        f"{np.round(x_actual, 3)}"
    )

    print(
        f"   Error Awal "
        f"(x_start_traj - x_ee_pose): "
        f"{np.round(x_start_traj - x_actual, 3)}"
    )

    return (
        x_start_traj,
        q,
        dq,
        x_actual,
        v_actual,
    )

# ==============================================================
# HIL14-FAITHFUL HELPERS
# ==============================================================

HIL14_PRE_CBF_DISTURBANCE = True   # hil14 menambahkan d(t) 2x pada admittance
FORCE_DEADBAND = 2.0               # N, hil14
VERBOSE_DEBUG = False              # blok debug lama (sigma_min, cond(J))


def _disturbance(t):
    d_max = 0.05
    noise = np.random.uniform(-0.02, 0.02, size=6)
    d_t = np.zeros(6)
    d_t[0] = (d_max * 0.8) * np.sin(2.3 * t) + noise[0]
    d_t[1] = (d_max * 0.8) * np.cos(1.7 * t) + noise[1]
    d_t[2] = (d_max * 0.8) * np.sin(3.1 * t) + noise[2]
    d_t[3] = (d_max * 0.4) * np.cos(2.9 * t) + noise[3]
    d_t[4] = (d_max * 0.4) * np.sin(1.3 * t) + noise[4]
    d_t[5] = (d_max * 0.4) * np.cos(3.7 * t) + noise[5]
    return np.clip(d_t, -d_max, d_max)


def _limit_position_error(raw_e_pos, exp_config):
    """hil14: batasi tarikan error posisi."""
    if exp_config.traj == "wp":
        max_pull = 0.07
        n = np.linalg.norm(raw_e_pos)
        if n > max_pull:
            return raw_e_pos * (max_pull / n)
        return raw_e_pos
    if exp_config.cbf != "off":
        return np.clip(raw_e_pos, -0.1, 0.1)
    return raw_e_pos


def _read_feedback(exp_config, robot, dynamics, twin):
    """
    Returns q, dq, x_actual, v_tcp (TCP speed dari robot).
    Real: x_actual = TCP pose RTDE (hil14).
    Sim : FK Pinocchio pada state PyBullet.
    """
    if exp_config.mode == "real":
        state = robot.get_state()
        return (
            state.q.copy(),
            state.dq.copy(),
            state.position.copy(),
            state.velocity.copy(),
        )

    q, dq = twin.get_state()
    q = np.asarray(q, dtype=float)
    dq = np.asarray(dq, dtype=float)
    J = dynamics.get_jacobian(q)
    x_actual = (
        dynamics.data.oMf[dynamics.ee_frame_id].translation.copy()
        * np.array([-1.0, -1.0, 1.0])
    )
    return q, dq, x_actual, J @ dq


def _print_cbf_intervention(dq_cmd, dq_cmd_to_send, h_vals, x_actual):
    diff_norm = np.linalg.norm(dq_cmd - dq_cmd_to_send)
    if diff_norm <= 0.05:
        return

    all_radii = np.concatenate(ur5e_collision_data["radii"])
    n_wb_cols = 6 * len(all_radii)

    min_containment = np.min(h_vals[:6]) if len(h_vals) >= 6 else float("inf")
    if len(h_vals) > 6 + n_wb_cols:
        min_obstacle = np.min(h_vals[6:-n_wb_cols])
    else:
        min_obstacle = float("inf")
    min_wb = np.min(h_vals[-n_wb_cols:]) if len(h_vals) > n_wb_cols else float("inf")

    if min_obstacle < 0.1 and min_obstacle <= min_containment and min_obstacle <= min_wb:
        print(f"   🛡️ [CBF ACTIVE] Menghindari Obstacle! (Intervensi dq: {diff_norm:.4f}, h_min: {min_obstacle:.3f})")
    elif min_wb < 0.1 and min_wb <= min_containment:
        print(f"   🔵 [CBF ACTIVE] Sikut Tertahan Whole-Body! (Intervensi dq: {diff_norm:.4f}, h_min: {min_wb:.3f})")
    elif min_containment < 0.1:
        print(f"   🚧 [CBF ACTIVE] EE Tertahan Containment Box! (Intervensi dq: {diff_norm:.4f}, h_min: {min_containment:.3f})")
    else:
        print(f"   ⚙️ [CBF ACTIVE] Penyesuaian Trajektori. (Intervensi dq: {diff_norm:.4f})")


def _control_loop_body(
    exp_config,
    control_config,
    dt,
    robot,
    dynamics,
    env,
    twin,
    trajectory,
    x_base,
    controller,
    nullspace,
    safety,
    visualizer,
    logger,
    monitor,
):
    is_real = exp_config.mode == "real"
    alpha_f = control_config.alpha_filter
    v_clip = control_config.max_velocity_command

    # ==========================================================
    # 0. INITIAL STATE + SINKRONISASI (hil14 bagian F)
    # ==========================================================
    q, dq, x_actual, v_tcp = _read_feedback(exp_config, robot, dynamics, twin)
    q_cmd = q.copy()

    if isinstance(controller, AdmittanceController):
        # pa = x_ee_pose (TCP aktual), pa_dot = 0
        controller.reset(x_actual)
        print(f"⚖️ Admittance disinkronkan ke EE aktual: {np.round(x_actual, 4)}")

    if is_real:
        twin.sync_from_robot(q, dq)
        twin.reset_prediction()

    dq_cmd_prev = np.zeros(6)
    dx_cmd_prev = np.zeros(3)

    runtime = exp_config.runtime
    if exp_config.traj == "wp":
        _, _, cycle_time = trajectory.get_wp_timing()
        runtime = cycle_time + 0.1
    elif exp_config.traj == "circle_3d":
        runtime = 17.0
    if exp_config.loop:
        runtime = float("inf")

    t_traj = 0.0
    is_cbf_blocking = False
    last_printed_t = -1.0

    if is_real:
        robot.enable_watchdog(min_frequency=10.0)

    print("\n🎮 Memulai Loop Kontrol (Tekan Ctrl+C untuk berhenti)...")
    start_time = time.time()
    loop_start = start_time
    last_print_wall_time = start_time

    while time.time() - start_time < runtime:
        t_wall = time.time() - start_time
        t = t_traj if exp_config.mode == "sim" else t_wall

        if is_real:
            robot.kick_watchdog()

        # ======================================================
        # 1. FEEDBACK
        # ======================================================
        q, dq, x_actual, v_tcp = _read_feedback(
            exp_config, robot, dynamics, twin
        )

        # ======================================================
        # 2. TARGET
        # ======================================================
        if exp_config.traj == "mouse":
            x_des = env.get_mouse_target_rtde()
            if x_des is None:
                x_des = x_actual.copy()
            xd_des = np.zeros(3)
            xdd_des = np.zeros(3)
        else:
            t_eval = t
            if exp_config.traj == "circle_3d" and exp_config.pertimbangan:
                t_eval = t_traj
            x_des, xd_des, xdd_des = trajectory.compute(t_eval, x_base)
            if exp_config.traj == "circle_3d" and exp_config.pertimbangan and is_cbf_blocking:
                xd_des = np.zeros(3)
                xdd_des = np.zeros(3)

        # ======================================================
        # 3. DYNAMICS
        # ======================================================
        J = dynamics.get_jacobian(q)
        M = dynamics.get_mass_matrix(q)
        try:
            Lambda = np.linalg.inv(J @ np.linalg.inv(M) @ J.T)
        except np.linalg.LinAlgError:
            Lambda = np.eye(3)  # hil14 fallback

        # hil14: --speed calc -> v = J dq ; act -> TCP speed
        v_actual = J @ dq if exp_config.speed == "calc" else v_tcp

        # External force (hil14: sensor F/T + deadband 2 N)
        if is_real:
            F_raw = robot.get_actual_force()
            F_ee = np.where(np.abs(F_raw) < FORCE_DEADBAND, 0.0, F_raw)
        else:
            F_ee = np.zeros(6)
        F_external = F_ee[:3]

        if VERBOSE_DEBUG:
            sv = np.linalg.svd(J, compute_uv=False)
            print(f"[dbg t={t:.3f}] sigma_min={sv[-1]:.3e} cond={sv[0]/max(sv[-1],1e-12):.2e}")

        # ======================================================
        # 4. NOMINAL CONTROLLER
        # ======================================================
        v_cmd = np.zeros(3)
        q_ddot = np.zeros(6)
        rhs = np.zeros(3)
        pa_ddot = np.zeros(3)
        pa_dot = np.zeros(3)
        dq_null = np.zeros(6)
        d_pre = np.zeros(6)
        tau_null = np.zeros(6)

        raw_e_pos = x_des - x_actual
        e_pos = _limit_position_error(raw_e_pos, exp_config)
        e_vel = xd_des - v_actual

        if exp_config.fitur == "admittance":
            F_cmd = Lambda @ (
                xdd_des
                + control_config.Kd_task @ e_vel
                + control_config.Kp_task @ e_pos
            )
            res = controller.compute(
                F_cmd=F_cmd,
                F_external=F_external,
                x_des=x_des,
                x_actual=x_actual,
            )
            v_cmd = res["v_cmd"]
            pa_dot = res["v_cmd"].copy()   # hil14: pa_dot = v_cmd (filtered)
            pa_ddot = res["pa_ddot"]
            rhs = res["rhs"]

            J_pinv = np.linalg.pinv(J)
            dq_cmd_task = J_pinv @ v_cmd

            if nullspace is not None:
                dq_null = nullspace.compute(q=q, dq=dq, J=J, t=t)
            dq_cmd = dq_cmd_task + dq_null

            # hil14: disturbance pertama (sebelum CBF) pada admittance
            if (
                HIL14_PRE_CBF_DISTURBANCE
                and exp_config.dist == "on"
                and exp_config.ip == "127.0.0.1"
            ):
                d_pre = _disturbance(t)
                dq_cmd = dq_cmd + d_pre

            # torque hanya untuk logging (hil14)
            tau_task = J.T @ (
                control_config.Kp_task @ raw_e_pos
                + control_config.Kd_task @ e_vel
            )
            tau_total = tau_task.copy()

        elif exp_config.fitur == "JSA":
            res = controller.compute(
                q, dq, x_actual, v_actual, x_des, xd_des, xdd_des,
                control_config.Kp_task, control_config.Kd_task,
            )
            dq_cmd_task = res["dq_cmd"]
            tau_task = res["tau_task"]
            tau_total = tau_task.copy()
            v_cmd = J @ dq_cmd_task
            if nullspace is not None:
                dq_null = nullspace.compute(q=q, dq=dq, J=J, t=t)
            dq_cmd = dq_cmd_task + dq_null
            dq_cmd_prev = dq_cmd.copy()   # hil14 (sebelum filter)

        elif exp_config.fitur == "TSA":
            res = controller.compute(
                q, dq, x_actual, v_actual, x_des, xd_des, xdd_des,
                control_config.Kp_task, control_config.Kd_task,
                dx_cmd_prev,
            )
            dq_cmd_task = res["dq_cmd"]
            dx_cmd_prev = res["dx_curr"].copy()
            v_cmd = dx_cmd_prev.copy()
            tau_task = J.T @ res["F_cmd"]
            tau_total = tau_task.copy()
            if nullspace is not None:
                dq_null = nullspace.compute(q=q, dq=dq, J=J, t=t)
            dq_cmd = dq_cmd_task + dq_null
            dq_cmd_prev = dq_cmd.copy()   # hil14 (sebelum filter)

        else:  # torque (nullspace off, sesuai hil14 cabang else)
            res = controller.compute(q, dq, x_actual, v_actual, x_des, xd_des)
            dq_cmd_task = res["dq_cmd"]
            dq_cmd = dq_cmd_task
            q_ddot = res["q_ddot"]
            tau_task = res["tau_task"]
            tau_total = res["tau_total"]

        # hil14: clip + low-pass untuk non-admittance
        if exp_config.fitur != "admittance":
            dq_cmd = np.clip(dq_cmd, -v_clip, v_clip)
            dq_cmd = (1.0 - alpha_f) * dq_cmd_prev + alpha_f * dq_cmd
            dq_cmd_prev = dq_cmd.copy()

        # ======================================================
        # 5. CBF SAFETY FILTER
        # ======================================================
        dq_cmd_nominal = dq_cmd.copy()
        dq_cmd_to_send = dq_cmd.copy()
        dq_cmd_safe = None
        h_values = np.zeros(6)
        cbf_time_ms = 0.0
        is_cbf_blocking = False

        if safety is not None:
            h_values = np.asarray(safety.h(q))
            t0 = time.perf_counter()
            dq_cmd_safe = np.asarray(safety.filter(q, dq_cmd))

            # hil14: sinkron admittance saat dekat Z-min (circle_3d)
            if (
                exp_config.traj == "circle_3d"
                and isinstance(controller, AdmittanceController)
                and exp_config.cbf in ("containment", "both")
                and h_values[5] < 0.10
            ):
                v_safe_z = (J @ dq_cmd_safe)[2]
                controller.pa_dot[2] = v_safe_z
                controller.pa[2] = x_actual[2]
                pa_dot = controller.pa_dot.copy()

            cbf_time_ms = (time.perf_counter() - t0) * 1000.0
            dq_cmd_to_send = dq_cmd_safe.copy()

            if exp_config.traj == "circle_3d" and exp_config.pertimbangan:
                if np.linalg.norm(dq_cmd - dq_cmd_to_send) > 0.01:
                    is_cbf_blocking = True

        if not is_cbf_blocking:
            t_traj += dt

        # ======================================================
        # 6. DISTURBANCE (setelah CBF)
        # ======================================================
        d_t = np.zeros(6)
        if exp_config.dist == "on" and exp_config.ip == "127.0.0.1":
            d_t = _disturbance(t)
            dq_cmd_to_send = dq_cmd_to_send + d_t

        # ======================================================
        # 7. COMMAND
        # ======================================================
        dq_cmd_clipped = np.clip(
            dq_cmd_to_send,
            -control_config.max_vel_joint,
            control_config.max_vel_joint,
        )

        if is_real and robot.is_connected():
            if exp_config.command == "speedJ":
                robot.speed_j(dq_cmd_clipped, control_config.acc, dt)
            elif exp_config.command == "servoL":
                servol_cmd = np.zeros(6)
                servol_cmd[:3] = v_cmd
                robot.servo_l(servol_cmd, 0.5, 0.1)

        # ======================================================
        # 8. DIGITAL TWIN
        # ======================================================
        if is_real:
            tw = twin.hil_step(q, dq, dq_cmd_clipped)
            q_sim = tw["q_sync"]
            dq_sim = tw["dq_sync"]
            q_pred_now = tw["q_pred_now"]
            twin_residual = tw["residual"]
        else:
            twin.command_velocity(dq_cmd_clipped)
            twin.step()
            q_sim, dq_sim = twin.get_state()
            q_sim = np.asarray(q_sim, dtype=float)
            dq_sim = np.asarray(dq_sim, dtype=float)
            q_pred_now = q_sim.copy()
            twin_residual = np.zeros(6)

        dynamics.get_jacobian(q_sim)
        x_pybullet_ee = (
            dynamics.data.oMf[dynamics.ee_frame_id].translation.copy()
            * np.array([-1.0, -1.0, 1.0])
        )

        # ======================================================
        # 9. VISUALIZATION
        # ======================================================
        visualizer.update_target(env.target_sphere_id, x_des)
        if env.ee_actual_sphere_id is not None:
            visualizer.update_actual_ee(env.ee_actual_sphere_id, x_actual)
        if env.robot_sphere_ids and safety is not None:
            visualizer.update_collision_spheres(
                env.robot_sphere_ids,
                np.asarray(safety.robot.link_collision_data(q)),
            )

        # ======================================================
        # 10. MONITORING
        # ======================================================
        monitor.update_joint_state(q_actual=q, dq_actual=dq, tau_total=tau_total)
        monitor.update_cartesian(
            x_actual=x_actual,
            x_desired=x_des,
            error_x=x_des[0] - x_actual[0],
            v_target_x=xd_des[0],
            v_cmd_x=v_cmd[0],
        )
        if safety is not None:
            monitor.update_safety(
                safety_status=bool(np.any(h_values < 0)),
                obstacle_status=False,
                whole_body_status=False,
                dq_safe=dq_cmd_to_send[0],
                h_values=h_values,
            )
        monitor.update_digital_twin(q_sim=q_sim, dq_sim=dq_sim)

        # ======================================================
        # 11. LOGGING
        # ======================================================
        logger.log_step(
            t=t,
            x_act=x_actual,
            x_des=x_des,
            tau_task=tau_task,
            q_ddot=q_ddot,
            rhs=rhs,
            pa_ddot=pa_ddot,
            q_dot=dq_cmd,
            pa_dot=pa_dot,
            q_act=q,
            q_des=q_cmd,
            dq_act=dq,
            dq_task=dq_cmd_task,
            dq_null=dq_null,
            dq_des=dq_cmd_to_send,          # hil14: perintah yang dikirim
            dq_cmd_safe=(
                dq_cmd_safe if dq_cmd_safe is not None else np.array([])
            ),
            dq_cmd=dq_cmd_nominal,
            dq_cmd_to_send=dq_cmd_clipped,
            tau_total=tau_total,
            tau_task_only=tau_task,
            tau_null=tau_null,
            tau_coriolis_grav=dynamics.get_coriolis_gravity(q, dq),
            q_null=(
                nullspace.last_q_error if nullspace is not None else np.zeros(6)
            ),
            v_tcp_speed=v_tcp,
            v_tcp_calc=v_actual,
            q_pybullet=q_sim,
            q_ur5e=q,
            x_pybullet_ee=x_pybullet_ee,
            q_pybullet_pred=q_pred_now,
            twin_residual=twin_residual,
            manipulability=float(
                np.sqrt(max(np.linalg.det(J @ J.T), 1e-12))
            ),
            h_values=h_values,
            disturbance=d_t + d_pre,
            cbf_time=cbf_time_ms,
        )

        # ======================================================
        # 12. PRINTING (jadwal hil14)
        #   admittance: tiap step s/d t <= 0.30 s, lalu tiap 0.30 s
        #   lainnya   : tiap 0.30 s (wall clock)
        # ======================================================
        now = time.time()
        should_print = False
        if exp_config.fitur == "admittance" and t <= 0.30 + dt / 2.0:
            if last_printed_t < 0 or (t - last_printed_t >= dt - 1e-6):
                should_print = True
        elif now - last_print_wall_time >= 0.3 - 1e-6:
            should_print = True

        if should_print:
            e_print = x_des - x_actual
            msg = (
                f"[t={t:.2f}] ErrX: {e_print[0]:.4f} "
                f"| dqSafe: {dq_cmd_to_send[0]:.4f} "
                f"| vTargetX: {xd_des[0]:.4f}"
            )
            if exp_config.fitur == "admittance":
                msg += f" | vCmdX: {v_cmd[0]:.4f}"
            else:
                msg += f" | TskTrq0: {tau_task[0]:.2f} | qDotCmd0: {dq_cmd[0]:.4f}"

            gap = np.abs(twin_residual)
            msg += (
                f" || SimGap J0: {gap[0]:.5f} | J1: {gap[1]:.5f}"
                f" | max: {gap.max():.5f} rad"
            )
            print(msg)

            if (
                exp_config.traj == "circle_3d"
                and exp_config.cbf in ("containment", "both")
                and exp_config.fitur == "admittance"
                and np.any(h_values[:6] < 0.05)
            ):
                print(
                    f"[t={t:.2f}] h_Z_min={h_values[5]:.4f}, "
                    f"v_Z={v_actual[2]:.4f}, pa_dot_Z={pa_dot[2]:.4f}"
                )

            if safety is not None:
                _print_cbf_intervention(
                    dq_cmd_nominal, dq_cmd_to_send, h_values, x_actual
                )

            last_printed_t = t
            last_print_wall_time = time.time()

        # ======================================================
        # 13. LOOP TIMING
        # ======================================================
        sleep_time = (loop_start + dt) - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        loop_start += dt


def run_control_loop(**kwargs):
    """
    Wrapper keselamatan: APA PUN cara loop berakhir (runtime habis,
    Ctrl+C, exception), robot di-speedStop SEGERA di sini, sebelum
    main() menyimpan log dan membuat plot (yang bisa makan waktu lama).
    """
    exp_config = kwargs["exp_config"]
    robot = kwargs["robot"]
    try:
        _control_loop_body(**kwargs)
    finally:
        if exp_config.mode == "real" and robot is not None:
            print("\n🛑 Loop selesai -> speedStop robot...")
            robot.safe_stop()


marker_ids = [-1, -1, -1]
marker_size = 0.025  # 2.5 cm

def main():

    exp_config = parse_experiment_config()

    robot = None
    env = None
    logger = None
    plotter = None
    control_config = None
    result_dir = None
    timestamp = None
    rtde_min = rtde_max = None
    wb_rtde_min = wb_rtde_max = None

    try:

        print(
            "\n=================================================="
        )
        print(
            "          UR5e OSCBF / ISSf EXPERIMENT"
        )
        print(
            "=================================================="
        )

        print(f"Mode       : {exp_config.mode}")
        print(f"Trajectory : {exp_config.traj}")
        print(f"Control    : {exp_config.fitur}")
        print(f"CBF        : {exp_config.cbf}")
        print(f"ISSf       : {exp_config.issf}")

        # ------------------------------------------------------
        # Configuration
        # ------------------------------------------------------

        control_config = create_control_config(
            fitur=exp_config.fitur,
            speed_mode=exp_config.speed_mode,
            nullspace=exp_config.nullspace,
            Kp_arg=exp_config.Kp,
            Kd_arg=exp_config.Kd,
            Kpn_arg=exp_config.Kpn,
            kdn_arg=exp_config.kdn,
            Kj_arg=exp_config.Kj,
        )

        if exp_config.traj == "circle_3d":
            control_config.acc = 6.0

        dt = 1.0 / control_config.control_freq

        result_dir, timestamp = create_result_directory(
            exp_config,
            control_config,
        )

        # ------------------------------------------------------
        # Robot
        # ------------------------------------------------------

        robot = setup_robot(
            exp_config
        )

        # ------------------------------------------------------
        # Dynamics
        # ------------------------------------------------------

        dynamics = setup_dynamics(
            exp_config
        )

        # ------------------------------------------------------
        # Simulation / Digital Twin
        # ------------------------------------------------------

        env, twin = setup_simulation(
            exp_config
        )

        # ------------------------------------------------------
        # Homing
        # ------------------------------------------------------

        home_robot(
            exp_config=exp_config,
            robot=robot,
            env=env,
            dt=dt,
            twin=twin,
        )

        # ------------------------------------------------------
        # Ambil state setelah homing
        # ------------------------------------------------------

        if exp_config.mode == "real":
            state = robot.get_state()
            q = np.asarray(
                state.q,
                dtype=float,
            )

            dq = np.asarray(
                state.dq,
                dtype=float,
            )

            x_actual = np.asarray(
                state.position,
                dtype=float,
            )

            v_actual = np.asarray(
                state.velocity,
                dtype=float,
            )

        else:

            q, dq = env.get_joint_state()
            q = np.asarray(
                q,
                dtype=float,
            )

            dq = np.asarray(
                dq,
                dtype=float,
            )

            J = dynamics.get_jacobian(q)

            x_actual = (
                dynamics.data
                .oMf[dynamics.ee_frame_id]
                .translation.copy()
            )

            # Pinocchio -> RTDE
            x_actual *= np.array(
                [-1.0, -1.0, 1.0]
            )

            v_actual = J @ dq

        print(
            f"📍 EE setelah homing: "
            f"{np.round(x_actual, 6)}"
        )

        # ------------------------------------------------------
        # Trajectory
        # ------------------------------------------------------

        trajectory = setup_trajectory(
            exp_config,
            dt,
        )

        # Pusat trajectory ditentukan oleh trajectory_offset
        x_base = np.zeros(3)

        # ------------------------------------------------------
        # Move to Start Point
        # ------------------------------------------------------

        (
            x_start_traj,
            q,
            dq,
            x_actual,
            v_actual,
        ) = move_to_start_point(
            exp_config=exp_config,
            robot=robot,
            dynamics=dynamics,
            env=env,
            trajectory=trajectory,
            x_base=x_base,
            dt=dt,
            twin=twin,
        )

        if exp_config.setup_only:
            print("\n✅ --setup-only: homing & move ke titik awal selesai. Keluar.")
            return

        # ------------------------------------------------------
        # Verifikasi titik awal trajectory
        # ------------------------------------------------------

        x_des, xd_des, xdd_des = trajectory.compute(
            0.0,
            x_base,
        )

        print(
            f"\n🎯 Verifikasi x_des(0): "
            f"{np.round(x_des, 6)}"
        )

        print(
            f"📏 Error terhadap EE aktual: "
            f"{np.round(x_des - x_actual, 6)}"
        )
        # ============================================================
        # DEBUG MARKER: x_des -> PyBullet frame
        # ============================================================
        x_des_pyb = (
            env.robot_base_position
            + x_des * np.array([-1.0, -1.0, 1.0])
        )

        marker_ids[0] = p.addUserDebugLine(
            x_des_pyb + np.array([-marker_size, 0.0, 0.0]),
            x_des_pyb + np.array([ marker_size, 0.0, 0.0]),
            lineColorRGB=[1, 0, 0],
            lineWidth=4,
            replaceItemUniqueId=marker_ids[0],
        )

        marker_ids[1] = p.addUserDebugLine(
            x_des_pyb + np.array([0.0, -marker_size, 0.0]),
            x_des_pyb + np.array([0.0,  marker_size, 0.0]),
            lineColorRGB=[1, 0, 0],
            lineWidth=4,
            replaceItemUniqueId=marker_ids[1],
        )

        marker_ids[2] = p.addUserDebugLine(
            x_des_pyb + np.array([0.0, 0.0, -marker_size]),
            x_des_pyb + np.array([0.0, 0.0,  marker_size]),
            lineColorRGB=[1, 0, 0],
            lineWidth=4,
            replaceItemUniqueId=marker_ids[2],
        )

        print(
            f"🎯 x_des(0): "
            f"{np.round(x_des, 6)}"
        )

        print(
            f"📏 error awal: "
            f"{np.round(x_des - x_actual, 6)}"
        )

        x_actual_pyb = (
            env.robot_base_position
            + x_actual * np.array([-1.0, -1.0, 1.0])
        )
        # ------------------------------------------------------
        # Visualization
        # ------------------------------------------------------

        visualizer = PyBulletVisualizer(
            robot_base_position=env.robot_base_position
        )

        preview_duration = exp_config.runtime

        if exp_config.traj == "circle_3d":
            preview_duration = 17.0

        elif exp_config.traj == "wp":
            _, _, cycle_time = trajectory.get_wp_timing()
            preview_duration = cycle_time

        visualizer.draw_trajectory_preview(
            trajectory=trajectory,
            trajectory_name=exp_config.traj,
            x_base=x_base,
            duration=preview_duration,
            dt=dt,
            sample_skip=20,
        )

        # ------------------------------------------------------
        # Controllers
        # ------------------------------------------------------

        controller = setup_controller(
            exp_config,
            control_config,
            dynamics,
            dt,
        )

        nullspace = setup_nullspace(
            exp_config,
            control_config,
        )

        # ------------------------------------------------------
        # Safety
        # ------------------------------------------------------

        safety = setup_safety(
            exp_config,
            dynamics,
            env,
        )

        if safety is not None:
            rtde_min = getattr(safety, "rtde_min", None)
            rtde_max = getattr(safety, "rtde_max", None)
            wb_rtde_min = getattr(safety, "wb_rtde_min", None)
            wb_rtde_max = getattr(safety, "wb_rtde_max", None)
        else:
            rtde_min = None
            rtde_max = None
            wb_rtde_min = None
            wb_rtde_max = None

        # ------------------------------------------------------
        # Logging
        # ------------------------------------------------------

        logger = ExperimentLogger(
            save_dir=result_dir
        )
        print(
            "LOGGER:",
            logger,
            type(logger),
        )
        logger.set_metadata(
            command=exp_config.command,
            config={
                **vars(exp_config),
                **vars(control_config),
                "dt": dt,
            },
            rtde_min=rtde_min,
            rtde_max=rtde_max,
            wb_rtde_min=wb_rtde_min,
            wb_rtde_max=wb_rtde_max,
        )

        # logger.save_pickle(
        #     f"HIL_log_{timestamp}.pkl"
        # )

        # ------------------------------------------------------
        # Prometheus
        # ------------------------------------------------------

        try:
            monitor = UR5ePrometheusMonitor(
                port=8000
            )
        except OSError as exc:
            print(
                f"⚠️ Prometheus tidak bisa dibuka di port 8000 ({exc}). "
                "Eksperimen tetap jalan TANPA monitoring."
            )
            monitor = _NullMonitor()

        # # ==========================================================
        # # 13. VISUALIZER
        # # ==========================================================

        # visualizer = PyBulletVisualizer(
        #     robot_base_position=env.robot_base_position
        # )

        # ------------------------------------------------------
        # Plotter
        # ------------------------------------------------------

        plotter = ExperimentPlotter(
            output_dir=result_dir,
            trajectory=exp_config.traj,
            control_mode=exp_config.fitur,
            speed_mode=exp_config.speed_mode,
            cbf_mode=exp_config.cbf,
            timestamp=timestamp,
        )

        # ------------------------------------------------------
        # Run experiment
        # ------------------------------------------------------

        # ------------------------------------------------------
        # hil14: WARM-UP JAX JIT (sebelum start_time!)
        # ------------------------------------------------------
        if exp_config.mode == "real":
            q_warm = robot.get_q()
        else:
            q_warm, _ = env.get_joint_state()
        if safety is not None:
            safety.warmup(q_warm)
            # link_collision_data dipakai tiap siklus untuk bola visual
            np.asarray(safety.robot.link_collision_data(q_warm))

        # ------------------------------------------------------
        # hil14: stabilisasi 1.0 s + sinkronisasi ulang twin
        # ------------------------------------------------------
        print("\n⏳ Menunggu stabilisasi (1.0s)...")
        time.sleep(1.0)
        if exp_config.mode == "real":
            state = robot.get_state()
            twin.sync_from_robot(state.q, state.dq)
            print(
                f"   -> Posisi EE aktual setelah stabil: "
                f"{np.round(state.position, 4)}"
            )

        run_control_loop(
            exp_config=exp_config,
            control_config=control_config,
            dt=dt,
            robot=robot,
            dynamics=dynamics,
            env=env,
            twin=twin,
            trajectory=trajectory,
            x_base=x_base,
            controller=controller,
            nullspace=nullspace,
            safety=safety,
            visualizer=visualizer,
            logger=logger,
            monitor=monitor,
        )

    except KeyboardInterrupt:

        print(
            "\n⚠️ Experiment dihentikan "
            "dengan Ctrl+C."
        )

    except Exception as exc:

        print(
            f"\n❌ Fatal error: {exc}"
        )

        traceback.print_exc()

    finally:

        # ------------------------------------------------------
        # Stop robot DULU (sebelum simpan log / plotting)
        # ------------------------------------------------------
        if robot is not None:
            try:
                robot.safe_stop()
            except Exception:
                pass

        # ------------------------------------------------------
        # Save results
        # ------------------------------------------------------

        try:
            if logger is not None:

                log_data = logger.finalize()

                logger.save_pickle(
                    f"HIL_log_{timestamp}.pkl"
                )

                if plotter is not None:
                    plotter.plot_all(
                        log_data,
                        rtde_min=rtde_min,
                        rtde_max=rtde_max,
                        wb_rtde_min=wb_rtde_min,
                        wb_rtde_max=wb_rtde_max,
                        robot_base_position=getattr(
                            env,
                            "robot_base_position",
                            None,
                        ),
                        obstacle_params=getattr(
                            env,
                            "obstacle_params",
                            None,
                        ),
                        tube_params=getattr(
                            env,
                            "tube_params",
                            None,
                        ),
                        tube_radius=getattr(
                            env,
                            "tube_radius",
                            None,
                        ),
                        epsilon_0=getattr(
                            exp_config,
                            "epsilon_0",
                            0.0,
                        ),
                        alpha_c=getattr(
                            exp_config,
                            "alpha_c",
                            0.0,
                        ),
                        waypoint_events=getattr(
                            exp_config,
                            "waypoint_events",
                            None,
                        ),
                        waypoint_positions=getattr(
                            exp_config,
                            "waypoint_positions",
                            None,
                        ),
                    )

        except Exception:
            print(
                "⚠️ Gagal menyimpan hasil "
                "atau membuat plot."
            )
            traceback.print_exc()

        # ------------------------------------------------------
        # Robot shutdown
        # ------------------------------------------------------

        if robot is not None:
            try:
                robot.disconnect()   # speedStop -> stopScript -> disconnect
            except Exception:
                pass

        # ------------------------------------------------------
        # PyBullet shutdown
        # ------------------------------------------------------

        try:
            if env is not None:
                env.close()
        except Exception:
            pass

        print(
            "\n✅ Experiment selesai."
        )


if __name__ == "__main__":
    main()