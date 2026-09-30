from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from osissf.datalog.metrics import (
    compute_tracking_metrics,
    compute_cbf_metrics,
    compute_digital_twin_metrics,
)


class ExperimentPlotter:
    """
    Matplotlib visualization for experiment results.

    Responsibilities:
    - tracking plots
    - CBF velocity plots
    - h(z) plots
    - 3D trajectory
    - 2D trajectory
    - nullspace torque
    - Digital Twin comparison
    - waypoint accuracy

    This class does NOT:
    - execute control
    - communicate with robot
    - modify controller state
    - run PyBullet
    """

    def __init__(
        self,
        output_dir: str | os.PathLike,
        trajectory: Optional[str] = None,
        control_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        cbf_mode: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.trajectory = trajectory
        self.control_mode = control_mode
        self.speed_mode = speed_mode
        self.cbf_mode = cbf_mode
        self.timestamp = timestamp

    # ==============================================================
    # Utility
    # ==============================================================

    @staticmethod
    def _array(
        data: Dict[str, Any],
        key: str,
        ndim: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        value = data.get(key)

        if value is None:
            return None

        arr = np.asarray(value)

        if arr.size == 0:
            return None

        if ndim is not None and arr.ndim != ndim:
            return None

        return arr

    def _save(
        self,
        fig,
        filename: str,
    ) -> Path:
        path = self.output_dir / filename

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"📊 Grafik disimpan: {path}")

        return path

    @staticmethod
    def _close(fig):
        plt.close(fig)

    def _mode_label(self) -> str:
        if self.control_mode in ("JSA", "TSA"):
            return (
                f"{self.control_mode}"
                + (
                    f" ({self.speed_mode})"
                    if self.speed_mode
                    else ""
                )
            )

        if self.control_mode:
            return self.control_mode.capitalize()

        return "Experiment"

    # ==============================================================
    # 1. Tracking XYZ
    # ==============================================================

    def plot_tracking_xyz(
        self,
        log_data: Dict[str, Any],
        rtde_min: Optional[np.ndarray] = None,
        rtde_max: Optional[np.ndarray] = None,
    ):
        t = self._array(log_data, "log_t")
        x_act = self._array(log_data, "log_x_act", ndim=2)
        x_des = self._array(log_data, "log_x_des", ndim=2)

        if t is None or x_act is None or x_des is None:
            return None

        fig, axs = plt.subplots(
            3,
            1,
            figsize=(10, 15),
            sharex=True,
        )

        axes_names = ["X", "Y", "Z"]

        for i in range(3):
            axs[i].plot(
                t,
                x_des[:, i],
                "r--",
                linewidth=2,
                label="Desired",
            )

            axs[i].plot(
                t,
                x_act[:, i],
                "b-",
                linewidth=1.5,
                label="Actual",
            )

            if rtde_min is not None and rtde_max is not None:
                axs[i].axhline(
                    y=rtde_max[i],
                    color="g",
                    linestyle="--",
                    linewidth=2,
                    label=(
                        "Safety Containment"
                        if i == 0
                        else ""
                    ),
                )

                axs[i].axhline(
                    y=rtde_min[i],
                    color="g",
                    linestyle="--",
                    linewidth=2,
                )

            error = x_des[:, i] - x_act[:, i]

            rmse_total = np.sqrt(
                np.mean(error ** 2)
            )

            max_error = np.max(
                np.abs(error)
            )

            dq_cmd = self._array(
                log_data,
                "q_dot",
                ndim=2,
            )

            dq_safe = self._array(
                log_data,
                "log_dq_cmd_safe",
                ndim=2,
            )

            if (
                dq_cmd is not None
                and dq_safe is not None
                and len(dq_cmd) == len(dq_safe)
            ):
                diff = np.linalg.norm(
                    dq_cmd - dq_safe,
                    axis=1,
                )

                free_space = diff < 1e-3

                if np.any(free_space):
                    rmse_free = np.sqrt(
                        np.mean(
                            error[free_space] ** 2
                        )
                    )
                else:
                    rmse_free = 0.0
            else:
                rmse_free = rmse_total

            stats = (
                f"Total RMSE: {rmse_total:.4f} m\n"
                f"Free RMSE: {rmse_free:.4f} m\n"
                f"Max Error: {max_error:.4f} m"
            )

            axs[i].text(
                0.96,
                0.96,
                stats,
                transform=axs[i].transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="black",
                ),
            )

            axs[i].set_ylabel(
                f"Pos {axes_names[i]} (m)"
            )

            axs[i].set_title(
                f"Tracking Axis {axes_names[i]}"
            )

            axs[i].grid(
                True,
                linestyle="--",
                alpha=0.7,
            )

            axs[i].legend(
                loc="upper left"
            )

        axs[2].set_xlabel("Time (s)")

        fig.suptitle(
            f"Tracking Performance - {self._mode_label()}",
            fontsize=16,
        )

        fig.tight_layout(
            rect=[0, 0.03, 1, 0.97]
        )

        return self._save(
            fig,
            f"Plot_Tracking_XYZ_{self.timestamp or 'result'}.png",
        )

    # ==============================================================
    # 2. Joint velocity: unsafe vs safe vs actual
    # ==============================================================

    def plot_joint_velocities(
        self,
        log_data: Dict[str, Any],
    ):
        t = self._array(log_data, "log_t")
        dq_cmd = self._array(
            log_data,
            "q_dot",
            ndim=2,
        )

        if t is None or dq_cmd is None:
            return None

        dq_safe = self._array(
            log_data,
            "log_dq_cmd_safe",
            ndim=2,
        )

        dq_actual = self._array(
            log_data,
            "dq_act",
            ndim=2,
        )

        fig, axes = plt.subplots(
            3,
            2,
            figsize=(15, 10),
            sharex=True,
        )

        axes = axes.ravel()

        fig.suptitle(
            f"CBF Joint Velocity Filtering - "
            f"{self._mode_label()}",
            fontsize=14,
        )

        for i in range(6):

            axes[i].plot(
                t,
                dq_cmd[:, i],
                "r--",
                linewidth=1.2,
                label="dq_cmd (Nominal/Unsafe)",
            )

            if (
                dq_safe is not None
                and len(dq_safe) == len(dq_cmd)
            ):
                axes[i].plot(
                    t,
                    dq_safe[:, i],
                    "g-",
                    linewidth=1.5,
                    label="dq_safe (CBF)",
                )

            if (
                dq_actual is not None
                and len(dq_actual) == len(dq_cmd)
            ):
                axes[i].plot(
                    t,
                    dq_actual[:, i],
                    "b-.",
                    alpha=0.8,
                    linewidth=1.2,
                    label="dq_act (Actual)",
                )

            if dq_safe is not None:
                deviation = np.abs(
                    dq_cmd[:, i]
                    - dq_safe[:, i]
                )

                stats = (
                    f"Max CBF Dev: "
                    f"{np.max(deviation):.4f} rad/s\n"
                    f"Avg CBF Dev: "
                    f"{np.mean(deviation):.4f} rad/s"
                )

            elif dq_actual is not None:
                error = np.abs(
                    dq_cmd[:, i]
                    - dq_actual[:, i]
                )

                stats = (
                    "CBF Mode: OFF\n"
                    f"Avg Cmd-Act Err: "
                    f"{np.mean(error):.4f} rad/s"
                )

            else:
                stats = "CBF Mode: OFF"

            axes[i].text(
                0.96,
                0.96,
                stats,
                transform=axes[i].transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="black",
                ),
            )

            axes[i].set_title(
                f"Joint {i + 1} Velocity"
            )

            axes[i].set_ylabel("rad/s")

            axes[i].grid(
                True,
                linestyle="--",
                alpha=0.7,
            )

        axes[4].set_xlabel("Time (s)")
        axes[5].set_xlabel("Time (s)")

        handles, labels = (
            axes[0].get_legend_handles_labels()
        )

        fig.legend(
            handles,
            labels,
            loc="upper right",
        )

        fig.tight_layout(
            rect=[0, 0.03, 1, 0.97]
        )

        return self._save(
            fig,
            f"Plot_CBF_Velocities_{self.timestamp or 'result'}.png",
        )

    # ==============================================================
    # 3. h(z)
    # ==============================================================

    def plot_h_values(
        self,
        log_data: Dict[str, Any],
        epsilon_0: float = 0.0,
        alpha_c: float = 0.0,
    ):
        h_values = self._array(
            log_data,
            "log_h_values",
            ndim=2,
        )

        t = self._array(
            log_data,
            "log_t",
        )

        if (
            h_values is None
            or t is None
            or len(h_values) <= 1
        ):
            return None

        fig, ax = plt.subplots(
            figsize=(6, 5),
            dpi=300,
        )

        labels = [
            "Max X",
            "Max Y",
            "Max Z",
            "Min X",
            "Min Y",
            "Min Z",
        ]

        num_constraints = h_values.shape[1]

        for i in range(num_constraints):
            label = (
                labels[i]
                if i < len(labels)
                else f"Constraint_{i - 5}"
            )

            ax.plot(
                t,
                h_values[:, i],
                linewidth=1.0,
                alpha=0.7,
                label=label if i < 8 else None,
            )

        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

        # ISSf theoretical bound
        disturbance = self._array(
            log_data,
            "log_disturbance",
        )

        if (
            disturbance is not None
            and alpha_c > 0
            and epsilon_0 > 0
        ):
            d_max = np.max(
                np.abs(disturbance)
            )

            h_star = (
                -(d_max ** 2 * epsilon_0)
                / (4.0 * alpha_c)
            )

            ax.axhline(
                h_star,
                color="red",
                linestyle=":",
                linewidth=2,
                label=f"ISSf Bound $h^*$ ({h_star:.3f})",
            )

        min_h = float(
            np.min(h_values)
        )

        violation = (
            "Yes"
            if min_h < 0
            else "No"
        )

        stats = (
            f"Min h: {min_h:.8f}\n"
            f"Violation: {violation}\n"
            f"Alpha: {alpha_c}\n"
            f"$\\epsilon_0$: {epsilon_0}"
        )

        ax.text(
            0.96,
            0.04,
            stats,
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8,
                edgecolor="black",
            ),
        )

        ax.set_ylabel("$h(z)$")
        ax.set_xlabel("Time (s)")
        ax.set_title("Constraint Evolution h(z)")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        return self._save(
            fig,
            f"Plot_h_Values_{self.timestamp or 'result'}.png",
        )

    # ==============================================================
    # 4. 3D trajectory
    # ==============================================================

    def plot_trajectory_3d(
        self,
        log_data,
        robot_base_position=None,
        obstacle_params=None,
        tube_params=None,
        tube_radius=None,
        rtde_min=None,
        rtde_max=None,
        wb_rtde_min=None,
        wb_rtde_max=None,
    ):
        x_act = self._array(
            log_data,
            "log_x_act",
            ndim=2,
        )

        x_des = self._array(
            log_data,
            "log_x_des",
            ndim=2,
        )

        if x_act is None:
            return None

        fig = plt.figure(
            figsize=(10, 8)
        )

        ax = fig.add_subplot(
            111,
            projection="3d",
        )

        ax.plot(
            x_act[:, 0],
            x_act[:, 1],
            x_act[:, 2],
            label="Actual Trajectory",
            linewidth=2,
        )

        if (
            x_des is not None
            and self.trajectory != "wp"
        ):
            ax.plot(
                x_des[:, 0],
                x_des[:, 1],
                x_des[:, 2],
                linestyle="--",
                label="Desired Trajectory",
            )

        ax.scatter(
            x_act[0, 0],
            x_act[0, 1],
            x_act[0, 2],
            marker="o",
            s=100,
            label="Start",
        )

        ax.scatter(
            x_act[-1, 0],
            x_act[-1, 1],
            x_act[-1, 2],
            marker="X",
            s=120,
            label="End",
        )

        # ----------------------------------------------------------
        # Obstacle visualization
        # ----------------------------------------------------------

        if (
            self.cbf_mode in ("obstacle", "both")
            and obstacle_params is not None
            and robot_base_position is not None
        ):
            base = np.asarray(
                robot_base_position,
                dtype=float,
            )

            if (
                obstacle_params.get("position")
                is not None
                and obstacle_params.get("size")
                is not None
            ):
                center_pyb = np.asarray(
                    obstacle_params["position"],
                    dtype=float,
                )

                center_rtde = (
                    center_pyb - base
                ) * np.array(
                    [-1.0, -1.0, 1.0]
                )

                radius = (
                    float(obstacle_params["size"][0])
                    / 2.0
                )

                self._draw_sphere_3d(
                    ax,
                    center_rtde,
                    radius,
                    label="Obstacle",
                    alpha=0.3,
                )

            # ------------------------------------------------------
            # Tube obstacles
            # ------------------------------------------------------

            if tube_params:
                for i, tube in enumerate(
                    tube_params
                ):
                    radius = float(
                        tube_radius
                    ) if tube_radius is not None else 0.0

                    if radius <= 0:
                        radius = float(
                            tube.get(
                                "tube_radius",
                                0.0,
                            )
                        )

                    sphere_positions = tube.get(
                        "sphere_positions",
                        [],
                    )

                    for j, pos_pyb in enumerate(
                        sphere_positions
                    ):
                        pos_pyb = np.asarray(
                            pos_pyb,
                            dtype=float,
                        )

                        center_rtde = (
                            pos_pyb - base
                        ) * np.array(
                            [-1.0, -1.0, 1.0]
                        )

                        self._draw_sphere_3d(
                            ax,
                            center_rtde,
                            radius,
                            label=(
                                "Tube Obstacle"
                                if (
                                    i == 0
                                    and j == 0
                                )
                                else None
                            ),
                            alpha=0.15,
                        )

        # ----------------------------------------------------------
        # EE containment boundary
        # ----------------------------------------------------------

        if (
            rtde_min is not None
            and rtde_max is not None
        ):
            self._draw_box_3d(
                ax,
                np.asarray(rtde_min),
                np.asarray(rtde_max),
                linestyle="--",
                label="EE Safety Boundary",
            )

        # ----------------------------------------------------------
        # Whole-body boundary
        # ----------------------------------------------------------

        if (
            wb_rtde_min is not None
            and wb_rtde_max is not None
        ):
            self._draw_box_3d(
                ax,
                np.asarray(wb_rtde_min),
                np.asarray(wb_rtde_max),
                linestyle=":",
                label="Whole-Body Boundary",
            )

                # ----------------------------------------------------------
        # Obstacle visualization
        # ----------------------------------------------------------

        if (
            self.cbf_mode in ("obstacle", "both")
            and obstacle_params is not None
            and robot_base_position is not None
        ):
            base = np.asarray(
                robot_base_position,
                dtype=float,
            )

            if (
                obstacle_params.get("position")
                is not None
                and obstacle_params.get("size")
                is not None
            ):
                center_pyb = np.asarray(
                    obstacle_params["position"],
                    dtype=float,
                )

                center_rtde = (
                    center_pyb - base
                ) * np.array(
                    [-1.0, -1.0, 1.0]
                )

                radius = (
                    float(obstacle_params["size"][0])
                    / 2.0
                )

                self._draw_sphere_3d(
                    ax,
                    center_rtde,
                    radius,
                    label="Obstacle",
                    alpha=0.3,
                )

            # ------------------------------------------------------
            # Tube obstacles
            # ------------------------------------------------------

            if tube_params:
                for i, tube in enumerate(
                    tube_params
                ):
                    radius = float(
                        tube.get(
                            "radius",
                            0.0,
                        )
                    )

                    if radius <= 0:
                        radius = float(
                            tube.get(
                                "tube_radius",
                                0.0,
                            )
                        )

                    sphere_positions = tube.get(
                        "sphere_positions",
                        [],
                    )

                    for j, pos_pyb in enumerate(
                        sphere_positions
                    ):
                        pos_pyb = np.asarray(
                            pos_pyb,
                            dtype=float,
                        )

                        center_rtde = (
                            pos_pyb - base
                        ) * np.array(
                            [-1.0, -1.0, 1.0]
                        )

                        self._draw_sphere_3d(
                            ax,
                            center_rtde,
                            radius,
                            label=(
                                "Tube Obstacle"
                                if (
                                    i == 0
                                    and j == 0
                                )
                                else None
                            ),
                            alpha=0.15,
                        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        ax.set_title(
            "3D Trajectory Visualization"
        )

        ax.legend()
        ax.grid(True)

        self._set_equal_3d_aspect(ax)

        fig.tight_layout()

        return self._save(
            fig,
            f"Plot_3D_Trajectory_{self.timestamp or 'result'}.png",
        )

    # ==============================================================
    # 5. 2D trajectory
    # ==============================================================

    def plot_trajectory_2d(
        self,
        log_data,
        robot_base_position=None,
        obstacle_params=None,
        tube_params=None,
        tube_radius=None,
        rtde_min=None,
        rtde_max=None,
        wb_rtde_min=None,
        wb_rtde_max=None,
    ):
        x_act = self._array(
            log_data,
            "log_x_act",
            ndim=2,
        )

        x_des = self._array(
            log_data,
            "log_x_des",
            ndim=2,
        )

        if x_act is None:
            return None

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        ax.plot(
            x_act[:, 0],
            x_act[:, 1],
            label="Actual Trajectory",
            linewidth=2,
        )

        if (
            x_des is not None
            and self.trajectory != "wp"
        ):
            ax.plot(
                x_des[:, 0],
                x_des[:, 1],
                linestyle="--",
                label="Desired Trajectory",
            )

        ax.scatter(
            x_act[0, 0],
            x_act[0, 1],
            marker="o",
            s=100,
            label="Start",
        )

        ax.scatter(
            x_act[-1, 0],
            x_act[-1, 1],
            marker="X",
            s=120,
            label="End",
        )

        # ----------------------------------------------------------
        # Obstacle visualization
        # ----------------------------------------------------------

        if (
            self.cbf_mode in ("obstacle", "both")
            and robot_base_position is not None
        ):
            base = np.asarray(
                robot_base_position,
                dtype=float,
            )

            # Sphere obstacle
            if (
                obstacle_params is not None
                and "position" in obstacle_params
                and "size" in obstacle_params
            ):
                center_pyb = np.asarray(
                    obstacle_params["position"],
                    dtype=float,
                )

                center_rtde = (
                    center_pyb - base
                ) * np.array(
                    [-1.0, -1.0, 1.0]
                )

                radius = (
                    float(
                        obstacle_params["size"][0]
                    ) / 2.0
                )

                circle = patches.Circle(
                    (
                        center_rtde[0],
                        center_rtde[1],
                    ),
                    radius=radius,
                    alpha=0.3,
                    label="Obstacle",
                )

                ax.add_patch(circle)

            # Tube obstacle
            if tube_params:
                for i, tube in enumerate(
                    tube_params
                ):
                    sphere_positions = tube.get(
                        "sphere_positions",
                        [],
                    )

                    if not sphere_positions:
                        continue

                    pos_pyb = np.asarray(
                        sphere_positions[0],
                        dtype=float,
                    )

                    center_rtde = (
                        pos_pyb - base
                    ) * np.array(
                        [-1.0, -1.0, 1.0]
                    )

                    radius = (
                        float(tube_radius)
                        if tube_radius is not None
                        else 0.0
                    )

                    circle = patches.Circle(
                        (
                            center_rtde[0],
                            center_rtde[1],
                        ),
                        radius=radius,
                        alpha=0.3,
                        label=(
                            "Tube Obstacle"
                            if i == 0
                            else None
                        ),
                    )

                    ax.add_patch(circle)

        # EE containment box
        if (
            rtde_min is not None
            and rtde_max is not None
        ):
            rtde_min = np.asarray(rtde_min)
            rtde_max = np.asarray(rtde_max)

            rect = patches.Rectangle(
                (
                    rtde_min[0],
                    rtde_min[1],
                ),
                rtde_max[0] - rtde_min[0],
                rtde_max[1] - rtde_min[1],
                linewidth=2,
                fill=False,
                linestyle="--",
                label="EE Safety Boundary",
            )

            ax.add_patch(rect)

        # Whole body box
        if (
            wb_rtde_min is not None
            and wb_rtde_max is not None
        ):
            wb_rtde_min = np.asarray(
                wb_rtde_min
            )
            wb_rtde_max = np.asarray(
                wb_rtde_max
            )

            rect = patches.Rectangle(
                (
                    wb_rtde_min[0],
                    wb_rtde_min[1],
                ),
                wb_rtde_max[0]
                - wb_rtde_min[0],
                wb_rtde_max[1]
                - wb_rtde_min[1],
                linewidth=2,
                fill=False,
                linestyle=":",
                label="Whole-Body Boundary",
            )

            ax.add_patch(rect)



        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        ax.set_title(
            "2D Trajectory Visualization (Top View)"
        )

        ax.set_aspect(
            "equal",
            adjustable="datalim",
        )

        ax.grid(True)

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.0),
        )

        fig.tight_layout()

        return self._save(
            fig,
            f"Plot_2D_TopView_{self.timestamp or 'result'}.png",
        )



    # ==============================================================
    # 6. Nullspace torque
    # ==============================================================

    def plot_nullspace_torque(
        self,
        log_data: Dict[str, Any],
    ):
        t = self._array(log_data, "log_t")
        tau_null = self._array(
            log_data,
            "tau_null",
            ndim=2,
        )

        if t is None or tau_null is None:
            return None

        fig, axes = plt.subplots(
            3,
            2,
            figsize=(15, 10),
            sharex=True,
        )

        axes = axes.ravel()

        fig.suptitle(
            f"Nullspace Torque over Time - "
            f"{self._mode_label()}",
            fontsize=16,
        )

        for i in range(6):
            axes[i].plot(
                t,
                tau_null[:, i],
                linewidth=1.5,
                label=f"Joint {i + 1} Null Torque",
            )

            axes[i].set_title(
                f"Joint {i + 1}"
            )

            axes[i].set_ylabel(
                "Torque (Nm)"
            )

            axes[i].grid(
                True,
                linestyle="--",
                alpha=0.7,
            )

            axes[i].legend()

            if i >= 4:
                axes[i].set_xlabel(
                    "Time (s)"
                )

        fig.tight_layout(
            rect=[0, 0.03, 1, 0.97]
        )

        return self._save(
            fig,
            f"Plot_Nullspace_Torque_{self.timestamp or 'result'}.png",
        )

    # ==============================================================
    # 7. Digital Twin
    # ==============================================================

    def plot_digital_twin(
        self,
        log_data: Dict[str, Any],
    ):
        t = self._array(log_data, "log_t")

        q_sim = self._array(
            log_data,
            "log_q_pybullet",
            ndim=2,
        )

        q_real = self._array(
            log_data,
            "log_q_ur5e",
            ndim=2,
        )

        if (
            t is None
            or q_sim is None
            or q_real is None
        ):
            return None

        fig, axes = plt.subplots(
            3,
            2,
            figsize=(15, 10),
            sharex=True,
        )

        axes = axes.ravel()

        fig.suptitle(
            "Comparison: PyBullet vs Actual UR5e Joint Positions",
            fontsize=16,
        )

        for i in range(6):

            axes[i].plot(
                t,
                q_real[:, i],
                linewidth=2.5,
                alpha=0.7,
                label="UR5e (Actual)",
            )

            axes[i].plot(
                t,
                q_sim[:, i],
                "--",
                linewidth=1.5,
                label="PyBullet (Visual)",
            )

            avg_diff = np.mean(
                np.abs(
                    q_real[:, i]
                    - q_sim[:, i]
                )
            )

            axes[i].text(
                0.96,
                0.04,
                f"Avg Diff: {avg_diff:.6f} rad",
                transform=axes[i].transAxes,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.8,
                ),
            )

            axes[i].set_title(
                f"Joint {i + 1}"
            )

            axes[i].set_ylabel(
                "Position (rad)"
            )

            axes[i].grid(
                True,
                linestyle="--",
                alpha=0.7,
            )

            axes[i].legend()

            if i >= 4:
                axes[i].set_xlabel(
                    "Time (s)"
                )

        fig.tight_layout(
            rect=[0, 0.03, 1, 0.97]
        )

        return self._save(
            fig,
            f"Plot_PyBullet_vs_UR5e_{self.timestamp or 'result'}.png",
        )



    def plot_waypoint_accuracy(
        self,
        log_data,
        events,
        waypoint_positions,
    ):
        if self.trajectory not in ("wp", "pap2"):
            return None

        t = self._array(
            log_data,
            "log_t",
        )

        x_act = self._array(
            log_data,
            "log_x_act",
            ndim=2,
        )

        if t is None or x_act is None:
            return None

        wp_labels = []
        wp_errors_mm = []
        cbf_affected = []

        q_dot = self._array(
            log_data,
            "q_dot",
            ndim=2,
        )

        dq_safe = self._array(
            log_data,
            "log_dq_cmd_safe",
            ndim=2,
        )

        for event_time, label in events:

            idx = int(
                np.argmin(
                    np.abs(
                        t - event_time
                    )
                )
            )

            pos_act = x_act[idx]

            if label not in waypoint_positions:
                continue

            pos_des = np.asarray(
                waypoint_positions[label],
                dtype=float,
            )

            error_mm = (
                np.linalg.norm(
                    pos_act - pos_des
                )
                * 1000.0
            )

            # ----------------------------------------------------------
            # Detect CBF intervention
            # ----------------------------------------------------------

            is_cbf_active = False

            if (
                q_dot is not None
                and dq_safe is not None
                and len(q_dot) == len(dq_safe)
            ):
                idx_start = max(
                    0,
                    idx - 5,
                )

                idx_end = min(
                    len(q_dot),
                    idx + 5,
                )

                diff_vel = np.linalg.norm(
                    q_dot[idx_start:idx_end]
                    - dq_safe[idx_start:idx_end],
                    axis=1,
                )

                if np.any(
                    diff_vel > 1e-3
                ):
                    is_cbf_active = True

            # Match behavior from the original code
            if label in ("P0", "P1", "P2"):
                if error_mm <= 15.0:
                    is_cbf_active = False

            wp_labels.append(
                f"{label}\n"
                f"(t={event_time:.1f}s)"
            )

            wp_errors_mm.append(
                error_mm
            )

            cbf_affected.append(
                is_cbf_active
            )

        if not wp_errors_mm:
            return None

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        bars = ax.bar(
            wp_labels,
            wp_errors_mm,
            alpha=0.8,
            edgecolor="black",
        )

        for i, bar in enumerate(bars):
            y = bar.get_height()

            text = f"{y:.1f} mm"

            if cbf_affected[i]:
                text += "\nCBF"

            ax.text(
                bar.get_x()
                + bar.get_width() / 2.0,
                y + 0.2,
                text,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        errors = np.asarray(
            wp_errors_mm
        )

        normal_errors = np.asarray(
            [
                error
                for error, cbf in zip(
                    wp_errors_mm,
                    cbf_affected,
                )
                if not cbf
            ]
        )

        total_rmse = np.sqrt(
            np.mean(errors ** 2)
        )

        free_rmse = (
            np.sqrt(
                np.mean(
                    normal_errors ** 2
                )
            )
            if normal_errors.size > 0
            else 0.0
        )

        stats = (
            f"Total Waypoint RMSE: "
            f"{total_rmse:.2f} mm\n"
            f"Free Waypoint RMSE: "
            f"{free_rmse:.2f} mm"
        )

        ax.text(
            0.98,
            0.95,
            stats,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.9,
                edgecolor="gray",
            ),
        )

        ax.set_xlabel(
            "Target Waypoint & Arrival Time"
        )

        ax.set_ylabel(
            "Absolute Distance Error (mm)"
        )

        ax.set_title(
            "Waypoint Arrival Accuracy"
        )

        ax.grid(
            axis="y",
            linestyle=":",
            alpha=0.7,
        )

        fig.tight_layout()

        return self._save(
            fig,
            f"Plot_Waypoint_CBF_{self.timestamp or 'result'}.png",
        )


    def plot_waypoint_xyz(
        self,
        log_data,
        events,
        waypoint_positions,
    ):
        if self.trajectory not in ("wp", "pap2"):
            return None

        t = self._array(
            log_data,
            "log_t",
        )

        x_act = self._array(
            log_data,
            "log_x_act",
            ndim=2,
        )

        if t is None or x_act is None:
            return None

        labels = []

        errors_x = []
        errors_y = []
        errors_z = []

        for event_time, label in events:

            if label not in waypoint_positions:
                continue

            idx = int(
                np.argmin(
                    np.abs(
                        t - event_time
                    )
                )
            )

            target = np.asarray(
                waypoint_positions[label],
                dtype=float,
            )

            actual = x_act[idx]

            labels.append(
                f"{label}\n"
                f"({event_time:.1f}s)"
            )

            errors_x.append(
                abs(target[0] - actual[0])
                * 1000.0
            )

            errors_y.append(
                abs(target[1] - actual[1])
                * 1000.0
            )

            errors_z.append(
                abs(target[2] - actual[2])
                * 1000.0
            )

        if not labels:
            return None

        x_indices = np.arange(
            len(labels)
        )

        width = 0.25

        fig, ax = plt.subplots(
            figsize=(14, 6)
        )

        ax.bar(
            x_indices - width,
            errors_x,
            width,
            label="Error X",
            edgecolor="black",
            alpha=0.85,
        )

        ax.bar(
            x_indices,
            errors_y,
            width,
            label="Error Y",
            edgecolor="black",
            alpha=0.85,
        )

        ax.bar(
            x_indices + width,
            errors_z,
            width,
            label="Error Z",
            edgecolor="black",
            alpha=0.85,
        )

        rmse_x = np.sqrt(
            np.mean(
                np.asarray(errors_x) ** 2
            )
        )

        rmse_y = np.sqrt(
            np.mean(
                np.asarray(errors_y) ** 2
            )
        )

        rmse_z = np.sqrt(
            np.mean(
                np.asarray(errors_z) ** 2
            )
        )

        stats = (
            f"WP RMSE X: {rmse_x:.2f} mm\n"
            f"WP RMSE Y: {rmse_y:.2f} mm\n"
            f"WP RMSE Z: {rmse_z:.2f} mm"
        )

        ax.text(
            0.98,
            0.95,
            stats,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.9,
                edgecolor="gray",
            ),
            fontsize=11,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Target Waypoint & Arrival Time"
        )

        ax.set_ylabel(
            "Absolute Error per Axis (mm)"
        )

        ax.set_title(
            "Waypoint Arrival Accuracy per Axis"
        )

        ax.set_xticks(x_indices)
        ax.set_xticklabels(labels)

        ax.legend(
            loc="upper left"
        )

        ax.grid(
            axis="y",
            linestyle=":",
            alpha=0.7,
        )

        max_error = max(
            max(errors_x),
            max(errors_y),
            max(errors_z),
        )

        ax.set_ylim(
            0,
            max_error * 1.25
            if max_error > 0
            else 1.0,
        )

        fig.tight_layout()

        return self._save(
            fig,
            f"Plot_Waypoint_XYZ_{self.timestamp or 'result'}.png",
        )

    # ==============================================================
    # 8. Generate all standard plots
    # ==============================================================

    def plot_all(
        self,
        log_data,
        rtde_min=None,
        rtde_max=None,
        wb_rtde_min=None,
        wb_rtde_max=None,
        robot_base_position=None,
        obstacle_params=None,
        tube_params=None,
        tube_radius=None,
        epsilon_0=0.0,
        alpha_c=0.0,
        waypoint_events=None,
        waypoint_positions=None,
    ):
        results = {}

        results["tracking"] = self.plot_tracking_xyz(
            log_data,
            rtde_min=rtde_min,
            rtde_max=rtde_max,
        )

        results["joint_velocity"] = (
            self.plot_joint_velocities(
                log_data
            )
        )

        results["h_values"] = self.plot_h_values(
            log_data,
            epsilon_0=epsilon_0,
            alpha_c=alpha_c,
        )

        results["trajectory_3d"] = (
            self.plot_trajectory_3d(
                log_data,
                robot_base_position=robot_base_position,
                obstacle_params=obstacle_params,
                tube_params=tube_params,
                tube_radius=tube_radius,
                rtde_min=rtde_min,
                rtde_max=rtde_max,
                wb_rtde_min=wb_rtde_min,
                wb_rtde_max=wb_rtde_max,
            )
        )

        results["trajectory_2d"] = (
            self.plot_trajectory_2d(
                log_data,
                rtde_min=rtde_min,
                rtde_max=rtde_max,
                wb_rtde_min=wb_rtde_min,
                wb_rtde_max=wb_rtde_max,
            )
        )

        results["nullspace"] = (
            self.plot_nullspace_torque(
                log_data
            )
        )

        results["digital_twin"] = (
            self.plot_digital_twin(
                log_data
            )
        )

        if (
            waypoint_events is not None
            and waypoint_positions is not None
        ):
            results["waypoint_accuracy"] = (
                self.plot_waypoint_accuracy(
                    log_data,
                    waypoint_events,
                    waypoint_positions,
                )
            )

            results["waypoint_xyz"] = (
                self.plot_waypoint_xyz(
                    log_data,
                    waypoint_events,
                    waypoint_positions,
                )
            )

        return results

    # ==============================================================
    # Internal geometry helpers
    # ==============================================================

    @staticmethod
    def _draw_box_3d(
        ax,
        min_b,
        max_b,
        linestyle="--",
        label=None,
    ):
        corners = [
            np.array([
                min_b[0],
                min_b[1],
                min_b[2],
            ]),
            np.array([
                max_b[0],
                min_b[1],
                min_b[2],
            ]),
            np.array([
                max_b[0],
                max_b[1],
                min_b[2],
            ]),
            np.array([
                min_b[0],
                max_b[1],
                min_b[2],
            ]),
            np.array([
                min_b[0],
                min_b[1],
                max_b[2],
            ]),
            np.array([
                max_b[0],
                min_b[1],
                max_b[2],
            ]),
            np.array([
                max_b[0],
                max_b[1],
                max_b[2],
            ]),
            np.array([
                min_b[0],
                max_b[1],
                max_b[2],
            ]),
        ]

        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        for i, (a, b) in enumerate(edges):
            p1 = corners[a]
            p2 = corners[b]

            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                linestyle=linestyle,
                label=(
                    label if i == 0 else None
                ),
            )

    @staticmethod
    def _set_equal_3d_aspect(ax):
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        z_limits = ax.get_zlim()

        x_range = abs(
            x_limits[1] - x_limits[0]
        )

        y_range = abs(
            y_limits[1] - y_limits[0]
        )

        z_range = abs(
            z_limits[1] - z_limits[0]
        )

        radius = 0.5 * max(
            x_range,
            y_range,
            z_range,
        )

        x_mid = np.mean(x_limits)
        y_mid = np.mean(y_limits)
        z_mid = np.mean(z_limits)

        ax.set_xlim(
            x_mid - radius,
            x_mid + radius,
        )

        ax.set_ylim(
            y_mid - radius,
            y_mid + radius,
        )

        ax.set_zlim(
            z_mid - radius,
            z_mid + radius,
        )

    def _draw_sphere_3d(
        self,
        ax,
        center,
        radius,
        label=None,
        alpha=0.3,
    ):
        center = np.asarray(center, dtype=float)

        u, v = np.mgrid[
            0 : 2 * np.pi : 20j,
            0 : np.pi : 10j,
        ]

        x = (
            center[0]
            + radius * np.cos(u) * np.sin(v)
        )

        y = (
            center[1]
            + radius * np.sin(u) * np.sin(v)
        )

        z = (
            center[2]
            + radius * np.cos(v)
        )

        ax.plot_surface(
            x,
            y,
            z,
            alpha=alpha,
            label=label,
        )