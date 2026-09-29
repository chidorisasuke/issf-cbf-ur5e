from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pybullet as p


class PyBulletVisualizer:
    """
    Visual helpers for the UR5e PyBullet digital twin.

    This class does NOT:
    - create the PyBullet environment
    - communicate with UR5e
    - calculate trajectories
    - calculate CBF
    - calculate control commands

    It only updates/draws visual objects.
    """

    def __init__(
        self,
        robot_base_position: Iterable[float],
    ):
        self.robot_base_position = np.asarray(
            robot_base_position,
            dtype=float,
        )

        self.trajectory_line_ids = []

    # ==========================================================
    # TRAJECTORY PREVIEW
    # ==========================================================

    def draw_trajectory_preview(
        self,
        trajectory,
        trajectory_name,
        x_base,
        duration,
        dt,
        sample_skip=20,
    ):
        """
        Draw desired trajectory preview in PyBullet.

        Trajectory is defined in RTDE frame.
        PyBullet visualization uses:
            x_pyb = -x_rtde
            y_pyb = -y_rtde
            z_pyb =  z_rtde

        The robot base position is then added to obtain
        world coordinates.
        """

        if not p.isConnected():
            return

        if trajectory_name == "mouse":
            return

        print(
            f"\n🎨 Menggambar lintasan "
            f"{trajectory_name} di PyBullet..."
        )

        # Clear old trajectory preview
        for line_id in getattr(
            self,
            "trajectory_line_ids",
            [],
        ):
            try:
                p.removeUserDebugItem(line_id)
            except Exception:
                pass

        self.trajectory_line_ids = []

        # Disable rendering temporarily
        p.configureDebugVisualizer(
            p.COV_ENABLE_RENDERING,
            0,
        )

        try:
            sample_count = max(
                2,
                int(duration / (dt * sample_skip)) + 1,
            )

            times = np.linspace(
                0.0,
                duration,
                sample_count,
            )

            prev_pos_rtde = None

            for t_preview in times:

                pos_rtde, _, _ = trajectory.compute(
                    t_preview,
                    x_base,
                )

                pos_rtde = np.asarray(
                    pos_rtde,
                    dtype=float,
                ).copy()

                # --------------------------------------------------
                # RTDE -> PyBullet
                # --------------------------------------------------
                pos_pyb = pos_rtde.copy()
                pos_pyb[0] *= -1.0
                pos_pyb[1] *= -1.0

                world_pos = (
                    pos_pyb
                    + self.robot_base_position
                )

                if prev_pos_rtde is not None:

                    prev_pyb = prev_pos_rtde.copy()
                    prev_pyb[0] *= -1.0
                    prev_pyb[1] *= -1.0

                    prev_world = (
                        prev_pyb
                        + self.robot_base_position
                    )

                    line_id = p.addUserDebugLine(
                        lineFromXYZ=prev_world.tolist(),
                        lineToXYZ=world_pos.tolist(),
                        lineColorRGB=[0.0, 1.0, 0.0],
                        lineWidth=3.0,
                        lifeTime=0,
                    )

                    self.trajectory_line_ids.append(
                        line_id
                    )

                prev_pos_rtde = pos_rtde

        finally:
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING,
                1,
            )

        print(
            f"✅ Lintasan {trajectory_name} "
            f"berhasil digambar."
        )

    # ==============================================================
    # Coordinate conversion
    # ==============================================================

    @staticmethod
    def rtde_to_pybullet(
        position_rtde: Iterable[float],
        robot_base_position: Iterable[float],
    ) -> np.ndarray:
        """
        RTDE robot frame -> PyBullet/world frame.

        Matches the original hil14 implementation:
        X and Y are sign-flipped, then robot base position is added.
        """
        position_rtde = np.asarray(position_rtde, dtype=float)
        base = np.asarray(robot_base_position, dtype=float)

        position_pyb = position_rtde.copy()
        position_pyb[0] *= -1.0
        position_pyb[1] *= -1.0

        return position_pyb + base

    @staticmethod
    def pybullet_to_rtde(
        position_pybullet: Iterable[float],
        robot_base_position: Iterable[float],
    ) -> np.ndarray:
        """
        PyBullet/world frame -> RTDE robot frame.
        """
        position_pybullet = np.asarray(position_pybullet, dtype=float)
        base = np.asarray(robot_base_position, dtype=float)

        return (
            position_pybullet - base
        ) * np.array([-1.0, -1.0, 1.0])

    # ==============================================================
    # Target / EE visualization
    # ==============================================================

    def update_target(
        self,
        target_sphere_id: Optional[int],
        x_des: Iterable[float],
    ) -> None:
        """
        Update desired EE target sphere.
        """
        if target_sphere_id is None:
            return

        if not p.isConnected():
            return

        pos_world = self.rtde_to_pybullet(
            x_des,
            self.robot_base_position,
        )

        p.resetBasePositionAndOrientation(
            int(target_sphere_id),
            pos_world.tolist(),
            [0.0, 0.0, 0.0, 1.0],
        )

    def update_actual_ee(
        self,
        ee_actual_sphere_id: Optional[int],
        x_actual: Iterable[float],
    ) -> None:
        """
        Update actual EE visualization sphere.
        """
        if ee_actual_sphere_id is None:
            return

        if not p.isConnected():
            return

        pos_world = self.rtde_to_pybullet(
            x_actual,
            self.robot_base_position,
        )

        p.resetBasePositionAndOrientation(
            int(ee_actual_sphere_id),
            pos_world.tolist(),
            [0.0, 0.0, 0.0, 1.0],
        )

    # ==============================================================
    # Robot joint visualization
    # ==============================================================

    @staticmethod
    def update_joint_velocity_command(
        robot_sim: int,
        sim_joint_indices: Iterable[int],
        dq_cmd: Iterable[float],
        max_force: float = 150.0,
    ) -> None:
        """
        Drive the PyBullet robot using the same velocity command
        used by the original hil14 implementation.
        """
        if not p.isConnected():
            return

        dq_cmd = np.asarray(dq_cmd, dtype=float)

        for i, joint_idx in enumerate(sim_joint_indices):
            p.setJointMotorControl2(
                bodyIndex=int(robot_sim),
                jointIndex=int(joint_idx),
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=float(dq_cmd[i]),
                force=float(max_force),
            )

    @staticmethod
    def synchronize_joint_states(
        robot_sim: int,
        sim_joint_indices: Iterable[int],
        q: Iterable[float],
        dq: Iterable[float],
    ) -> None:
        """
        Hard-sync PyBullet joint states to measured robot states.

        Useful during initialization or explicit synchronization.
        """
        if not p.isConnected():
            return

        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)

        for i, joint_idx in enumerate(sim_joint_indices):
            p.resetJointState(
                int(robot_sim),
                int(joint_idx),
                float(q[i]),
                float(dq[i]),
            )

    # ==============================================================
    # Collision sphere visualization
    # ==============================================================

    def update_collision_spheres(
        self,
        robot_sphere_ids: Iterable[int],
        collision_data: np.ndarray,
    ) -> None:
        """
        Update CBF collision spheres attached to the robot.

        collision_data[:, :3] is expected to contain positions in
        the robot/Pinocchio frame.
        """
        if not p.isConnected():
            return

        robot_sphere_ids = list(robot_sphere_ids)

        if len(robot_sphere_ids) == 0:
            return

        collision_data = np.asarray(
            collision_data,
            dtype=float,
        )

        n = min(
            len(robot_sphere_ids),
            collision_data.shape[0],
        )

        for i in range(n):
            pos_pino = collision_data[i, :3]

            pos_world = (
                pos_pino
                + self.robot_base_position
            )

            p.resetBasePositionAndOrientation(
                int(robot_sphere_ids[i]),
                pos_world.tolist(),
                [0.0, 0.0, 0.0, 1.0],
            )

    # ==============================================================
    # Trajectory preview
    # ==============================================================

    def draw_trajectory(
        self,
        trajectory_fn,
        x_base: Iterable[float],
        duration: float,
        dt: float,
        skip: int = 20,
        line_color: Iterable[float] = (0.0, 1.0, 0.0),
        line_width: float = 3.0,
    ) -> None:
        """
        Draw a trajectory preview in PyBullet.

        trajectory_fn must have the form:

            x, xd, xdd = trajectory_fn(t, x_base)

        This keeps trajectory mathematics inside trajectory/.
        """
        if not p.isConnected():
            return

        x_base = np.asarray(x_base, dtype=float)

        print("\n🎨 Menggambar trajectory preview...")

        p.configureDebugVisualizer(
            p.COV_ENABLE_RENDERING,
            0,
        )

        try:
            steps = max(
                1,
                int(duration / dt),
            )

            prev_pos, _, _ = trajectory_fn(
                0.0,
                x_base,
            )

            prev_pos = np.asarray(
                prev_pos,
                dtype=float,
            )

            for i in range(
                0,
                steps,
                max(1, int(skip)),
            ):
                t_future = i * dt

                pos_future, _, _ = trajectory_fn(
                    t_future,
                    x_base,
                )

                pos_future = np.asarray(
                    pos_future,
                    dtype=float,
                )

                viz_from = self.rtde_to_pybullet(
                    prev_pos,
                    self.robot_base_position,
                )

                viz_to = self.rtde_to_pybullet(
                    pos_future,
                    self.robot_base_position,
                )

                p.addUserDebugLine(
                    lineFromXYZ=viz_from.tolist(),
                    lineToXYZ=viz_to.tolist(),
                    lineColorRGB=list(line_color),
                    lineWidth=float(line_width),
                    lifeTime=0,
                )

                prev_pos = pos_future

        finally:
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING,
                1,
            )

        print("✅ Trajectory preview selesai.")

    # ==============================================================
    # Mouse target controller
    # ==============================================================

    class MouseTargetController:
        """
        Camera-relative mouse dragging for the target sphere.

        This preserves the behavior of hil14:
        - left click starts drag
        - mouse movement changes X/Y along camera-right direction
        - vertical mouse movement changes Z
        - releasing left click stops drag
        """

        def __init__(
            self,
            target_sphere_id: int,
            robot_base_position: Iterable[float],
            pixel_scale: float = 0.002,
            min_height_offset: float = 0.05,
        ):
            self.target_sphere_id = int(target_sphere_id)

            self.robot_base_position = np.asarray(
                robot_base_position,
                dtype=float,
            )

            self.pixel_scale = float(pixel_scale)
            self.min_height_offset = float(
                min_height_offset
            )

            self.drag_active = False
            self.drag_start_pos = np.zeros(3)
            self.drag_start_mouse = (0, 0)

        def update(self) -> Optional[np.ndarray]:
            """
            Process PyBullet mouse events.

            Returns:
                target position in RTDE frame when available,
                otherwise None.
            """
            if not p.isConnected():
                return None

            mouse_events = p.getMouseEvents()
            key_events = p.getKeyboardEvents()

            alt_down = (
                key_events.get(p.B3G_ALT, 0)
                in [1, 3]
            )

            ctrl_down = (
                key_events.get(p.B3G_CONTROL, 0)
                in [1, 3]
            )

            shift_down = (
                key_events.get(p.B3G_SHIFT, 0)
                in [1, 3]
            )

            modifier_pressed = (
                alt_down
                or ctrl_down
                or shift_down
            )

            for event in mouse_events:
                event_type = event[0]

                # --------------------------------------------------
                # Left mouse button
                # --------------------------------------------------
                if event_type == 2:
                    button_index = event[3]
                    button_state = event[4]

                    if button_index != 0:
                        continue

                    # Button pressed
                    if (
                        button_state == 3
                        and not modifier_pressed
                    ):
                        self.drag_active = True

                        curr_pos, _ = (
                            p.getBasePositionAndOrientation(
                                self.target_sphere_id
                            )
                        )

                        self.drag_start_pos = np.asarray(
                            curr_pos,
                            dtype=float,
                        )

                        self.drag_start_mouse = (
                            event[1],
                            event[2],
                        )

                    # Button released
                    elif button_state == 4:
                        self.drag_active = False

                # --------------------------------------------------
                # Mouse movement
                # --------------------------------------------------
                elif (
                    event_type == 1
                    and self.drag_active
                ):
                    mouse_x = event[1]
                    mouse_y = event[2]

                    dx_pixels = (
                        mouse_x
                        - self.drag_start_mouse[0]
                    )

                    dy_pixels = (
                        mouse_y
                        - self.drag_start_mouse[1]
                    )

                    cam_info = p.getDebugVisualizerCamera()

                    view_matrix = cam_info[2]

                    right_x = view_matrix[0]
                    right_y = view_matrix[4]

                    norm = np.sqrt(
                        right_x**2
                        + right_y**2
                    ) + 1e-6

                    right_x /= norm
                    right_y /= norm

                    delta_x = (
                        dx_pixels
                        * self.pixel_scale
                        * right_x
                    )

                    delta_y = (
                        dx_pixels
                        * self.pixel_scale
                        * right_y
                    )

                    delta_z = (
                        -dy_pixels
                        * self.pixel_scale
                    )

                    new_x = (
                        self.drag_start_pos[0]
                        + delta_x
                    )

                    new_y = (
                        self.drag_start_pos[1]
                        + delta_y
                    )

                    new_z = max(
                        self.robot_base_position[2]
                        + self.min_height_offset,
                        self.drag_start_pos[2]
                        + delta_z,
                    )

                    p.resetBasePositionAndOrientation(
                        self.target_sphere_id,
                        [
                            float(new_x),
                            float(new_y),
                            float(new_z),
                        ],
                        [0.0, 0.0, 0.0, 1.0],
                    )

            # ------------------------------------------------------
            # Read target sphere and convert to RTDE
            # ------------------------------------------------------
            sphere_pos_world, _ = (
                p.getBasePositionAndOrientation(
                    self.target_sphere_id
                )
            )

            sphere_pos_world = np.asarray(
                sphere_pos_world,
                dtype=float,
            )

            sphere_pos_rtde = (
                sphere_pos_world
                - self.robot_base_position
            ) * np.array(
                [-1.0, -1.0, 1.0]
            )

            return sphere_pos_rtde