import numpy as np


class TrajectoryGenerator:
    """
    Generator trajectory Cartesian 3-DoF untuk UR5e.
    Output utama:
        x_d      : desired position      (3,)
        xd_d     : desired velocity      (3,)
        xdd_d    : desired acceleration  (3,)
    """

    def __init__(
        self,
        trajectory,
        amplitude=0.32,
        omega=-2.0 * np.pi * 0.07,
        tilt_angle=np.pi / 6.0,
        move_duration=5.0,
        ramp_duration=2.0,
        table_height=0.59,
        trajectory_offset=None,
    ):
        self.trajectory = trajectory
        self.amplitude = amplitude
        self.omega = omega

        self.tilt_angle = tilt_angle
        self.c_tilt = np.cos(tilt_angle)
        self.s_tilt = np.sin(tilt_angle)

        self.move_duration = move_duration
        self.ramp_duration = ramp_duration
        self.table_height = table_height

        # Digunakan oleh visualisasi waypoint/PAP
        self.x_viz = None

        if trajectory_offset is None:
            trajectory_offset = np.zeros(3)

        self.trajectory_offset = np.asarray(
            trajectory_offset,
            dtype=float,
        )

    # =========================================================
    # COMMON UTILITIES
    # =========================================================

    @staticmethod
    def smooth_move(
        start_p, end_p, t_local, duration,
    ):
        """
        Gerakan smooth menggunakan cubic polynomial:
            s = 3τ² - 2τ³
        sehingga posisi, velocity, dan acceleration
        memiliki zero boundary velocity/acceleration.
        """
        duration = float(duration)

        t_local = max(
            0.0,
            min(t_local, duration),
        )
        tau = t_local / duration
        s = (
            3.0 * tau**2 - 2.0 * tau**3
        )

        if t_local >= duration:
            s_dot = 0.0
            s_ddot = 0.0
        else:
            s_dot = (6.0 * tau - 6.0 * tau**2) / duration
            s_ddot = (6.0 - 12.0 * tau) / (duration**2)

        delta = (np.asarray(end_p) - np.asarray(start_p))
        position = (np.asarray(start_p) + delta * s)
        velocity = delta * s_dot
        acceleration = delta * s_ddot

        return (
            position,
            velocity,
            acceleration,
        )

    # =========================================================
    # BASIC SINUSOIDAL TRAJECTORIES
    # =========================================================

    def calculate_full_speed(
        self, t, x_base,
    ):
        """
        Full-speed trajectory tanpa ramp-up.
        """
        A = self.amplitude
        omega = self.omega

        # -----------------------------------------------------
        # Linear X
        # -----------------------------------------------------
        if self.trajectory == "linear_x":
            offset = np.array(
                [
                    A * np.sin(omega * t),
                    0.0,
                    0.0,
                ]
            )

            x_d = x_base + offset
            xd_d = np.array(
                [
                    A * omega * np.cos(omega * t),
                    0.0,
                    0.0,
                ]
            )
            xdd_d = np.array(
                [
                    -A * omega**2 * np.sin(omega * t),
                    0.0,
                    0.0,
                ]
            )
            return x_d, xd_d, xdd_d

        # -----------------------------------------------------
        # Linear Y
        # -----------------------------------------------------
        elif self.trajectory == "linear_y":
            offset = np.array(
                [
                    0.0,
                    A * np.sin(omega * t),
                    0.0,
                ]
            )
            x_d = x_base + offset
            xd_d = np.array(
                [
                    0.0,
                    A * omega * np.cos(omega * t),
                    0.0,
                ]
            )
            xdd_d = np.array(
                [
                    0.0,
                    -A * omega**2 * np.sin(omega * t),
                    0.0,
                ]
            )
            return x_d, xd_d, xdd_d

        # -----------------------------------------------------
        # 2D Circle
        # -----------------------------------------------------
        elif self.trajectory == "circle":
            offset = np.array(
                [
                    A * np.cos(omega * t),
                    A * np.sin(omega * t),
                    0.0,
                ]
            )
            x_d = x_base + offset
            xd_d = np.array(
                [
                    -A * omega * np.sin(omega * t),
                    A * omega * np.cos(omega * t),
                    0.0,
                ]
            )

            xdd_d = np.array(
                [
                    -A * omega**2 * np.cos(omega * t),
                    -A * omega**2 * np.sin(omega * t),
                    0.0,
                ]
            )
            return x_d, xd_d, xdd_d

        # -----------------------------------------------------
        # 3D Circle
        # -----------------------------------------------------

        elif self.trajectory == "circle_3d":
            off_x = (A * np.cos(omega * t))
            off_y_flat = (A * np.sin(omega * t))
            off_y = (off_y_flat * self.c_tilt)
            off_z = (off_y_flat * self.s_tilt)
            x_d = (x_base + np.array([
                        off_x,
                        off_y,
                        off_z,
                    ]
                )
            )

            vel_x = (-A * omega * np.sin(omega * t))
            vel_y_flat = (A * omega * np.cos(omega * t))
            xd_d = np.array(
                [
                    vel_x,
                    vel_y_flat * self.c_tilt,
                    vel_y_flat * self.s_tilt,
                ]
            )
            acc_x = (-A * omega**2 * np.cos(omega * t))
            acc_y_flat = (-A * omega**2 * np.sin(omega * t))
            xdd_d = np.array(
                [
                    acc_x,
                    acc_y_flat * self.c_tilt,
                    acc_y_flat * self.s_tilt,
                ]
            )

            return x_d, xd_d, xdd_d

        # -----------------------------------------------------
        # Obstacles trajectory
        # -----------------------------------------------------

        elif self.trajectory == "obstacles":
            table_height = self.table_height
            p_start = np.array([ 0.20, 0.0, table_height + 0.25, ])
            p_orange = np.array([ 0.613, -0.080, table_height + 0.10, ])
            p_purple = np.array([0.300, -0.366, table_height + 0.10, ])
            T_move = 6.0
            T_pause = 3.0

            if t <= T_move:
                return self.smooth_move(
                    p_start, p_orange, t, T_move,
                )

            elif t <= T_move + T_pause:
                return (
                    p_orange, np.zeros(3), np.zeros(3),
                )

            elif t <= (2.0 * T_move + T_pause):
                t_local = (t - T_move - T_pause)
                return self.smooth_move(
                    p_orange, p_purple, t_local, T_move,
                )

            else:
                return (
                    p_purple, np.zeros(3), np.zeros(3),
                )

        return (
            np.zeros(3), np.zeros(3), np.zeros(3),
        )

    # =========================================================
    # WAYPOINT TIMING
    # =========================================================

    @staticmethod
    def get_wp_timing():

        seq_names = [
            ("P1", 3.0),
            ("P2", 5.0),
            ("P3", 6.0),
            ("P4", 5.0),
            ("P3", 6.0),
            ("P5", 7.0),
            ("P6", 3.0),
            ("P5", 3.0),
            ("P7", 6.0),
            ("P8", 5.0),
            ("P0", 6.0),
        ]

        events = [
            (0.0, "P0")
        ]

        accumulated_time = 0.0

        for name, duration in seq_names:
            accumulated_time += duration
            events.append(
                (
                    accumulated_time,
                    name,
                )
            )

        cycle_time = accumulated_time

        return (
            seq_names,
            events,
            cycle_time,
        )

    # =========================================================
    # PAP / PICK-AND-PLACE TIMING
    # =========================================================

    @staticmethod
    def get_pap_timing():

        t_pendek = 3.0
        t_panjang = 6.0
        t_dip = 3.0
        t_pause = 1.0

        t_extra = 1.0

        t1 = (t_pendek + t_extra)
        t2 = (t1 + t_pendek + t_extra)
        t3 = (t2 + t_pendek  + t_extra)
        t4 = (t3 + t_dip + t_extra)
        t5 = (t4 + t_pause + t_extra)
        t6 = (t5 + t_dip + t_extra)
        t7 = (t6 + t_panjang + t_extra)
        t8 = (t7 + t_dip + t_extra)
        t9 = (t8 + t_pause + t_extra)
        t10 = (t9 + t_dip + t_extra)
        t11 = (t10 + t_panjang + t_extra)
        t12 = (t11 + t_panjang + t_extra)
        t13 = (t12 + t_panjang + t_extra)
        cycle_time = t13 + 1.0

        return (
            t_pendek, t_panjang, t_dip, t_pause, t1, t2, t3,
            t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, cycle_time
        )

    # =========================================================
    # WP TRAJECTORY
    # =========================================================

    def waypoint_trajectory(
        self, t,
    ):

        points = {
            "P0": np.array([-0.300, 0.000, 0.500]), "P1": np.array([-0.500, 0.000, 0.500]),
            "P2": np.array([-0.325, -0.309, 0.500]), "P3": np.array([-0.419, -0.660, 0.500]),
            "P4": np.array([-0.419, -0.660, 0.120]), "P5": np.array([-0.816, -0.084, 0.350]),
            "P6": np.array([-0.816, -0.084, 0.120]), "P7": np.array([-0.725, 0.250, 0.350]),
            "P8": np.array([-0.356, 0.632, 0.350])}

        seq_names, _, cycle_time = (
            self.get_wp_timing()
        )
        t_mod = (
            t % cycle_time
        )

        accumulated_time = 0.0
        current_target = points["P0"]

        for name, duration in seq_names:
            accumulated_time += duration
            if t_mod <= accumulated_time:
                current_target = points[
                    name
                ]
                break

        self.x_viz = current_target

        return (
            current_target.copy(),
            np.zeros(3),
            np.zeros(3),
        )

    # =========================================================
    # PAP3 TRAJECTORY
    # =========================================================

    def pap_trajectory(
        self, t,
    ):

        p0 = np.array([
            -0.300,
            0.000,
            0.500,
        ], dtype=float)

        p1 = np.array([
            -0.500,
            0.000,
            0.500,
        ], dtype=float)

        p2 = np.array([
            -0.325,
            -0.309,
            0.500,
        ], dtype=float)

        p3 = np.array([
            -0.419,
            -0.660,
            0.500,
        ], dtype=float)

        p4 = np.array([
            -0.419,
            -0.660,
            0.120,
        ], dtype=float)

        p5 = np.array([
            -0.816,
            -0.084,
            0.350,
        ], dtype=float)

        p6 = np.array([
            -0.816,
            -0.084,
            0.120,
        ], dtype=float)

        p7 = np.array([
            -0.725,
            0.250,
            0.350,
        ], dtype=float)

        p8 = np.array([
            -0.356,
            0.632,
            0.350,
        ], dtype=float)

        (
            t_pendek, t_panjang, t_dip, t_pause, t1, t2, t3, t4,
            t5, t6, t7, t8, t9, t10, t11, t12, t13, cycle_time
        ) = self.get_pap_timing()

        t_mod = (t % cycle_time)
        segments = [
            (0, t1, p0, p1, t_pendek),
            (t1 + t_pause, t2, p1, p2, t_pendek),
            (t2 + t_pause, t3, p2, p3, t_pendek),
            (t3 + t_pause, t4, p3, p4, t_dip),
            (t5, t6, p4, p3, t_dip),
            (t6 + t_pause, t7, p3, p5, t_panjang),
            (t7 + t_pause, t8, p5, p6, t_dip),
            (t9, t10, p6, p5, t_dip),
            (t10 + t_pause, t11, p5, p7, t_panjang),
            (t11 + t_pause, t12, p7, p8, t_panjang),
            (t12 + t_pause, t13, p8, p0, t_panjang)]

        x_smooth = p0
        xd_smooth = np.zeros(3)
        xdd_smooth = np.zeros(3)
        self.x_viz = p0

        for i, (
            t_start, t_arrive, p_start, p_end, duration,
        ) in enumerate(segments):
            if t_mod < t_start:
                x_smooth = p_start
                xd_smooth = np.zeros(3)
                xdd_smooth = np.zeros(3)
                self.x_viz = p_end
                break

            elif (
                t_start <= t_mod <= t_arrive
            ):
                x_smooth, xd_smooth, xdd_smooth = (
                    self.smooth_move(
                        p_start, p_end,
                        t_mod - t_start,
                        duration,
                    )
                )

                self.x_viz = p_end
                break

            elif (
                t_arrive < t_mod <
                (
                    segments[i + 1][0]
                    if i + 1 < len(segments)
                    else cycle_time
                )
            ):

                x_smooth = p_end
                xd_smooth = np.zeros(3)
                xdd_smooth = np.zeros(3)
                self.x_viz = (
                    segments[i + 1][3]
                    if i + 1 < len(segments)
                    else segments[0][3]
                )

                break

        return (
            x_smooth, xd_smooth, xdd_smooth,
        )

    # =========================================================
    # POINT TRAJECTORY
    # =========================================================

    def point_trajectory(
        self, t, x_base,
    ):
        target_offset_1, target_offset_2 = np.array([0.15, -0.08, 0.0]), np.array([0.08, 0.15, 0.0])
        T_MOVE = self.move_duration
        T_STAY = 3.0
        if t <= T_MOVE:
            tau = (t / T_MOVE)
            s = (3.0 * tau**2 - 2.0 * tau**3)
            v_scale = (6.0 * tau - 6.0 * tau**2) / T_MOVE
            a_scale = (6.0 - 12.0 * tau) / (T_MOVE**2)
            x_d = (x_base + target_offset_1 * s)
            xd_d = (target_offset_1 * v_scale)
            xdd_d = (target_offset_1 * a_scale)

        elif t <= (
            T_MOVE + T_STAY
        ):
            x_d = (x_base + target_offset_1)
            xd_d = np.zeros(3)
            xdd_d = np.zeros(3)

        elif t <= (
            2.0 * T_MOVE + T_STAY
        ):
            t_local = (t - T_MOVE - T_STAY)
            tau = (t_local / T_MOVE)
            s = (3.0 * tau**2 - 2.0 * tau**3)
            v_scale = (6.0 * tau - 6.0 * tau**2) / T_MOVE
            a_scale = (6.0 - 12.0 * tau) / (T_MOVE**2)
            start_pos = (x_base + target_offset_1)
            x_d = (start_pos + target_offset_2 * s)
            xd_d = (target_offset_2 * v_scale)
            xdd_d = (target_offset_2 * a_scale)

        else:
            x_d = (x_base + target_offset_1 + target_offset_2)
            xd_d = np.zeros(3)
            xdd_d = np.zeros(3)

        return x_d, xd_d, xdd_d

    # =========================================================
    # MAIN TRAJECTORY INTERFACE
    # =========================================================

    def compute(
        self, t, x_base_pos,
    ):
        """
        Main trajectory interface.
        Returns:
            x_d
            xd_d
            xdd_d
        """
        self.x_viz = None
        x_base = np.asarray(
            x_base_pos,
            dtype=float,
        ) + self.trajectory_offset

        if self.trajectory == "wp":
            return self.waypoint_trajectory(t)

        if self.trajectory in [
            "pap2", "pap3",
        ]:
            return self.pap_trajectory(t)

        if self.trajectory == "point":
            return self.point_trajectory(
                t, x_base,
            )
        # Sinusoidal trajectories:
        # linear_x, linear_y,
        # circle, circle_3d,
        # obstacles

        if (self.trajectory == "obstacles"):
            return self.calculate_full_speed(
                t, x_base,
            )

        # -----------------------------------------------------
        # Two-phase ramp-up
        # -----------------------------------------------------

        if t >= self.ramp_duration:
            return self.calculate_full_speed(
                t, x_base,
            )
        tau = (t / self.ramp_duration)
        s = 0.5 * (1.0 - np.cos(np.pi * tau))  # Cosine ramp-up
        x_d_full, xd_d_full, xdd_d_full = (self.calculate_full_speed(t, x_base,))
        # Preserve hil14:
        # position follows full trajectory,
        # velocity and acceleration are scaled
        x_d = x_d_full
        xd_d = (xd_d_full * s)
        xdd_d = (xdd_d_full * s)
        return (
            x_d, xd_d, xdd_d,
        )

    # =========================================================
    # BASE POSITION
    # =========================================================

    @staticmethod
    def get_base_position_from_start(
        self,
        x_start,
    ):
        x_start = np.asarray(
            x_start,
            dtype=float,
        )

        if self.trajectory in (
            "circle",
            "circle_3d",
        ):
            return x_start - np.array(
                [self.amplitude, 0.0, 0.0]
            )

        return x_start.copy()