import numpy as np


class AdmittanceController:

    def __init__(
        self,
        Ma,
        Ba,
        Ka,
        dt,
        max_velocity=0.4,
        alpha_filter=0.05,
        reset_distance=0.03,
        reset_velocity_scale=0.8,
    ):
        self.Ma = np.asarray(Ma, dtype=float)
        self.Ba = np.asarray(Ba, dtype=float)
        self.Ka = np.asarray(Ka, dtype=float)

        self.dt = dt

        self.max_velocity = max_velocity
        self.alpha_filter = alpha_filter

        self.reset_distance = reset_distance
        self.reset_velocity_scale = reset_velocity_scale

        # Virtual mass states
        self.pa = np.zeros(3)
        self.pa_dot = np.zeros(3)
        self.pa_ddot = np.zeros(3)

        # Previous filtered command
        self.v_cmd_prev = np.zeros(3)

    def reset(self, x_initial):
        """
        Sinkronisasi state virtual admittance
        dengan posisi aktual end-effector.
        """
        self.pa = np.asarray(
            x_initial,
            dtype=float,
        ).copy()

        self.pa_dot = np.zeros(3)
        self.pa_ddot = np.zeros(3)

        self.v_cmd_prev = np.zeros(3)

    def compute_force_command(
        self,
        F_cmd,
        F_external,
        x_des,
    ):
        """
        Menghitung percepatan massa virtual:

            Ma * pa_ddot =
                F_cmd
                + F_external
                - Ba * pa_dot
                - Ka * (pa - x_des)
        """
        F_cmd = np.asarray(F_cmd, dtype=float)
        F_external = np.asarray(F_external, dtype=float)
        x_des = np.asarray(x_des, dtype=float)

        rhs = (F_cmd + F_external - self.Ba @ self.pa_dot - self.Ka @ (self.pa - x_des))
        self.pa_ddot = np.linalg.solve(
            self.Ma, rhs,
        )
        return self.pa_ddot, rhs

    def integrate(self):
        """
        Integrasi state virtual admittance.
        """
        self.pa_dot += (
            self.pa_ddot * self.dt
        )

        self.pa += (
            self.pa_dot * self.dt
        )

        return self.pa, self.pa_dot

    def synchronize(
        self,
        x_actual,
    ):
        """
        Reset state virtual jika terlalu jauh
        dari posisi aktual robot.
        """
        distance = np.linalg.norm(
            self.pa - x_actual
        )

        if distance > self.reset_distance:
            self.pa = np.asarray(
                x_actual,
                dtype=float,
            ).copy()
            self.pa_dot *= self.reset_velocity_scale
            return True
        return False

    def filter_velocity(
        self,
        v_cmd,
    ):
        """
        Clip velocity dan low-pass filter.
        """
        v_cmd = np.clip(v_cmd, -self.max_velocity, self.max_velocity,)
        v_cmd = ((1.0 - self.alpha_filter) * self.v_cmd_prev + self.alpha_filter * v_cmd)
        self.v_cmd_prev = v_cmd.copy()
        # Sinkronkan state internal dengan
        # velocity command hasil filter.
        self.pa_dot = v_cmd.copy()
        return v_cmd

    def compute(
        self,
        F_cmd,
        F_external,
        x_des,
        x_actual,
    ):
        """
        Satu langkah lengkap admittance control.
        Returns:
            pa
            pa_dot
            pa_ddot
            v_cmd
            rhs
            reset
        """

        # 1. Virtual mass acceleration
        pa_ddot, rhs = self.compute_force_command(
            F_cmd=F_cmd,
            F_external=F_external,
            x_des=x_des,
        )

        # 2. Integrate virtual mass state
        pa, pa_dot = self.integrate()

        # 3. Synchronize if virtual state drifts too far
        was_reset = self.synchronize(
            x_actual
        )

        # 4. Convert pa_dot to velocity command
        v_cmd = self.filter_velocity(
            pa_dot
        )

        return {
            "pa": pa.copy(),
            "pa_dot": pa_dot.copy(),
            "pa_ddot": pa_ddot.copy(),
            "v_cmd": v_cmd.copy(),
            "rhs": rhs.copy(),
            "reset": was_reset,
        }