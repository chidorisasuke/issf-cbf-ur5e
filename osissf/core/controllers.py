import numpy as np

from .dynamics import PinocchioDynamicsCalculator


class OperationalSpaceController:
    def __init__(self, dynamics, Kp, Kd, dt=0.008):
        self.dynamics = dynamics
        self.Kp = Kp
        self.Kd = Kd
        self.dt = dt

    def compute_task_force(
        self, x, v, x_des, v_des,
    ):
        e_x = x_des - x
        e_v = v_des - v
        return self.Kp @ e_x + self.Kd @ e_v

    def compute_task_torque(
        self, q, x, v, x_des, v_des,
    ):
        J = self.dynamics.get_jacobian(q)
        F = self.compute_task_force(
            x, v, x_des, v_des
        )

        tau_task = J.T @ F
        return tau_task

    def compute(
        self, q, dq, x, v, x_des, v_des,
    ):
        # Task-space position and velocity error
        e_x = x_des - x
        e_v = v_des - v

        # Cartesian PD force
        f_cmd = self.Kp @ e_x + self.Kd @ e_v

        # Task torque
        J = self.dynamics.get_jacobian(q)
        tau_task = J.T @ f_cmd

        # Coriolis + gravity compensation
        tau_coriolis_grav = self.dynamics.get_coriolis_gravity(q, dq)

        # Total nominal torque
        tau_total = tau_task + tau_coriolis_grav

        # Joint acceleration
        M = self.dynamics.get_mass_matrix(q)
        q_ddot = np.linalg.solve(M, tau_total)

        # Joint velocity command
        dq_cmd = dq + q_ddot * self.dt

        return {
            "tau_task": tau_task,
            "tau_coriolis_grav": tau_coriolis_grav,
            "tau_total": tau_total,
            "q_ddot": q_ddot,
            "dq_cmd": dq_cmd,
        }

class JointSpaceApproach:
    def __init__(self, dynamics, gain, dt):
        self.dynamics = dynamics
        self.gain = gain
        self.dt = dt

    def compute(
        self,
        q,
        dq,
        x,
        v,
        x_des,
        v_des,
        xdd_des,
        Kp,
        Kd,
    ):
        # Jacobian
        J = self.dynamics.get_jacobian(q)

        # Error
        e_pos = x_des - x
        e_vel = v_des - v

        # Operational-space inertia
        M = self.dynamics.get_mass_matrix(q)
        M_inv = np.linalg.inv(M)

        A = J @ M_inv @ J.T

        try:
            Lambda = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            Lambda = np.linalg.pinv(A)

        # Cartesian force
        F_cmd = Lambda @ (xdd_des + Kd @ e_vel + Kp @ e_pos)

        # Task torque
        tau_task = J.T @ F_cmd

        # Joint acceleration approximation
        ddq = self.gain * tau_task

        # Integrate joint velocity
        dq_cmd = dq + ddq * self.dt

        return {
            "F_cmd": F_cmd,
            "tau_task": tau_task,
            "ddq": ddq,
            "dq_cmd": dq_cmd,
        }

class TaskSpaceApproach:
    def __init__(self, dynamics, gain, dt):
        self.dynamics = dynamics
        self.gain = gain
        self.dt = dt

    def compute(
        self,
        q,
        dq,
        x,
        v,
        x_des,
        v_des,
        xdd_des,
        Kp,
        Kd,
        dx_prev,
    ):
        # Jacobian
        J = self.dynamics.get_jacobian(q)

        # Position and velocity error
        e_pos = x_des - x
        e_vel = v_des - v

        # Operational-space inertia
        M = self.dynamics.get_mass_matrix(q)
        M_inv = np.linalg.inv(M)

        A = J @ M_inv @ J.T

        try:
            Lambda = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            Lambda = np.linalg.pinv(A)

        # Cartesian force
        F_cmd = Lambda @ (xdd_des + Kd @ e_vel + Kp @ e_pos)

        # Cartesian acceleration
        xdd = self.gain * F_cmd

        # Integrate Cartesian velocity
        dx_curr = dx_prev + xdd * self.dt

        # Convert Cartesian velocity to joint velocity
        dq_cmd = np.linalg.pinv(J) @ dx_curr

        return {
            "F_cmd": F_cmd,
            "xdd": xdd,
            "dx_curr": dx_curr,
            "dq_cmd": dq_cmd,
        }