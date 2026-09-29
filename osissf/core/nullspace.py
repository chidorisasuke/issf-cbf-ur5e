import numpy as np
import pinocchio as pin

from .dynamics import PinocchioDynamicsCalculator


def manipulability(J):
    """
    Menghitung manipulability scalar μ(q) = sqrt(det(J J^T))
    J: Jacobian translasi 3x6
    """
    A = J @ J.T  # 3x3
    detA = np.linalg.det(A)
    # clamp supaya gak negatif karena error numerik
    detA = max(detA, 1e-12)
    return np.sqrt(detA)


def compute_singularity_torque(q, dyn, k_sing=1.0, eps=1e-4):
    """
    Bangun torsi Null Space untuk menghindari singularity:
    Gamma_sing = k_sing * ∂μ/∂q
    """
    n = q.shape[0]
    grad_mu = np.zeros(n)

    for i in range(n):
        dq = np.zeros_like(q)
        dq[i] = eps
        Jp = dyn.get_jacobian(q + dq)
        Jm = dyn.get_jacobian(q - dq)
        mu_p = manipulability(Jp)
        mu_m = manipulability(Jm)
        grad_mu[i] = (mu_p - mu_m) / (2.0 * eps)

    Gamma_sing = k_sing * grad_mu  # shape (6,)
    return Gamma_sing


class SimpleObstacle:
    def __init__(self, center, safe_dist, k_obs):
        self.center = np.asarray(center).reshape(3,)
        self.safe_dist = safe_dist
        self.k_obs = k_obs

    def get_point_and_jacobian(self, q, dyn):
        pin.forwardKinematics(dyn.model, dyn.data, q)
        pin.updateFramePlacements(dyn.model, dyn.data)
        oMf = dyn.data.oMf[dyn.ee_frame_id]  # SE3
        p = oMf.translation
        Jp = dyn.get_jacobian(q)
        return p, Jp


def compute_obstacle_torque(q, dyn, obstacles):
    Gamma_obs = np.zeros_like(q)

    for obs in obstacles:
        p, Jp = obs.get_point_and_jacobian(q, dyn)
        diff = p - obs.center
        d = np.linalg.norm(diff) + 1e-12

        if d >= obs.safe_dist:
            continue

        term = 1.0 / d - 1.0 / obs.safe_dist
        dU_dd = obs.k_obs * term * (-1.0 / (d**2))
        dd_dq = (diff / d) @ Jp
        gradU = dU_dd * dd_dq
        Gamma_obs -= gradU

    return Gamma_obs


Q_HOME_DEFAULT = np.array(
    [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
    dtype=float,
)


class NullspaceController:
    """
    Velocity-level null-space controller (identik dengan hil14,
    cabang admittance/JSA/TSA dengan --nullspace on):

        ramp_null   = min(t / 2, 1)
        dq_null_ref = Kp_null (q_home - q) * ramp_null - Kd_null dq
        dq_null_ref = clip(dq_null_ref, ±min(0.5 + t, 5))
        dq_null     = (I - J^+ J) dq_null_ref
    """

    def __init__(
        self,
        Kp_null,
        Kd_null,
        k_singularity=0.0,
        q_home=None,
        ramp_time=2.0,
    ):
        self.Kp_null = np.asarray(Kp_null, dtype=float)
        self.Kd_null = np.asarray(Kd_null, dtype=float)
        self.k_singularity = float(k_singularity)
        self.q_home = (
            Q_HOME_DEFAULT.copy()
            if q_home is None
            else np.asarray(q_home, dtype=float)
        )
        self.ramp_time = float(ramp_time)

        # untuk logging (hil14: log_q_null / log_dq_null)
        self.last_q_error = np.zeros(6)
        self.last_dq_null = np.zeros(6)

    def compute(
        self,
        q,
        dq,
        J,
        t=0.0,
    ):
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        J = np.asarray(J, dtype=float)

        # 1. Posture error
        q_error_null = self.q_home - q

        # 2. Posture velocity reference + ramp (hil14)
        ramp_null = min(t / self.ramp_time, 1.0)
        dq_null_ref = (
            self.Kp_null @ q_error_null * ramp_null
            - self.Kd_null @ dq
        )

        # 3. Dynamic clip (hil14)
        current_clip = min(0.5 + t, 5.0)
        dq_null_ref = np.clip(
            dq_null_ref,
            -current_clip,
            current_clip,
        )

        # 4. Null-space projection
        J_pinv = np.linalg.pinv(J)
        N = np.eye(6) - J_pinv @ J

        dq_null = N @ dq_null_ref

        self.last_q_error = q_error_null.copy()
        self.last_dq_null = dq_null.copy()

        return dq_null