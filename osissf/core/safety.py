import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cbfpy import CBF

from oscbf.core.manipulator import Manipulator
from oscbf.core.oscbf_configs import OSCBFVelocityConfig


# ============================================================
# STANDARD CBF
# ============================================================

@jax.tree_util.register_static
class StandardCBF(CBF):
    """
    Standard Control Barrier Function.

    Tidak mengubah formulasi CBF dari cbfpy.
    Class ini dibuat sebagai interface eksplisit agar
    Standard CBF dan ISSf-CBF terpisah secara kode.
    """

    pass


# ============================================================
# ISSf-CBF
# ============================================================

@jax.tree_util.register_static
class ISSfCBF(CBF):
    """
    Input-to-State Safe Control Barrier Function.

    Standard CBF constraint:

        L_f h + L_g h u >= -alpha(h)

    atau dalam bentuk QP:

        -L_g h u <= alpha(h) + L_f h

    ISSf-CBF menambahkan robustification term:

        L_f h + L_g h u >= -alpha(h) + ||L_g h||^2 epsilon(h)

    sehingga:

        -L_g h u <= alpha(h) + L_f h - ||L_g h||^2 epsilon(h)

    Formulasi ini mengikuti implementasi ISSfCBF pada hil14.
    """

    def __init__(
        self,
        n,
        m,
        num_cbf,
        u_min,
        u_max,
        control_constrained,
        relax_cbf,
        cbf_relaxation_penalty,
        h_1,
        h_2,
        f,
        g,
        alpha,
        alpha_2,
        P,
        q,
        solver_tol,
        epsilon_0=0.5,
        issf_threshold=0.20,
    ):
        super().__init__(
            n=n,
            m=m,
            num_cbf=num_cbf,
            u_min=u_min,
            u_max=u_max,
            control_constrained=control_constrained,
            relax_cbf=relax_cbf,
            cbf_relaxation_penalty=cbf_relaxation_penalty,
            h_1=h_1,
            h_2=h_2,
            f=f,
            g=g,
            alpha=alpha,
            alpha_2=alpha_2,
            P=P,
            q=q,
            solver_tol=solver_tol,
        )

        self.epsilon_0 = float(epsilon_0)
        self.issf_threshold = float(issf_threshold)

    def h_qp(
        self,
        z,
        u_des,
        *args,
        **kwargs,
    ):
        """
        ISSf-CBF QP upper bound.

        Standard:
            h_standard = alpha(h) + L_f h

        ISSf:
            h_issf = h_standard
                     - ||L_g h||^2 epsilon(h)

        epsilon(h) menggunakan activation linear:

            activation =
                clip(
                    1 - h / threshold,
                    0,
                    1
                )

        sehingga ISSf aktif terutama ketika barrier
        mendekati boundary.
        """

        # ----------------------------------------------------
        # Barrier and Lie derivative
        # ----------------------------------------------------
        hz, lfh = self.h_and_Lfh(
            z,
            *args,
            **kwargs,
        )

        h_standard = self.alpha(hz) + lfh

        # ----------------------------------------------------
        # L_g h
        # ----------------------------------------------------
        Lgh = self.Lgh(
            z,
            *args,
            **kwargs,
        )

        # ||L_g h_i||^2
        Lgh_norm_sq = jnp.sum(
            Lgh ** 2,
            axis=1,
        )

        # ----------------------------------------------------
        # ISSf activation
        # ----------------------------------------------------
        activation = jnp.clip(
            1.0
            - (
                hz
                / self.issf_threshold
            ),
            0.0,
            1.0,
        )

        # epsilon(h)
        issf_term = (
            Lgh_norm_sq
            * self.epsilon_0
            * activation
        )

        # ----------------------------------------------------
        # ISSf-CBF constraint
        # ----------------------------------------------------
        h_cbf = (
            h_standard
            - issf_term
        )

        # ----------------------------------------------------
        # Control constraints
        # ----------------------------------------------------
        if self.control_constrained:
            return jnp.concatenate(
                [
                    h_cbf,
                    jnp.asarray(self.u_max),
                    -jnp.asarray(self.u_min),
                ]
            )

        return h_cbf

    @classmethod
    def from_config(
        cls,
        config,
    ):
        """
        Construct ISSf-CBF from a CBFConfig.

        Tetap menggunakan API CBF.from_config()
        dari cbfpy versi yang sedang digunakan.
        """

        instance = super().from_config(config)

        instance.epsilon_0 = float(
            getattr(
                config,
                "epsilon_0",
                0.5,
            )
        )

        instance.issf_threshold = float(
            getattr(
                config,
                "issf_threshold",
                0.20,
            )
        )

        return instance


# ============================================================
# CONTAINMENT CBF CONFIGURATION
# ============================================================

@jax.tree_util.register_static
class ContainmentVelocityConfig(OSCBFVelocityConfig):
    """
    CBF untuk menjaga end-effector tetap berada
    di dalam containment box.
    """

    def __init__(
        self,
        robot: Manipulator,
        pos_min_rtde: ArrayLike,
        pos_max_rtde: ArrayLike,
        epsilon_0: float = 0.2,
        issf_threshold: float = 0.20,
        alpha_c: float = 20.0,
    ):
        self.pos_min_rtde = jnp.asarray(
            pos_min_rtde
        )

        self.pos_max_rtde = jnp.asarray(
            pos_max_rtde
        )

        self.epsilon_0 = float(
            epsilon_0
        )

        self.issf_threshold = float(
            issf_threshold
        )

        self.alpha_c = float(
            alpha_c
        )

        super().__init__(
            robot,
            joint_obj_weight=0.3,
        )

    def h_1(
        self,
        z,
        **kwargs,
    ):
        q = z[: self.num_joints]

        ee_pos_pino = self.robot.ee_position(q)

        ee_pos_rtde = (
            ee_pos_pino
            * jnp.array(
                [-1.0, -1.0, 1.0]
            )
        )

        return jnp.concatenate(
            [
                self.pos_max_rtde
                - ee_pos_rtde,

                ee_pos_rtde
                - self.pos_min_rtde,
            ]
        )

    def alpha(self, h):
        return self.alpha_c * h


# ============================================================
# OBSTACLE CBF CONFIGURATION
# ============================================================

@jax.tree_util.register_static
class ObstacleVelocityConfig(OSCBFVelocityConfig):
    """
    CBF untuk obstacle avoidance berbasis
    posisi end-effector.
    """

    def __init__(
        self,
        robot: Manipulator,
        obstacle_centers,
        obstacle_safe_dist: float = 0.3,
        epsilon_0: float = 0.2,
        issf_threshold: float = 0.20,
        alpha_c: float = 20.0,
    ):
        self.obstacle_centers = [
            jnp.asarray(c)
            for c in obstacle_centers
        ]

        self.num_obstacles = len(
            obstacle_centers
        )

        self.obstacle_safe_dist = float(
            obstacle_safe_dist
        )

        self.epsilon_0 = float(
            epsilon_0
        )

        self.issf_threshold = float(
            issf_threshold
        )

        self.alpha_c = float(
            alpha_c
        )

        super().__init__(
            robot,
            joint_obj_weight=0.3,
        )

    def h_1(
        self,
        z,
        **kwargs,
    ):
        q = z[: self.num_joints]

        ee_pos_pino = self.robot.ee_position(q)

        ee_pos_rtde = (
            ee_pos_pino
            * jnp.array(
                [-1.0, -1.0, 1.0]
            )
        )

        h_obs = []

        for center in self.obstacle_centers:
            dist = jnp.linalg.norm(
                ee_pos_rtde - center
            )

            h_obs.append(
                dist
                - self.obstacle_safe_dist
            )

        return jnp.array(h_obs)

    def alpha(self, h):
        return self.alpha_c * h


# ============================================================
# BOTH:
# EE CONTAINMENT + OBSTACLE + WHOLE-BODY
# ============================================================

@jax.tree_util.register_static
class BothVelocityConfig(OSCBFVelocityConfig):
    """
    CBF untuk:

    1. EE containment
    2. Robot-vs-obstacle collision avoidance
    3. Whole-body containment
    """

    def __init__(
        self,
        robot: Manipulator,
        pos_min_rtde: ArrayLike,
        pos_max_rtde: ArrayLike,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
        wb_pos_min_rtde: ArrayLike,
        wb_pos_max_rtde: ArrayLike,
        epsilon_0: float = 0.2,
        issf_threshold: float = 0.20,
        alpha_c: float = 20.0,
    ):
        self.pos_min_rtde = jnp.asarray(
            pos_min_rtde
        )

        self.pos_max_rtde = jnp.asarray(
            pos_max_rtde
        )

        self.wb_pos_min_rtde = jnp.asarray(
            wb_pos_min_rtde
        )

        self.wb_pos_max_rtde = jnp.asarray(
            wb_pos_max_rtde
        )

        self.epsilon_0 = float(
            epsilon_0
        )

        self.issf_threshold = float(
            issf_threshold
        )

        self.alpha_c = float(
            alpha_c
        )

        self.collision_positions = (
            jnp.atleast_2d(
                collision_positions
            )
        )

        self.collision_radii = (
            jnp.ravel(
                collision_radii
            )
        )

        super().__init__(
            robot,
            joint_obj_weight=0.3,
        )

    def h_1(
        self,
        z,
        **kwargs,
    ):
        q = z[: self.num_joints]

        # =================================================
        # 1. END-EFFECTOR CONTAINMENT
        # =================================================

        ee_pos_pino = self.robot.ee_position(q)

        ee_pos_rtde = (
            ee_pos_pino
            * jnp.array(
                [-1.0, -1.0, 1.0]
            )
        )

        h_containment = jnp.concatenate(
            [
                self.pos_max_rtde
                - ee_pos_rtde,

                ee_pos_rtde
                - self.pos_min_rtde,
            ]
        )

        # =================================================
        # 2. WHOLE ROBOT COLLISION GEOMETRY
        # =================================================

        robot_collision_pos_rad = (
            self.robot.link_collision_data(q)
        )

        robot_collision_positions_pino = (
            robot_collision_pos_rad[:, :3]
        )

        robot_collision_radii = (
            robot_collision_pos_rad[:, 3, None]
        )

        robot_collision_positions_rtde = (
            robot_collision_positions_pino
            * jnp.array(
                [-1.0, -1.0, 1.0]
            )
        )

        robot_num_pts = (
            robot_collision_positions_rtde.shape[0]
        )

        # =================================================
        # 3. ROBOT VS OBSTACLE
        # =================================================

        center_deltas = (
            robot_collision_positions_rtde[:, None, :]
            - self.collision_positions[None, :, :]
        ).reshape(-1, 3)

        radii_sums = (
            robot_collision_radii[:, None]
            + self.collision_radii[None, :]
        ).reshape(-1)

        h_collision = (
            jnp.linalg.norm(
                center_deltas,
                axis=1,
            )
            - radii_sums
        )

        # =================================================
        # 4. WHOLE-BODY CONTAINMENT
        # =================================================

        h_whole_body_upper = (
            jnp.tile(
                self.wb_pos_max_rtde,
                (robot_num_pts, 1),
            )
            - robot_collision_positions_rtde
            - robot_collision_radii
        ).ravel()

        h_whole_body_lower = (
            robot_collision_positions_rtde
            - jnp.tile(
                self.wb_pos_min_rtde,
                (robot_num_pts, 1),
            )
            - robot_collision_radii
        ).ravel()

        return jnp.concatenate(
            [
                h_containment,
                h_collision,
                h_whole_body_upper,
                h_whole_body_lower,
            ]
        )

    def alpha(self, h):
        return self.alpha_c * h


# Backward compatibility
EESafeSetVelocityConfig = BothVelocityConfig


# ============================================================
# SAFETY FILTER WRAPPER
# ============================================================

class SafetyFilter:

    def __init__(
        self,
        config,
        use_issf=False,
    ):
        self.config = config
        self.use_issf = use_issf

        if use_issf:
            self.cbf = ISSfCBF.from_config(
                config
            )
        else:
            self.cbf = StandardCBF.from_config(
                config
            )

        # -----------------------------------------------------
        # JIT (sama seperti hil14: compute_safe_control_jit)
        # -----------------------------------------------------
        cbf = self.cbf

        @jax.jit
        def _safe_control(q_jnp, dq_jnp):
            return cbf.safety_filter(q_jnp, dq_jnp)

        @jax.jit
        def _h(q_jnp):
            return cbf.h(q_jnp)

        self._safe_control_jit = _safe_control
        self._h_jit = _h
        self.is_warm = False

    def warmup(self, q=None):
        """
        Kompilasi JIT SEBELUM control loop dimulai (hil14:
        'WARM-UP JAX JIT'). Tanpa ini, panggilan pertama di dalam
        loop memakan ~1-2 detik sehingga t melompat dari 0.00
        ke ~1.67 s dan robot tidak menerima speedJ selama itu.
        """
        import time
        import numpy as np

        if q is None:
            q = np.zeros(6)
        q = np.asarray(q, dtype=float)
        dq = np.zeros_like(q)

        print("\n🔥 Kompilasi JIT CBF (warm-up)... mohon tunggu...")
        t0 = time.perf_counter()
        jax.block_until_ready(self._h_jit(jnp.asarray(q)))
        jax.block_until_ready(
            self._safe_control_jit(jnp.asarray(q), jnp.asarray(dq))
        )
        t1 = time.perf_counter()
        # panggilan kedua untuk mengukur waktu per-iterasi
        jax.block_until_ready(
            self._safe_control_jit(jnp.asarray(q), jnp.asarray(dq))
        )
        t2 = time.perf_counter()
        self.is_warm = True
        print(
            f"✅ JIT selesai: compile {1e3*(t1-t0):.0f} ms, "
            f"per-call {1e3*(t2-t1):.2f} ms"
        )

    @property
    def robot(self):
        return self.config.robot

    @property
    def rtde_min(self):
        return getattr(
            self.config,
            "pos_min_rtde",
            None,
        )

    @property
    def rtde_max(self):
        return getattr(
            self.config,
            "pos_max_rtde",
            None,
        )

    @property
    def wb_rtde_min(self):
        return getattr(
            self.config,
            "wb_pos_min_rtde",
            None,
        )

    @property
    def wb_rtde_max(self):
        return getattr(
            self.config,
            "wb_pos_max_rtde",
            None,
        )

    def filter(
        self,
        q,
        dq_cmd,
    ):
        """
        Apply selected safety filter.

        q:
            Joint position.

        dq_cmd:
            Nominal joint velocity.

        Returns:
            Safe joint velocity.
        """

        q = jnp.asarray(q)
        dq_cmd = jnp.asarray(dq_cmd)

        return self._safe_control_jit(
            q,
            dq_cmd,
        )

    def h(self, q):
        """
        Evaluate all active barrier functions.
        """
        q = jnp.asarray(q)

        return self._h_jit(q)

    def qp_data(
        self,
        q,
        dq_cmd,
    ):
        """
        Expose QP data untuk debugging / logging.
        """
        q = jnp.asarray(q)
        dq_cmd = jnp.asarray(dq_cmd)

        return self.cbf.qp_data(
            q,
            dq_cmd,
        )