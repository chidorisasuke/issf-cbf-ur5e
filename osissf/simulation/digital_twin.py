import numpy as np
import pybullet as p


class DigitalTwin:
    """
    Digital Twin UR5e berbasis PyBullet.

    Mode REAL (HIL) -- metode hil14, "UR5e/URSim adalah chief":
        Setiap siklus kontrol k:
          1. Twin di-teleport ke state chief q_real(k), dq_real(k).
          2. Twin menerima perintah yang SAMA dengan robot
             (speedJ -> VELOCITY_CONTROL, force 150 Nm) lalu
             stepSimulation() satu kali (dt = 0.008 s).
             Hasilnya adalah PREDIKSI twin: q_pred(k+1).
          3. Twin di-teleport lagi ke q_real(k) sehingga yang
             ditampilkan & dipakai visualisasi = state chief.

        Karena twin punya dinamika sendiri (motor PyBullet, gravitasi,
        batas gaya, integrasi 8 ms) sedangkan UR5e punya controller
        internal + latency speedJ, q_pred(k+1) TIDAK akan persis sama
        dengan q_real(k+1). Selisih ini:

            residual(k+1) = q_real(k+1) - q_pred(k+1)

        adalah "gap" digital twin yang sebenarnya (one-step prediction
        error). Pada hil14 gap ini tidak pernah terlihat karena
        q_pybullet_actual ditimpa q_real.copy() sehingga SimGap = 0.

    Mode SIM:
        PyBullet adalah plant. Tidak ada teleport; twin hanya
        menerima velocity command lalu step.
    """

    def __init__(
        self,
        environment,
        robot_cbf=None,
        max_force=150.0,
    ):
        self.environment = environment
        self.robot_cbf = robot_cbf
        self.max_force = float(max_force)

        # Prediksi twin untuk sample berikutnya (dibuat di siklus k-1)
        self._q_pred_next = None
        self._dq_pred_next = None

    # =========================================================
    # LOW LEVEL: TELEPORT (COPY STATE DARI CHIEF)
    # =========================================================

    def sync_from_robot(
        self,
        q,
        dq,
        update_geometry=True,
    ):
        """
        Teleport PyBullet ke state UR5e aktual (hil14: resetJointState).
        """
        if not p.isConnected():
            return

        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)

        for i, joint_idx in enumerate(
            self.environment.sim_joint_indices
        ):
            p.resetJointState(
                bodyUniqueId=int(self.environment.robot_id),
                jointIndex=int(joint_idx),
                targetValue=float(q[i]),
                targetVelocity=float(dq[i]),
            )

        if update_geometry:
            self.update_collision_geometry(q)

    def reset_prediction(self):
        """
        Dipanggil sebelum control loop agar residual sampel pertama
        tidak dibandingkan dengan prediksi dari fase homing.
        """
        self._q_pred_next = None
        self._dq_pred_next = None

    # =========================================================
    # HIL STEP (METODE hil14 + PENGUKURAN GAP)
    # =========================================================

    def hil_step(
        self,
        q_real,
        dq_real,
        dq_cmd_sent,
    ):
        """
        Satu siklus digital twin pada mode real.

        Parameters
        ----------
        q_real, dq_real : state UR5e yang dibaca di awal siklus k
        dq_cmd_sent     : perintah speedJ yang BENAR-BENAR dikirim
                          ke robot pada siklus k (setelah CBF + clip)

        Returns
        -------
        dict:
            q_sync      : state twin yang ditampilkan (= q_real)
            q_pred_now  : prediksi twin untuk sample k (dibuat di k-1)
            residual    : q_real(k) - q_pred_now   (gap twin)
            q_pred_next : prediksi twin untuk sample k+1
        """
        q_real = np.asarray(q_real, dtype=float)
        dq_real = np.asarray(dq_real, dtype=float)

        if not p.isConnected():
            zeros = np.zeros_like(q_real)
            return {
                "q_sync": zeros,
                "dq_sync": zeros,
                "q_pred_now": zeros,
                "residual": zeros,
                "q_pred_next": zeros,
            }

        # ---- gap: bandingkan chief sekarang dengan prediksi lama ----
        if self._q_pred_next is None:
            q_pred_now = q_real.copy()
        else:
            q_pred_now = self._q_pred_next.copy()
        residual = q_real - q_pred_now

        # ---- 1. teleport ke chief ----
        self.sync_from_robot(q_real, dq_real, update_geometry=False)

        # ---- 2. perintah yang sama dengan robot + step fisika ----
        self.environment.apply_joint_velocity(
            dq_cmd_sent,
            max_force=self.max_force,
        )
        self.environment.step()

        q_pred, dq_pred = self.environment.get_joint_state()
        self._q_pred_next = np.asarray(q_pred, dtype=float).copy()
        self._dq_pred_next = np.asarray(dq_pred, dtype=float).copy()

        # ---- 3. teleport kembali ke chief (yang ditampilkan) ----
        self.sync_from_robot(q_real, dq_real)

        return {
            "q_sync": q_real.copy(),
            "dq_sync": dq_real.copy(),
            "q_pred_now": q_pred_now,
            "residual": residual,
            "q_pred_next": self._q_pred_next.copy(),
        }

    # =========================================================
    # TRACK ASYNC MOTION (hil14: move_and_sync)
    # =========================================================

    def follow(
        self,
        q,
        dq,
    ):
        """
        Dipakai saat homing / moveJ ke titik awal:
        cukup teleport tiap polling (100 Hz), seperti move_and_sync hil14.
        """
        self.sync_from_robot(q, dq)

    # =========================================================
    # COLLISION GEOMETRY
    # =========================================================

    def update_collision_geometry(
        self,
        q,
    ):
        if self.robot_cbf is None:
            return

        collision_data = np.asarray(
            self.robot_cbf.link_collision_data(q)
        )

        self.environment.update_robot_collision_spheres(
            collision_data
        )

    # =========================================================
    # SIM MODE (PyBullet sebagai plant)
    # =========================================================

    def command_velocity(
        self,
        dq_cmd,
    ):
        self.environment.apply_joint_velocity(
            dq_cmd,
            max_force=self.max_force,
        )

    def step(self):
        self.environment.step()

    def get_state(self):
        return self.environment.get_joint_state()

    # =========================================================
    # TARGET / EE
    # =========================================================

    def update_target(self, x_des):
        self.environment.update_target(x_des)

    def update_actual_ee(self, x_actual):
        self.environment.update_actual_ee(x_actual)

    def get_ee_position_world(self):
        """
        Posisi link terakhir PyBullet (world frame), seperti
        pos_ee_pyb pada hil14.
        """
        if not p.isConnected():
            return np.zeros(3)
        ee_link_idx = self.environment.sim_joint_indices[-1]
        state = p.getLinkState(
            int(self.environment.robot_id),
            int(ee_link_idx),
        )
        if state is None:
            return np.zeros(3)
        return np.asarray(state[0], dtype=float)