import time
from dataclasses import dataclass
import numpy as np
import rtde_control
import rtde_receive

@dataclass
class RobotState:
    q: np.ndarray
    dq: np.ndarray
    tcp_pose: np.ndarray
    tcp_speed: np.ndarray

    @property
    def position(self):
        return self.tcp_pose[:3]

    @property
    def velocity(self):
        return self.tcp_speed[:3]

class UR5eRTDE:
    """
    Interface komunikasi UR5e melalui RTDE.
    Class ini hanya menangani:
    - koneksi
    - pembacaan state
    - inverse kinematics
    - motion command
    - stop/disconnect
    Tidak menangani:
    - CBF
    - controller
    - trajectory
    - PyBullet
    - logging
    """

    def __init__(
        self,
        robot_ip,
        mode="real",
    ):
        self.robot_ip = robot_ip
        self.mode = mode
        self.rtde_c = None
        self.rtde_r = None

    # =========================================================
    # CONNECTION
    # =========================================================
    def connect(self):
        """
        Connect ke UR5e melalui RTDE.
        """
        print(
            f"\n🔌 Menghubungkan ke Robot "
            f"({self.robot_ip})..."
        )
        try:
            self.rtde_c = (rtde_control.RTDEControlInterface(self.robot_ip))
            self.rtde_r = (rtde_receive.RTDEReceiveInterface(self.robot_ip))
            print("✅ Koneksi RTDE berhasil.")

        except Exception as exc:
            self.rtde_c = None
            self.rtde_r = None
            raise RuntimeError(
                f"Gagal menghubungkan ke UR5e: {exc}"
            ) from exc

    def is_connected(self):
        if self.rtde_c is None:
            return False
        try:
            return bool(
                self.rtde_c.isConnected()
            )
        except Exception:
            return False

    # =========================================================
    # STATE FEEDBACK
    # =========================================================

    def get_q(self):
        self._check_receive()
        return np.asarray(
            self.rtde_r.getActualQ(),
            dtype=float,
        )

    def get_dq(self):
        self._check_receive()
        return np.asarray(
            self.rtde_r.getActualQd(),
            dtype=float,
        )

    def get_tcp_pose(self):
        self._check_receive()
        return np.asarray(
            self.rtde_r.getActualTCPPose(),
            dtype=float,
        )

    def get_tcp_position(self):
        return self.get_tcp_pose()[:3]

    def get_tcp_speed(self):
        self._check_receive()
        return np.asarray(
            self.rtde_r.getActualTCPSpeed(),
            dtype=float,
        )

    def get_tcp_velocity(self):
        return self.get_tcp_speed()[:3]

    def get_actual_force(self):
        self._check_receive()
        return np.asarray(
            self.rtde_r.getActualTCPForce(),
            dtype=float,
        )

    def get_state(self):
        """
        Membaca seluruh state yang diperlukan
        control loop dalam satu pemanggilan.
        """
        self._check_receive()
        q = np.asarray(
            self.rtde_r.getActualQ(),
            dtype=float,
        )

        dq = np.asarray(
            self.rtde_r.getActualQd(),
            dtype=float,
        )

        tcp_pose = np.asarray(
            self.rtde_r.getActualTCPPose(),
            dtype=float,
        )

        tcp_speed = np.asarray(
            self.rtde_r.getActualTCPSpeed(),
            dtype=float,
        )

        return RobotState(
            q=q,
            dq=dq,
            tcp_pose=tcp_pose,
            tcp_speed=tcp_speed,
        )

    # =========================================================
    # KINEMATICS / MOTION
    # =========================================================

    def get_inverse_kinematics(
        self,
        tcp_pose,
    ):
        """
        Mendapatkan konfigurasi joint dari target TCP pose.
        """
        self._check_control()
        tcp_pose = list(tcp_pose)
        return np.asarray(
            self.rtde_c.getInverseKinematics(
                tcp_pose
            ),
            dtype=float,
        )

    # =========================================================
    # ASYNCHRONOUS MOTION
    # =========================================================

    def move_j(
        self,
        q_target,
        speed=0.2,
        acceleration=0.4,
        asynchronous=True,
    ):
        if self.mode != "real":
            return

        self._check_control()

        q_target = np.asarray(
            q_target,
            dtype=float,
        ).tolist()

        return self.rtde_c.moveJ(
            q_target,
            speed,
            acceleration,
            asynchronous,
        )

    def move_l(
        self,
        target,
        speed=0.5,
        accel=0.5,
        asynchronous=True,
    ):
        self._check_control()

        return self.rtde_c.moveL(
            list(target),
            speed,
            accel,
            asynchronous,
        )

    # =========================================================
    # VELOCITY COMMAND
    # =========================================================

    def speed_j(
        self,
        dq_cmd,
        acceleration,
        dt,
    ):
        """
        Kirim joint velocity command menggunakan speedJ.
        """
        self._check_control()
        dq_cmd = np.asarray(
            dq_cmd,
            dtype=float,
        )

        return self.rtde_c.speedJ(
            dq_cmd.tolist(),
            acceleration,
            dt,
        )

    def servo_l(
        self,
        command,
        lookahead_time=0.5,
        gain=0.1,
    ):
        """
        Kirim Cartesian servo command menggunakan servoL.
        """
        self._check_control()
        command = np.asarray(
            command,
            dtype=float,
        )

        return self.rtde_c.servoL(
            command.tolist(),
            lookahead_time,
            gain,
        )

    # =========================================================
    # SYNCHRONOUS MOVE
    # =========================================================

    def move_and_wait(
        self,
        target,
        mode="joint",
        speed=0.5,
        accel=0.5,
        threshold=0.01,
        poll_period=0.01,
    ):
        """
        Menjalankan moveJ/moveL secara asynchronous lalu
        menunggu sampai target tercapai.

        Tidak mengetahui PyBullet.
        Sinkronisasi visual dilakukan oleh simulation layer.
        """
        target = np.asarray(
            target,
            dtype=float,
        )

        if mode == "joint":
            self.move_j(
                target,
                speed,
                accel,
                asynchronous=True,
            )

        elif mode == "pose":
            self.move_l(
                target,
                speed,
                accel,
                asynchronous=True,
            )

        else:
            raise ValueError(
                f"Unknown motion mode: {mode}"
            )

        while True:
            state = self.get_state()
            if mode == "joint":
                distance = np.linalg.norm(
                    state.q - target
                )

            else:
                distance = np.linalg.norm(
                    state.position
                    - target[:3]
                )

            if distance < threshold:
                break

            time.sleep(
                poll_period
            )

        return self.get_state()

    # =========================================================
    # WATCHDOG (proteksi jika Python hang / crash)
    # =========================================================

    def enable_watchdog(
        self,
        min_frequency=10.0,
    ):
        """
        ur_rtde watchdog: jika kick_watchdog() tidak dipanggil
        minimal min_frequency Hz, control script di controller
        UR menghentikan robot sendiri, tanpa bergantung pada Python.
        Aktifkan TEPAT sebelum control loop (setelah JIT warm-up).
        """
        if self.rtde_c is None:
            return False
        try:
            self.rtde_c.setWatchdog(float(min_frequency))
            print(
                f"🐕 RTDE watchdog aktif (min {min_frequency:.0f} Hz)."
            )
            return True
        except AttributeError:
            print(
                "⚠️ Versi ur_rtde tidak mendukung setWatchdog(). "
                "Pertimbangkan upgrade ur_rtde."
            )
        except Exception as exc:
            print(f"⚠️ Gagal mengaktifkan watchdog: {exc}")
        return False

    def kick_watchdog(self):
        if self.rtde_c is None:
            return
        try:
            self.rtde_c.kickWatchdog()
        except Exception:
            pass

    # =========================================================
    # STOP
    # =========================================================

    def speed_stop(
        self,
        acceleration=10.0,
    ):
        """
        WAJIB setelah speedJ. speedJ pada ur_rtde terus
        mengeksekusi kecepatan TERAKHIR sampai speedStop()
        dipanggil, walaupun Python sudah berhenti mengirim.
        """
        if self.rtde_c is None:
            return
        try:
            self.rtde_c.speedStop(acceleration)
        except Exception as exc:
            print(f"⚠️ speedStop gagal: {exc}")

    def wait_until_still(
        self,
        timeout=3.0,
        velocity_tolerance=1e-3,
    ):
        """
        Verifikasi robot benar-benar diam (max |dq| < tol).
        """
        if self.rtde_r is None:
            return False
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                dq = np.asarray(self.rtde_r.getActualQd(), dtype=float)
            except Exception:
                return False
            if np.max(np.abs(dq)) < velocity_tolerance:
                return True
            time.sleep(0.01)
        return False

    def stop_j(
        self,
        acceleration=0.5,
    ):
        if not self.is_connected():
            return

        return self.rtde_c.stopJ(
            acceleration
        )

    def stop_l(
        self,
        acceleration=0.5,
    ):
        if not self.is_connected():
            return

        return self.rtde_c.stopL(
            acceleration
        )

    def stop_script(self):
        if not self.is_connected():
            return

        try:
            return self.rtde_c.stopScript()
        except Exception:
            return None

    # =========================================================
    # SAFE SHUTDOWN / DISCONNECT
    # =========================================================

    def safe_stop(self):
        """
        Hentikan gerakan SEGERA dan verifikasi robot diam.
        Dipanggil langsung saat control loop selesai / error /
        Ctrl+C, SEBELUM simpan log & plotting.
        """
        if self.rtde_c is None:
            return
        self.speed_stop(10.0)
        try:
            self.rtde_c.stopJ(2.0)
        except Exception:
            pass
        still = self.wait_until_still()
        if still:
            print("🛑 Robot berhenti (|dq| ≈ 0).")
        else:
            print(
                "🚨 PERINGATAN: robot belum terverifikasi diam! "
                "Siapkan E-STOP."
            )

    def disconnect(self):
        """
        Urutan aman: speedStop -> stopScript -> disconnect.
        stopScript membuat program di controller berhenti bersih
        (tidak meninggalkan status 'Running' / error program).
        """
        if self.rtde_c is not None:
            self.speed_stop(10.0)
            try:
                self.rtde_c.stopScript()
            except Exception:
                pass
            try:
                self.rtde_c.disconnect()
            except Exception:
                pass

        if self.rtde_r is not None:
            try:
                self.rtde_r.disconnect()
            except Exception:
                pass

        self.rtde_c = None
        self.rtde_r = None

    # =========================================================
    # INTERNAL CHECKS
    # =========================================================

    def _check_control(self):
        if self.rtde_c is None:
            raise RuntimeError(
                "RTDE control interface belum terhubung."
            )

    def _check_receive(self):
        if self.rtde_r is None:
            raise RuntimeError(
                "RTDE receive interface belum terhubung."
            )